# file_path: mlx_rl_trainer/src/mlx_rl_trainer/generation/caching.py
# revision_no: 001
# goals_of_writing_code_block: Paged KV Cache implementation for efficient memory management.
# type_of_code_response: add new code
"""Paged KV Cache implementation for efficient memory management."""

import logging
import math
from math import ceil
from typing import Dict, List, Any, Tuple
import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# --- MLX Global Config & Constants (re-declared for module independence) ---
TARGET_FLOAT_DTYPE = mx.bfloat16


class PagedKVCache:
    """
    A Paged Key-Value Cache for managing GPU memory efficiently.

    Instead of allocating a contiguous memory block for each sequence's KV cache,
    this class pre-allocates a large, paged memory pool. Each sequence is assigned
    a set of non-contiguous "pages" (blocks), and a mapping layer tracks which
    page and offset corresponds to each token. This prevents memory fragmentation
    and allows for much higher utilization in high-throughput environments.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 2048,
        dtype: mx.Dtype = TARGET_FLOAT_DTYPE,
    ):
        """
        Initializes the paged KV cache.

        Args:
            num_layers: Number of transformer layers in the model.
            num_kv_heads: Number of key-value heads in the model.
            head_dim: Dimension of each attention head.
            block_size: Number of tokens per cache block.
            num_blocks: Total number of blocks to pre-allocate in the cache pool.
            dtype: Data type for the cache tensors (e.g., mx.bfloat16).
        """
        logger.info(
            f"[PagedKVCache] Initializing with {num_blocks} blocks of size {block_size}..."
        )
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Shape of the cache storage: (num_blocks, num_layers, num_kv_heads, block_size, head_dim)
        cache_shape = (
            num_blocks,
            num_layers,
            num_kv_heads,
            block_size,
            head_dim,
        )
        self.key_cache = mx.zeros(cache_shape, dtype=dtype)
        self.value_cache = mx.zeros(cache_shape, dtype=dtype)
        mx.eval(self.key_cache, self.value_cache)  # Eagerly evaluate to allocate memory

        # Free blocks pool - contains indices of available blocks
        self.free_blocks: List[int] = list(range(num_blocks))

        # Mapping from sequence ID to a list of allocated block indices
        self.sequence_map: Dict[int, List[int]] = {}

        # Current length of each sequence (in tokens)
        self.sequence_lengths: Dict[int, int] = {}

        nbytes = self.key_cache.nbytes + self.value_cache.nbytes
        logger.info(
            f"[PagedKVCache] Allocated {nbytes / 1e9:.2f} GB total for keys and values."
        )

    def reset(self):
        """Clears all sequence mappings and returns all blocks to the free pool.
        This should be called between independent batches of sequences.
        """
        self.sequence_map.clear()
        self.sequence_lengths.clear()
        self.free_blocks = list(range(self.num_blocks))
        logger.debug("[PagedKVCache] Cache state has been reset.")

    def _allocate_block(self) -> int:
        """Pops a single block index from the free pool. Raises MemoryError if no blocks are available."""
        if not self.free_blocks:
            raise MemoryError("PagedKVCache is out of memory blocks.")
        return self.free_blocks.pop(0)

    def allocate_sequence(self, sequence_id: int, num_tokens: int):
        """
        Allocates the necessary number of blocks for a new sequence and maps them.

        Args:
            sequence_id: A unique integer ID for the new sequence.
            num_tokens: The initial number of tokens already present in the sequence (e.g., prompt length).

        Raises:
            ValueError: If `sequence_id` is already in use.
            MemoryError: If there are not enough free blocks.
        """
        if sequence_id in self.sequence_map:
            raise ValueError(f"Sequence ID {sequence_id} is already allocated.")

        # Calculate blocks needed (at least 1, even for 0 tokens initially)
        num_required_blocks = ceil(max(1, num_tokens) / self.block_size)

        if num_required_blocks > len(self.free_blocks):
            raise MemoryError(
                f"Not enough free blocks for sequence {sequence_id} (required: {num_required_blocks}, available: {len(self.free_blocks)})"
            )

        allocated = [self._allocate_block() for _ in range(num_required_blocks)]
        self.sequence_map[sequence_id] = allocated
        self.sequence_lengths[sequence_id] = num_tokens

    def free_sequence(self, sequence_id: int):
        """
        Frees all blocks associated with a given sequence ID and returns them to the free pool.

        Args:
            sequence_id: The unique ID of the sequence to free.
        """
        if sequence_id not in self.sequence_map:
            return  # Sequence not found, nothing to free

        blocks_to_free = self.sequence_map.pop(sequence_id)
        self.sequence_lengths.pop(sequence_id, None)
        self.free_blocks.extend(blocks_to_free)  # Return blocks to pool

    def append_token(self, sequence_id: int):
        """
        Increments the length of a sequence and allocates a new cache block if the new token
        exceeds the capacity of currently allocated blocks.

        Args:
            sequence_id: The unique ID of the sequence to append to.
        """
        if sequence_id not in self.sequence_map:
            # If sequence was not allocated (e.g., started with no tokens), allocate for 1st token
            self.allocate_sequence(sequence_id, 1)
            return

        current_len = self.sequence_lengths.get(sequence_id, 0)

        # Check if the next token requires a new block
        if (current_len + 1) > len(self.sequence_map[sequence_id]) * self.block_size:
            new_block = self._allocate_block()
            self.sequence_map[sequence_id].append(new_block)

        self.sequence_lengths[sequence_id] += 1

    def get_block_mapping_for_batch(
        self, sequence_ids: List[int]
    ) -> Dict[int, List[int]]:
        """
        Retrieves the block indices for a given batch of sequence IDs.

        Args:
            sequence_ids: A list of unique IDs for sequences in the current batch.

        Returns:
            A dictionary mapping each `sequence_id` to its list of allocated block indices.
        """
        return {sid: self.sequence_map.get(sid, []) for sid in sequence_ids}

    # NOTE: The full integration of PagedKVCache with MLX's MultiHeadAttention
    # involves custom cache logic within the model's forward pass. This class
    # provides the memory management, but the model's attention layers need
    # to be adapted to read/write from these paged blocks using the mappings
    # provided by `get_block_mapping_for_batch`. This typically requires
    # deeper modifications to the `mlx-lm` model architecture.
