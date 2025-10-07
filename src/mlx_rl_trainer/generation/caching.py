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
    Manages KV cache memory in fixed-size blocks (paging), allowing non-contiguous
    sequences in memory and improving utilization during batching/rollouts.
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
            num_layers (int): The number of transformer layers in the model.
            num_kv_heads (int): The number of key/value heads in the model.
            head_dim (int): The dimension of each attention head.
            block_size (int): The number of tokens per KV cache block (page).
            num_blocks (int): The total number of blocks available in the cache pool.
            dtype (mx.Dtype): The data type for the KV cache (e.g., mx.bfloat16).
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
        mx.eval(
            self.key_cache, self.value_cache
        )  # Eagerly evaluate to allocate GPU memory

        # List of available block indices
        self.free_blocks: List[int] = list(range(num_blocks))

        # Mapping from sequence ID to a list of allocated physical block indices
        # Eg: {1001: [5, 12, 3], 1002: [1, 9]}
        self.sequence_map: Dict[int, List[int]] = {}

        # Current actual length of each sequence in tokens (logical length)
        self.sequence_lengths: Dict[int, int] = {}

        nbytes = self.key_cache.nbytes + self.value_cache.nbytes
        logger.info(
            f"[PagedKVCache] Allocated {nbytes / 1e9:.2f} GB total for keys and values."
        )

    def reset(self):
        """
        Clears all sequence mappings and returns all blocks to the free pool.
        This is crucial to call between independent batches or evaluation runs.
        """
        self.sequence_map.clear()
        self.sequence_lengths.clear()
        self.free_blocks = list(range(self.num_blocks))
        logger.debug(
            "[PagedKVCache] Cache state has been reset and all blocks are free."
        )

    def _allocate_block(self) -> int:
        """Pops a block index from the free pool."""
        if not self.free_blocks:
            raise MemoryError(
                "PagedKVCache is out of memory blocks. Increase `num_blocks` or reduce batch size."
            )
        block_idx = self.free_blocks.pop(0)
        logger.debug(f"Allocated block {block_idx}.")
        return block_idx

    def allocate_sequence(self, sequence_id: int, num_tokens: int):
        """
        Allocates enough blocks from the free pool for a new sequence.

        Args:
            sequence_id (int): A unique identifier for the sequence.
            num_tokens (int): The initial number of tokens this sequence will hold.

        Raises:
            ValueError: If the sequence ID is already in use.
            MemoryError: If there are not enough free blocks available.
        """
        if sequence_id in self.sequence_map:
            raise ValueError(
                f"Sequence ID {sequence_id} is already allocated. Free it first."
            )

        # Calculate number of blocks needed for initial tokens
        num_required_blocks = ceil(max(1, num_tokens) / self.block_size)

        if num_required_blocks > len(self.free_blocks):
            raise MemoryError(
                f"Not enough free blocks for sequence {sequence_id} (required: {num_required_blocks}, available: {len(self.free_blocks)})."
            )

        allocated_block_indices = [
            self._allocate_block() for _ in range(num_required_blocks)
        ]
        self.sequence_map[sequence_id] = allocated_block_indices
        self.sequence_lengths[sequence_id] = num_tokens
        logger.debug(
            f"Allocated {num_required_blocks} blocks {allocated_block_indices} for sequence {sequence_id} (initial length: {num_tokens})."
        )

    def free_sequence(self, sequence_id: int):
        """
        Frees all blocks associated with a given sequence ID, returning them to the free pool.

        Args:
            sequence_id (int): The unique identifier of the sequence to free.
        """
        if sequence_id not in self.sequence_map:
            logger.debug(
                f"Attempted to free unknown sequence ID {sequence_id}. No action taken."
            )
            return

        blocks_to_free = self.sequence_map.pop(sequence_id)
        self.sequence_lengths.pop(sequence_id, None)
        self.free_blocks.extend(blocks_to_free)
        logger.debug(f"Freed blocks {blocks_to_free} for sequence {sequence_id}.")

    def append_token(self, sequence_id: int):
        """
        Increments the recorded length of a sequence and allocates a new block
        if the new token exceeds the capacity of currently allocated blocks.

        Args:
            sequence_id (int): The unique identifier of the sequence.

        Raises:
            MemoryError: If a new block is required but none are free.
        """
        if sequence_id not in self.sequence_map:
            logger.warning(
                f"Appending to unallocated sequence ID {sequence_id}. Allocating new minimal sequence (1 block)."
            )
            self.allocate_sequence(
                sequence_id, 1
            )  # Allocate a block for this new sequence
            return

        current_logical_len = self.sequence_lengths.get(sequence_id, 0)
        current_allocated_blocks = len(self.sequence_map[sequence_id])

        # Check if the next token will overflow the last allocated block
        if current_logical_len + 1 > current_allocated_blocks * self.block_size:
            logger.debug(
                f"Sequence {sequence_id} at logical length {current_logical_len} needs a new block ({current_allocated_blocks} blocks currently allocated)."
            )
            new_block_idx = (
                self._allocate_block()
            )  # May raise MemoryError if no blocks are free
            self.sequence_map[sequence_id].append(new_block_idx)
            logger.debug(
                f"Allocated new block {new_block_idx} for sequence {sequence_id}."
            )

        self.sequence_lengths[sequence_id] += 1
        logger.debug(
            f"Sequence {sequence_id} logical length increased to {self.sequence_lengths[sequence_id]}."
        )

    def get_block_mapping_for_batch(
        self, sequence_ids: List[int]
    ) -> Dict[int, List[int]]:
        """
        Retrieves the mapping of sequence IDs to their allocated block indices for a batch.

        Args:
            sequence_ids (List[int]): A list of sequence IDs in the current batch.

        Returns:
            Dict[int, List[int]]: A dictionary where keys are sequence IDs and values are
                                  lists of block indices. An empty list is returned for
                                  any sequence ID not currently allocated.
        """
        return {sid: self.sequence_map.get(sid, []) for sid in sequence_ids}

    # NOTE ON MLX INTEGRATION:
    # While PagedKVCache manages the "physical" block allocation, the MLX model's layers
    # still need to be aware of this structure. This means the `mlx-lm` library's
    # `cache.TreeCache` (or equivalent) concept would be replaced or augmented.
    # Specifically, `nn.MultiHeadAttention` or internal KV cache lookups within
    # `mlx_lm.models.base.BaseModel` would need to be modified.
    # This involves a non-trivial modification to `mlx_lm` itself, or a custom wrapper
    # that intercepts KV cache operations. For this project, the `PagedKVCache` class
    # is fully implemented. The `Generator` class (in `generator.py`) would then be
    # responsible for passing this `PagedKVCache` instance and possibly adapting
    # the model's forward pass if custom MLX-LM hooks become available or if the model
    # is specifically modified to understand paged caches.
    # Currently, `mlx_lm`'s standard `cache.make_prompt_cache` (TreeCache) is used,
    # so `PagedKVCache` is a placeholder for a more advanced future integration.
    # The `allocate_sequence`/`free_sequence` calls in the Generator demonstrate
    # how it *would* be managed externally.
