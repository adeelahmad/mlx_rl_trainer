import asyncio
import logging
import time
from typing import Any, Dict, List
import mlx.core as mx
from mlx_lm.utils import load
from rich import print as rprint
from mlx_lm.models import cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)

class AsyncBatchGenerator:
    def __init__(self, model: Any, tokenizer: Any, max_batch_size: int = 32, batch_timeout: float = 0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.batching_task = asyncio.create_task(self._batching_loop())
        logger.info(f"[AsyncGenerator] Started with max_batch_size={max_batch_size} and timeout={batch_timeout}s.")

    async def generate(self, prompt: str, max_tokens: int = 50) -> str:
        future = asyncio.Future()
        request = {"prompt": prompt, "max_tokens": max_tokens, "future": future}
        await self.queue.put(request)
        return await future

    async def _batching_loop(self):
        while not self.shutdown_event.is_set():
            batch = []
            try:
                first_request = await asyncio.wait_for(self.queue.get(), timeout=self.batch_timeout)
                batch.append(first_request)
                while len(batch) < self.max_batch_size and not self.queue.empty():
                    batch.append(self.queue.get_nowait())
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if batch:
                logger.info(f"[AsyncGenerator] Processing a batch of size {len(batch)}.")
                await asyncio.to_thread(self._process_batch, batch)

    def _process_batch(self, batch: List[Dict]):
        prompts = [item['prompt'] for item in batch]
        max_tokens = max(item['max_tokens'] for item in batch)
        try:
            results = self._batched_generate(prompts, max_tokens)
            for i, item in enumerate(batch):
                item['future'].set_result(results[i])
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            for item in batch:
                item['future'].set_exception(e)

    def _batched_generate(self, prompts: List[str], max_tokens: int) -> List[str]:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        prompt_tokens = [self.tokenizer.encode(p) for p in prompts]
        max_len = max(len(tokens) for tokens in prompt_tokens)
        padded_tokens = [[self.tokenizer.pad_token_id] * (max_len - len(tokens)) + tokens for tokens in prompt_tokens]
        prompt_mx = mx.array(padded_tokens)
        
        self.model.eval()
        kv_cache = cache.TreeCache(self.model.n_kv_heads, self.model.head_dim, self.model.n_layers)
        logits = self.model(prompt_mx, cache=kv_cache)
        y = logits[:, -1, :]
        y = mx.argmax(y, axis=-1)
        generated = [y]

        for _ in range(max_tokens - 1):
            logits = self.model(y[:, None], cache=kv_cache)
            y = logits[:, -1, :]
            y = mx.argmax(y, axis=-1)
            generated.append(y)

        mx.eval(generated)
        result_tokens = mx.stack(generated, axis=1).tolist()
        return self.tokenizer.batch_decode(result_tokens, skip_special_tokens=True)

    async def shutdown(self):
        logger.info("[AsyncGenerator] Shutdown requested.")
        self.shutdown_event.set()
        self.batching_task.cancel()
        try:
            await self.batching_task
        except asyncio.CancelledError:
            pass
        logger.info("[AsyncGenerator] Shutdown complete.")

async def run_async_inference_server(config: ExperimentConfig):
    rprint('\n[bold yellow]--- Running in Async Inference Server Mode ---[/]')
    try:
        model, tokenizer = load(str(config.model.model_path))
    except Exception as e:
        logger.critical(f"Failed to load model/tokenizer from {config.model.model_path}: {e}", exc_info=True)
        return

    generator = AsyncBatchGenerator(model=model, tokenizer=tokenizer, max_batch_size=16, batch_timeout=0.1)
    
    sample_prompts = [
        'Explain the theory of relativity in simple terms.', 'Write a short poem about the moon.',
        'What is the capital of Mongolia?', "Translate 'hello world' to French.",
        'List three benefits of using MLX.', "Who wrote 'The Hobbit'?",
        'What is the recipe for a margarita?', 'Explain how a CPU works.'
    ]
    
    start_time = time.time()
    
    async def client_request(client_id, prompt):
        logger.info(f"[Client {client_id}] Sending request...")
        result = await generator.generate(prompt, max_tokens=50)
        logger.info(f"[Client {client_id}] Got result: '{result[:100].strip()}...'")

    tasks = [asyncio.create_task(client_request(i, p)) for i, p in enumerate(sample_prompts)]
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    logger.info(f"\nTotal time for {len(sample_prompts)} requests: {end_time - start_time:.2f} seconds.")
    
    await generator.shutdown()
