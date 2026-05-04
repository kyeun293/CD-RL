"""vLLM sampling wrapper used for both policy rollouts and (optionally) the RPD judge."""

from typing import List, Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class VLLMSampler:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        seed: int = 42,
        dtype: str = "bfloat16",
        max_model_len: Optional[int] = None,
    ):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            seed=seed,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )

    def _format(self, problem: str) -> str:
        # Match verl's math eval convention: chat template + minimal instruction.
        messages = [
            {"role": "user",
             "content": problem + "\n\nPlease reason step by step, and put your "
                        "final answer within \\boxed{}."}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def sample(
        self,
        prompts: List[str],
        n: int = 16,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 16384,
    ) -> List[List[str]]:
        formatted = [self._format(p) for p in prompts]
        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(formatted, params)
        # vLLM preserves input order.
        return [[o.text for o in out.outputs] for out in outputs]

    def chat(
        self,
        messages_batch: List[List[dict]],
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
    ) -> List[str]:
        """Single-turn chat used by the RPD judge for step extraction."""
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        params = SamplingParams(
            n=1, temperature=temperature, top_p=1.0, max_tokens=max_new_tokens
        )
        outs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outs]