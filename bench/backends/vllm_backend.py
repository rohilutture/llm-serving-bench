"""vLLM offline engine: paged KV cache + continuous batching.

Options exercised by the experiment matrix:
- quantized checkpoints (W4A16 GPTQ/AWQ, FP8) via the model path / `quantization`
- serving the LoRA adapter directly (enable_lora) vs. a merged checkpoint
- n-gram speculative decoding (prompt lookup). Text-to-SQL is a good fit: most
  output tokens are table/column names copied from the schema in the prompt, so
  cheap n-gram drafts get accepted often.
- prefix caching: every request for the same database shares the schema prefix.
"""
from __future__ import annotations

from vllm import LLM, SamplingParams

from ..prompts import render_chat
from .base import BackendInfo, Generation


class VLLMBackend:
    def __init__(self, model: str, adapter: str | None = None,
                 quantization: str | None = None, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85, speculative_ngram: int = 0,
                 prefix_caching: bool = True, max_num_seqs: int = 256,
                 kv_cache_dtype: str = "auto"):
        kw = dict(model=model, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_memory_utilization,
                  enable_prefix_caching=prefix_caching, max_num_seqs=max_num_seqs,
                  kv_cache_dtype=kv_cache_dtype, seed=0)
        if quantization:
            kw["quantization"] = quantization
        if adapter:
            kw.update(enable_lora=True, max_lora_rank=64)
        if speculative_ngram:
            kw["speculative_config"] = {"method": "ngram",
                                        "num_speculative_tokens": speculative_ngram,
                                        "prompt_lookup_max": 4}
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("sql", 1, adapter)
        self.info = BackendInfo("vllm", model, {
            "adapter": adapter, "quantization": quantization,
            "speculative_ngram": speculative_ngram, "prefix_caching": prefix_caching,
            "max_num_seqs": max_num_seqs, "kv_cache_dtype": kv_cache_dtype})

    def render(self, system, user):
        return render_chat(self.tok, system, user)

    def generate(self, prompts, max_new_tokens):
        sp = SamplingParams(temperature=0.0, repetition_penalty=1.0, max_tokens=max_new_tokens)  # pure greedy
        outs = self.llm.generate(prompts, sp, lora_request=self.lora, use_tqdm=False)
        return [Generation(o.outputs[0].text, len(o.prompt_token_ids),
                           len(o.outputs[0].token_ids)) for o in outs]

    def peak_memory_gb(self):
        # vLLM pre-allocates gpu_memory_utilization of VRAM for its KV pool, so
        # "peak memory" is a setting, not a measurement. Report weight size instead
        # (see report.py / model_size_gb).
        return None
