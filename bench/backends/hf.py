"""Baseline: plain Hugging Face transformers generate() with static batching.

This is the "before" in the comparison. Each batch is left-padded to its longest
prompt and decodes until its longest output finishes, so short requests wait on
long ones and padding tokens burn compute. vLLM's continuous batching and paged
KV cache exist to remove exactly that waste.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..prompts import render_chat
from .base import BackendInfo, Generation


class HFBackend:
    def __init__(self, model: str, adapter: str | None = None,
                 dtype: str = "bfloat16", load_in_4bit: bool = False):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"  # decoder-only models must pad on the left
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {"torch_dtype": getattr(torch, dtype), "device_map": "cuda"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        torch.cuda.reset_peak_memory_stats()
        self.info = BackendInfo("hf", model, {"adapter": adapter, "dtype": dtype,
                                               "bnb4": load_in_4bit})

    def render(self, system, user):
        return render_chat(self.tok, system, user)

    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens):
        enc = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        gens = []
        for i in range(len(prompts)):
            ids = new[i]
            n = int((ids != self.tok.pad_token_id).sum())
            gens.append(Generation(self.tok.decode(ids, skip_special_tokens=True),
                                   int(enc["attention_mask"][i].sum()), n))
        return gens

    def peak_memory_gb(self):
        return torch.cuda.max_memory_allocated() / 1e9
