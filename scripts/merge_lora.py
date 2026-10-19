"""Merge the text-to-SQL LoRA adapter into the base weights -> a plain checkpoint.

Quantizers and most serving engines want one dense checkpoint; merging also
removes the per-token LoRA matmul at inference time.

    python scripts/merge_lora.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --adapter ../text-to-sql-finetune/checkpoints/best --out models/merged-bf16
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--adapter", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
print("merged ->", a.out)
