"""DPO on execution-verified pairs, as a fresh LoRA on top of the merged SFT model.

beta controls how far the policy may drift from the SFT reference; start at 0.1.
After training, merge with scripts/merge_lora.py and re-run bench.runner — the
claim to test is "accuracy up, mostly on wrong_rows errors", checked by kind.

    python dpo/train_dpo.py --model models/merged-bf16 --pairs dpo/pairs.jsonl --out models/dpo-lora
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--pairs", default="dpo/pairs.jsonl")
p.add_argument("--out", default="models/dpo-lora")
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--lr", type=float, default=5e-6)
p.add_argument("--epochs", type=int, default=1)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
ds = load_dataset("json", data_files=a.pairs, split="train").train_test_split(0.05, seed=0)

cfg = DPOConfig(output_dir=a.out, beta=a.beta, learning_rate=a.lr, num_train_epochs=a.epochs,
                per_device_train_batch_size=4, gradient_accumulation_steps=8,
                lr_scheduler_type="cosine", warmup_ratio=0.05, bf16=True,
                gradient_checkpointing=True, logging_steps=10, eval_strategy="steps",
                eval_steps=100, save_steps=200, max_length=2048, report_to="none")
peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
# ref_model=None + peft_config: TRL uses the frozen base (adapter disabled) as the reference,
# so no second copy of the model is loaded.
trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds["train"],
                     eval_dataset=ds["test"], processing_class=tok, peft_config=peft)
trainer.train()
trainer.save_model(a.out)
print("DPO adapter ->", a.out)
