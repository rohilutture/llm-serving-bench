"""Produce quantized checkpoints vLLM can load directly (llm-compressor format).

Schemes:
  w4a16-gptq : 4-bit weights, 16-bit activations, GPTQ with calibration data
  w4a16-awq  : same storage, AWQ scaling (protects salient channels)
  fp8        : FP8 weights+activations, dynamic per-token scales, no calibration
               (needs Ada/Hopper/Blackwell tensor cores: RTX 4090 and 5090 both qualify)

Calibration uses Spider TRAIN prompts rendered exactly like inference prompts.
Never calibrate on dev — that leaks the eval set into the weights.

    python scripts/quantize.py --model models/merged-bf16 --scheme w4a16-gptq \
        --spider-dir data/spider --out models/w4a16-gptq
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.prompts import DEFAULT_SYSTEM, load_template, render_chat, user_message
from bench.spider import db_path, schema_ddl

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--scheme", choices=["w4a16-gptq", "w4a16-awq", "fp8"], required=True)
p.add_argument("--spider-dir", default="data/spider")
p.add_argument("--prompt-template")
p.add_argument("--n-calib", type=int, default=512)
p.add_argument("--out", required=True)
a = p.parse_args()

from llmcompressor import oneshot  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype="auto", device_map="auto")

if a.scheme == "fp8":
    from llmcompressor.modifiers.quantization import QuantizationModifier
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)
else:
    spider = Path(a.spider_dir)
    train = json.loads((spider / "train_spider.json").read_text(encoding="utf-8"))
    random.Random(0).shuffle(train)
    tpl = load_template(a.prompt_template)
    texts = []
    for r in train[:a.n_calib]:
        user = user_message(tpl, schema_ddl(str(db_path(spider / "database", r["db_id"]))),
                            r["question"])
        # Include the gold answer so calibration sees realistic output activations too.
        texts.append(render_chat(tok, DEFAULT_SYSTEM, user) + r["query"] + tok.eos_token)
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tok(b["text"], truncation=True, max_length=2048, add_special_tokens=False),
        batched=True, remove_columns=["text"])
    if a.scheme == "w4a16-gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    else:
        from llmcompressor.modifiers.awq import AWQModifier
        recipe = AWQModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=2048,
            num_calibration_samples=len(texts))

model.save_pretrained(a.out, save_compressed=True)
tok.save_pretrained(a.out)
print("quantized ->", a.out)
