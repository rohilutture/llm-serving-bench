"""Offline benchmark: one serving configuration over Spider dev.

Measures what matters for a batch workload (throughput, not per-request latency;
see load_test.py for online latency) AND execution accuracy, so every speed-up
is reported next to what it cost in correctness.

    python -m bench.runner --backend vllm --model merged/ --name vllm-bf16 \
        --spider-dir data/spider --limit 1034
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .exec_match import compare
from .prompts import DEFAULT_SYSTEM, extract_sql, load_template, user_message
from .spider import db_path, load_examples, schema_text


def model_size_gb(model: str) -> float | None:
    p = Path(model)
    if not p.is_dir():
        return None
    files = [f for f in p.rglob("*") if f.suffix in {".safetensors", ".bin", ".pt"}]
    return round(sum(f.stat().st_size for f in files) / 1e9, 3) if files else None


def build_prompts(backend, examples, db_root, template, system=DEFAULT_SYSTEM):
    return [backend.render(system, user_message(
        template, schema_text(str(db_path(db_root, ex.db_id))), ex.question))
        for ex in examples]


def generate_all(backend, prompts, batch_size, max_new_tokens, sort_by_length):
    """Returns generations in the ORIGINAL order plus per-call timings.

    Sorting by prompt length before static batching is the fair way to run the
    HF baseline (less padding); it is a no-op for vLLM, which schedules itself.
    """
    order = list(range(len(prompts)))
    if sort_by_length:
        order.sort(key=lambda i: len(prompts[i]))
    bs = batch_size or len(prompts)
    gens = [None] * len(prompts)
    timings = []
    for start in range(0, len(order), bs):
        idx = order[start:start + bs]
        t0 = time.perf_counter()
        out = backend.generate([prompts[i] for i in idx], max_new_tokens)
        dt = time.perf_counter() - t0
        for i, g in zip(idx, out):
            gens[i] = g
        timings.append({"n": len(idx), "seconds": dt,
                        "completion_tokens": sum(g.completion_tokens for g in out)})
    return gens, timings


def score(examples, gens, db_root, workers=8):
    def one(pair):
        ex, g = pair
        sql = extract_sql(g.text)
        m = compare(str(db_path(db_root, ex.db_id)), sql, ex.gold_sql)
        return {"idx": ex.idx, "db_id": ex.db_id, "difficulty": ex.difficulty,
                "question": ex.question, "gold": ex.gold_sql, "pred": sql,
                "raw": g.text, "correct": m.correct, "strict": m.strict, "kind": m.kind,
                "detail": m.detail, "prompt_tokens": g.prompt_tokens,
                "completion_tokens": g.completion_tokens}
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, zip(examples, gens)))


def summarize(rows, timings, backend, model):
    n = len(rows)
    scored = [r for r in rows if r["kind"] != "gold_error"]
    wall = sum(t["seconds"] for t in timings)
    toks = sum(t["completion_tokens"] for t in timings)
    tiers = defaultdict(list)
    for r in scored:
        tiers[r["difficulty"]].append(r["correct"])
    return {
        "backend": backend.info.name, "model": backend.info.model, **backend.info.extra,
        "n": n, "n_scored": len(scored),
        "exec_accuracy": round(sum(r["correct"] for r in scored) / max(len(scored), 1), 4),
        "exec_accuracy_strict": round(sum(r["strict"] for r in scored) / max(len(scored), 1), 4),
        "by_kind": dict(Counter(r["kind"] for r in rows)),
        "by_difficulty": {k: round(sum(v) / len(v), 4) for k, v in sorted(tiers.items())},
        "wall_seconds": round(wall, 3),
        "requests_per_s": round(n / wall, 3) if wall else None,
        "output_tokens_per_s": round(toks / wall, 1) if wall else None,
        "mean_completion_tokens": round(toks / max(n, 1), 1),
        "peak_memory_gb": backend.peak_memory_gb(),
        # Blank when the weights are quantized at load time: the model dir's size would be the bf16 size.
        "model_size_gb": None if backend.info.extra.get("quantization") else model_size_gb(model),
    }


def make_backend(a):
    if a.backend == "hf":
        from .backends.hf import HFBackend
        return HFBackend(a.model, a.adapter, load_in_4bit=a.bnb4)
    if a.backend == "vllm":
        from .backends.vllm_backend import VLLMBackend
        return VLLMBackend(a.model, a.adapter, a.quantization,
                           speculative_ngram=a.spec_ngram,
                           prefix_caching=not a.no_prefix_cache,
                           max_num_seqs=a.max_num_seqs, kv_cache_dtype=a.kv_cache_dtype)
    raise ValueError(a.backend)


def run(backend, examples, db_root, out_dir, template, batch_size=0,
        max_new_tokens=256, sort_by_length=False, model=""):
    prompts = build_prompts(backend, examples, db_root, template)
    # Warm-up: first call pays CUDA graph capture / kernel autotuning.
    backend.generate(prompts[:min(4, len(prompts))], 8)
    gens, timings = generate_all(backend, prompts, batch_size, max_new_tokens, sort_by_length)
    rows = score(examples, gens, db_root)
    summary = summarize(rows, timings, backend, model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "timings.json").write_text(json.dumps(timings))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["hf", "vllm"], required=True)
    p.add_argument("--model", required=True, help="HF id or local dir (merged or quantized)")
    p.add_argument("--adapter", help="LoRA adapter dir (serve adapter instead of merged)")
    p.add_argument("--name", required=True, help="run name -> results/<name>/")
    p.add_argument("--spider-dir", default="data/spider")
    p.add_argument("--split-file", default="dev.json")
    p.add_argument("--difficulty-file")
    p.add_argument("--limit", type=int)
    p.add_argument("--prompt-template")
    p.add_argument("--batch-size", type=int, default=0, help="0 = all at once")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--quantization", help="vLLM quantization override, e.g. fp8")
    p.add_argument("--bnb4", action="store_true", help="HF: bitsandbytes NF4")
    p.add_argument("--spec-ngram", type=int, default=0)
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--kv-cache-dtype", default="auto")
    a = p.parse_args(argv)

    spider = Path(a.spider_dir)
    examples = load_examples(spider / a.split_file, a.limit, a.difficulty_file)
    backend = make_backend(a)
    bs = a.batch_size or (16 if a.backend == "hf" else 0)
    s = run(backend, examples, spider / "database", Path("results") / a.name,
            load_template(a.prompt_template), bs, a.max_new_tokens,
            sort_by_length=(a.backend == "hf"), model=a.model)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
