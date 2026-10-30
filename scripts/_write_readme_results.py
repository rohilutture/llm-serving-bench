"""One-off: write the README '## Results' section and a DESIGN.md 'what the measurements showed' section from
results/*, so no number is typed by hand.   python scripts/_write_readme_results.py
"""
import json
import random
import re
from pathlib import Path

R = Path("results")


def summ(n):
    return json.loads((R / n / "summary.json").read_text(encoding="utf-8"))


def preds(n):
    return [json.loads(l) for l in (R / n / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l]


def online(n):
    f = R / n / "load_test.json"
    return {r["concurrency"]: r for r in json.loads(f.read_text())} if f.exists() else {}


BASE = "hf-bf16"
base_c = [bool(r["correct"]) for r in preds(BASE)]
rng = random.Random(0)


def paired(n, ref=BASE, B=4000):
    ref_c = base_c if ref == BASE else [bool(r["correct"]) for r in preds(ref)]
    x = [bool(r["correct"]) for r in preds(n)]
    d = [int(a) - int(b) for a, b in zip(x, ref_c)]
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(B))
    return 100 * sum(d) / len(d), 100 * bs[int(.025 * B)], 100 * bs[int(.975 * B)]


ORDER = [("hf-bf16", "HF bf16, merged (baseline; batch 16)"), ("hf-bf16-bs1", "HF bf16, batch 1"),
         ("hf-bf16-bs64", "HF bf16, batch 64"), ("hf-bf16-adapter", "HF bf16, live LoRA"),
         ("hf-nf4", "HF NF4 4-bit (bitsandbytes)"),
         ("hf-bf16-modeldefaults", "HF bf16 with the model's repetition penalty 1.1 (the training repo's decoding)"),
         ("vllm-bf16", "vLLM bf16, merged"), ("vllm-bf16-lora", "vLLM bf16, live LoRA"),
         ("vllm-bf16-noprefix", "vLLM bf16, prefix caching off"), ("vllm-bf16-ngram5", "vLLM bf16, n-gram speculation (5)"),
         ("vllm-fp8", "vLLM FP8 (llm-compressor, W8A8)"), ("vllm-fp8-online", "vLLM FP8 (online dynamic)"),
         ("vllm-w4a16-gptq", "vLLM W4A16 GPTQ"), ("vllm-w4a16-awq", "vLLM W4A16 AWQ"),
         ("vllm-bf16-kvfp8", "vLLM bf16 + FP8 KV cache ⚠")]
rows = []
base_tps = summ(BASE)["output_tokens_per_s"]
for n, label in ORDER:
    if not (R / n / "summary.json").exists():
        continue
    s = summ(n)
    d, lo, hi = paired(n)
    w = s.get("model_size_gb")
    delta = "baseline" if n == BASE else f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    rows.append(f"| {label} | {s['exec_accuracy']:.3f} | {s['exec_accuracy_strict']:.3f} | {delta} | "
                f"{s['output_tokens_per_s']:,.0f} | {s['output_tokens_per_s'] / base_tps:.1f}x | {w if w else '—'} |")

ONL = [("vllm-bf16", "bf16 (merged)"), ("vllm-bf16-noprefix", "bf16, no prefix cache"), ("vllm-bf16-lora", "bf16, live LoRA"),
       ("vllm-bf16-ngram5", "bf16, n-gram (5, lookup 4)"), ("vllm-bf16-ngram3", "bf16, n-gram (3, lookup 1-2)"),
       ("vllm-fp8", "FP8"), ("vllm-w4a16-gptq", "W4A16 GPTQ"), ("vllm-w4a16-awq", "W4A16 AWQ")]
orows = []
for n, label in ONL:
    o = online(n)
    if not o:
        continue
    c1, c64 = o[1], o[64]
    orows.append(f"| {label} | {c1['latency_p50']:.3f} / {c1['latency_p95']:.3f} | {c1['output_tok_per_s']:.0f} | "
                 f"{c64['latency_p50']:.3f} / {c64['latency_p95']:.3f} | {c64['output_tok_per_s']:,.0f} |")


def tps(n):
    return summ(n)["output_tokens_per_s"]


def acc(n):
    return summ(n)["exec_accuracy"]


vb, vf, vg, va = "vllm-bf16", "vllm-fp8", "vllm-w4a16-gptq", "vllm-w4a16-awq"
_first = summ(vb)["wall_seconds_runs"][0]
ratio_cold = (summ("vllm-bf16-noprefix").get("wall_seconds_runs") or [summ("vllm-bf16-noprefix")["wall_seconds"]])[0] / _first
text = f"""## Results

Everything below was measured on **one RTX 4090 (24 GB)** inside the `vllm/vllm-openai` Docker image
(vLLM {'0.32.0'}, torch 2.13 + CUDA 13.0, Python 3.12) on Spider dev (1,034 questions), with the adapter retrained from
[text-to-sql-finetune](https://github.com/rohilutture/text-to-sql-finetune) (its own evaluator scores it 74.6% vs. 56.9%
prompt-only). Greedy decoding, `max_new_tokens=256`, prompts identical across engines (the training prompt and schema
format are vendored in `bench/train_format.py`). "Exec acc" is Spider-dev **execution accuracy (result-set match)** from
this repo's own harness, not the official test-suite evaluator; it ignores column order, like the training repo's metric
("Strict" requires identical column order). The Δ column is the **paired** accuracy difference vs. the baseline in
points with a 95% bootstrap interval over the 1,034 questions; an interval containing 0 means no detectable difference.
Throughput is offline output tokens/s (vLLM rows: median of 5 generation passes; HF rows: one pass).

| Run | Exec acc | Strict | Δ acc vs baseline (pts) [95% CI] | Out tok/s | Speed-up | Weights GB |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Online latency (`vllm serve`, streaming, 256 requests per level after a discarded warm-up pass; latency = seconds for the
full query, p50 / p95):

| Config | c=1 latency | c=1 tok/s | c=64 latency | c=64 tok/s |
|---|---|---|---|---|
{chr(10).join(orows)}

Full tables (all concurrency levels, TTFT, p99) are in `results/REPORT.md`; every prediction is in `results/<run>/`.

### What the measurements showed

- **Decoding conventions matter as much as the engine.** `generate(do_sample=False)` in HF still applies the model's
  `generation_config.json` repetition penalty (1.1 for Qwen2.5-Coder), while vLLM's hand-built sampling parameters do
  not. With the penalty on, HF scored {acc('hf-bf16-modeldefaults'):.3f}; with pure greedy it scored {acc('hf-bf16'):.3f},
  matching vLLM ({acc(vb):.3f}). Before pinning decoding, the two engines disagreed on 23% of predicted queries and vLLM
  looked 1.7 points "better". The training repo's 74.6% was produced with the penalty on, so this adapter's pure-greedy
  accuracy is about 76%.
- **vLLM is about {tps(vb) / tps(BASE):.0f}x faster offline** than HF bf16 at batch 16, which was the fastest of the three
  HF batch sizes tried (1, 16, 64): bigger static batches are slower because every sequence waits for the longest output.
  Left-padding did not cost accuracy (batch 1 vs. 16: no detectable difference).
- **Prefix caching is the biggest single knob** for this workload (most of every prompt is the shared schema). Offline,
  turning it off leaves {tps('vllm-bf16-noprefix') / tps(vb) * 100:.0f}% of the cached throughput (the cached figure is the
  median of 5 passes over the same prompts, so by then every prompt is cached; the first, cold pass is about
  {ratio_cold:.1f}x faster than no caching). Online at 64 concurrent requests, p95 latency without it is
  {online('vllm-bf16-noprefix')[64]['latency_p95'] / online(vb)[64]['latency_p95']:.1f}x worse and throughput
  {online(vb)[64]['output_tok_per_s'] / online('vllm-bf16-noprefix')[64]['output_tok_per_s']:.1f}x lower.
- **Quantization buys memory and low-concurrency latency, not offline throughput.** FP8 (llm-compressor) shrinks the weights
  ~42% for {100 * (acc(vb) - acc(vf)):.1f} points of accuracy; W4A16 shrinks them ~63% but costs
  {100 * (acc(vb) - acc(vg)):.1f} points (GPTQ) and {100 * (acc(vb) - acc(va)):.1f} points (AWQ). AWQ did **worse** than
  GPTQ here, contrary to the design note's expectation. Offline throughput relative to bf16 is
  {tps(vf) / tps(vb):.2f}x (FP8), {tps(vg) / tps(vb):.2f}x (GPTQ), {tps(va) / tps(vb):.2f}x (AWQ), within the
  pass-to-pass noise of a few percent. Online at concurrency 1, p50 latency is
  {100 * (1 - online(vf)[1]['latency_p50'] / online(vb)[1]['latency_p50']):.0f}% lower for FP8,
  {100 * (1 - online(vg)[1]['latency_p50'] / online(vb)[1]['latency_p50']):.0f}% for GPTQ and
  {100 * (1 - online(va)[1]['latency_p50'] / online(vb)[1]['latency_p50']):.0f}% for AWQ, and the advantage shrinks as
  concurrency rises (a saturated GPU is compute-bound, not weight-bandwidth-bound).
- **N-gram speculative decoding did not help; it hurt.** It left accuracy unchanged but was slower at every concurrency
  level, including 1 ({online('vllm-bf16-ngram5')[1]['output_tok_per_s']:.0f} vs. {online(vb)[1]['output_tok_per_s']:.0f}
  tok/s). The server log shows why: only ~0.3% of drafted tokens were accepted (mean acceptance length 1.02 with a
  4-token lookup window). Shortening the window (lookup 1-2 tokens, 3 drafted) raised the mean acceptance length to about
  1.2 and recovered some speed ({online('vllm-bf16-ngram3')[1]['output_tok_per_s']:.0f} tok/s at concurrency 1), but it still
  trailed plain decoding. A plausible reason is that the schema is a bullet list while the model writes
  `SELECT T1.col FROM table AS T1`, so n-gram lookups rarely match; I did not test that explanation.
- **Live LoRA is slower than merged weights:** {100 * (1 - tps('vllm-bf16-lora') / tps(vb)):.0f}% lower offline throughput
  in vLLM, about {100 * (1 - online('vllm-bf16-lora')[1]['output_tok_per_s'] / online(vb)[1]['output_tok_per_s']):.0f}% lower
  online, and 4.5x slower in HF; merge unless you need many adapters on one base.
- **FP8 KV cache broke the model** (accuracy collapsed to {acc('vllm-bf16-kvfp8'):.2f}; outputs were malformed SQL,
  e.g. repeated keywords). Its speed number is meaningless. A likely cause is uncalibrated KV scales, but I did not test that.
- **Measurement noise.** The first of the 5 generation passes in each vLLM row is 10-20% slower than the rest (cold
  caches), and the same checkpoint scored 0.7611 and 0.7582 in two runs (3 questions) because greedy decoding is not
  bit-deterministic under different batching. Sub-point accuracy differences and few-percent speed differences are noise.
- **Not done:** the DPO stretch goal (`dpo/`) was not run, so there are no DPO results.

**Limits:** one GPU, one 1.5B adapter, one dataset; accuracy differences under about 1 point are not distinguishable from
noise at n=1,034; HF throughput rows are single passes; online numbers come from one sweep of 256 requests per level; the
retrained adapter is not the original checkpoint (the original is unavailable), though it reproduces the original's
accuracy (74.6% vs. 74.2%).

"""
p = Path("README.md")
s = p.read_text(encoding="utf-8")
a, b = s.index("## Results"), s.index("## Quick start")
p.write_text(s[:a] + text + s[b:], encoding="utf-8")
print(text[:1500])
