# llm-serving-bench

How fast can a fine-tuned text-to-SQL model be served, and what does each speed-up cost in correctness?

This repo takes my fine-tuned **Qwen2.5-Coder-1.5B** text-to-SQL model ([text-to-sql-finetune](https://github.com/rohilutture/text-to-sql-finetune)) and benchmarks every common serving option against the same execution-accuracy harness on Spider dev:

| Axis | Configurations |
|---|---|
| Engine | HF `transformers.generate` (static batching) vs. **vLLM** (paged KV cache, continuous batching) |
| Precision | bf16 · FP8 (dynamic) · W4A16 GPTQ · W4A16 AWQ · bitsandbytes NF4 (HF) · FP8 KV cache |
| Serving knobs | merged weights vs. live LoRA adapter · prefix caching on/off · n-gram speculative decoding |
| Workload | offline throughput (`bench.runner`) · online latency under concurrency 1–64 (`bench.load_test`) |
| Stretch | **DPO with execution-verified preference pairs** (the database is the reward): code is included but was **not run**, so there are no DPO results |

Every row in the final report is **accuracy and speed together**. A speed-up that costs 3 points of execution accuracy is a decision you make on purpose, not something you find out about later.

## Results

Everything below was measured on **one RTX 4090 (24 GB)** inside the `vllm/vllm-openai` Docker image
(vLLM 0.32.0, torch 2.13 + CUDA 13.0, Python 3.12) on Spider dev (1,034 questions), with the adapter retrained from
[text-to-sql-finetune](https://github.com/rohilutture/text-to-sql-finetune) (its own evaluator scores it 74.6% vs. 56.9%
prompt-only). Greedy decoding, `max_new_tokens=256`, prompts identical across engines (the training prompt and schema
format are vendored in `bench/train_format.py`). "Exec acc" is Spider-dev **execution accuracy (result-set match)** from
this repo's own harness, not the official test-suite evaluator; it ignores column order, like the training repo's metric
("Strict" requires identical column order). The Δ column is the **paired** accuracy difference vs. the baseline in
points with a 95% bootstrap interval over the 1,034 questions; an interval containing 0 means no detectable difference.
Throughput is offline output tokens/s (vLLM rows: median of 5 generation passes; HF rows: one pass).

| Run | Exec acc | Strict | Δ acc vs baseline (pts) [95% CI] | Out tok/s | Speed-up | Weights GB |
|---|---|---|---|---|---|---|
| HF bf16, merged (baseline; batch 16) | 0.758 | 0.732 | baseline | 292 | 1.0x | 3.087 |
| HF bf16, batch 1 | 0.759 | 0.732 | +0.1 [-0.5, +0.7] | 64 | 0.2x | 3.087 |
| HF bf16, batch 64 | 0.760 | 0.733 | +0.2 [-0.3, +0.7] | 98 | 0.3x | 3.087 |
| HF bf16, live LoRA | 0.760 | 0.733 | +0.2 [-0.5, +0.9] | 64 | 0.2x | — |
| HF NF4 4-bit (bitsandbytes) | 0.725 | 0.689 | -3.3 [-5.2, -1.3] | 240 | 0.8x | 3.087 |
| HF bf16 with the model's repetition penalty 1.1 (the training repo's decoding) | 0.744 | 0.714 | -1.5 [-2.7, -0.2] | 294 | 1.0x | 3.087 |
| vLLM bf16, merged | 0.758 | 0.732 | +0.0 [-0.6, +0.6] | 9,001 | 30.8x | 3.087 |
| vLLM bf16, live LoRA | 0.759 | 0.733 | +0.1 [-0.4, +0.6] | 7,418 | 25.4x | — |
| vLLM bf16, prefix caching off | 0.759 | 0.733 | +0.1 [-0.7, +0.9] | 1,585 | 5.4x | 3.087 |
| vLLM bf16, n-gram speculation (5) | 0.761 | 0.734 | +0.3 [-0.4, +1.0] | 4,233 | 14.5x | 3.087 |
| vLLM FP8 (llm-compressor, W8A8) | 0.756 | 0.728 | -0.2 [-1.4, +1.0] | 8,378 | 28.7x | 1.779 |
| vLLM FP8 (online dynamic) | 0.757 | 0.732 | -0.1 [-1.3, +1.1] | 8,585 | 29.4x | — |
| vLLM W4A16 GPTQ | 0.740 | 0.715 | -1.8 [-3.5, -0.3] | 9,443 | 32.4x | 1.143 |
| vLLM W4A16 AWQ | 0.718 | 0.693 | -4.1 [-6.0, -2.1] | 9,338 | 32.0x | 1.148 |
| vLLM bf16 + FP8 KV cache ⚠ | 0.276 | 0.274 | -48.3 [-51.4, -45.1] | 11,947 | 40.9x | 3.087 |

Online latency (`vllm serve`, streaming, 256 requests per level after a discarded warm-up pass; latency = seconds for the
full query, p50 / p95):

| Config | c=1 latency | c=1 tok/s | c=64 latency | c=64 tok/s |
|---|---|---|---|---|
| bf16 (merged) | 0.114 / 0.313 | 216 | 0.265 / 0.618 | 5,019 |
| bf16, no prefix cache | 0.131 / 0.327 | 198 | 1.107 / 2.761 | 1,500 |
| bf16, live LoRA | 0.154 / 0.433 | 152 | 0.358 / 0.946 | 3,498 |
| bf16, n-gram (5, lookup 4) | 0.147 / 0.422 | 164 | 0.489 / 1.205 | 3,027 |
| bf16, n-gram (3, lookup 1-2) | 0.133 / 0.337 | 189 | 0.371 / 0.834 | 3,978 |
| FP8 | 0.096 / 0.254 | 246 | 0.257 / 0.624 | 4,996 |
| W4A16 GPTQ | 0.067 / 0.180 | 357 | 0.223 / 0.538 | 5,928 |
| W4A16 AWQ | 0.074 / 0.177 | 354 | 0.221 / 0.538 | 6,058 |

Full tables (all concurrency levels, TTFT, p99) are in `results/REPORT.md`; every prediction is in `results/<run>/`.

### What the measurements showed

- **Decoding conventions matter as much as the engine.** `generate(do_sample=False)` in HF still applies the model's
  `generation_config.json` repetition penalty (1.1 for Qwen2.5-Coder), while vLLM's hand-built sampling parameters do
  not. With the penalty on, HF scored 0.744; with pure greedy it scored 0.758,
  matching vLLM (0.758). Before pinning decoding, the two engines disagreed on 23% of predicted queries and vLLM
  looked 1.7 points "better". The training repo's 74.6% was produced with the penalty on, so this adapter's pure-greedy
  accuracy is about 76%.
- **vLLM is about 31x faster offline** than HF bf16 at batch 16, which was the fastest of the three
  HF batch sizes tried (1, 16, 64): bigger static batches are slower because every sequence waits for the longest output.
  Left-padding did not cost accuracy (batch 1 vs. 16: no detectable difference).
- **Prefix caching is the biggest single knob** for this workload (most of every prompt is the shared schema). Offline,
  turning it off leaves 18% of the cached throughput (the cached figure is the
  median of 5 passes over the same prompts, so by then every prompt is cached; the first, cold pass is about
  4.7x faster than no caching). Online at 64 concurrent requests, p95 latency without it is
  4.5x worse and throughput
  3.3x lower.
- **Quantization buys memory and low-concurrency latency, not offline throughput.** FP8 (llm-compressor) shrinks the weights
  ~42% for 0.2 points of accuracy; W4A16 shrinks them ~63% but costs
  1.8 points (GPTQ) and 4.1 points (AWQ). AWQ did **worse** than
  GPTQ here, contrary to the design note's expectation. Offline throughput relative to bf16 is
  0.93x (FP8), 1.05x (GPTQ), 1.04x (AWQ), within the
  pass-to-pass noise of a few percent. Online at concurrency 1, p50 latency is
  16% lower for FP8,
  41% for GPTQ and
  35% for AWQ, and the advantage shrinks as
  concurrency rises (a saturated GPU is compute-bound, not weight-bandwidth-bound).
- **N-gram speculative decoding did not help; it hurt.** It left accuracy unchanged but was slower at every concurrency
  level, including 1 (164 vs. 216
  tok/s). The server log shows why: only ~0.3% of drafted tokens were accepted (mean acceptance length 1.02 with a
  4-token lookup window). Shortening the window (lookup 1-2 tokens, 3 drafted) raised the mean acceptance length to about
  1.2 and recovered some speed (189 tok/s at concurrency 1), but it still
  trailed plain decoding. A plausible reason is that the schema is a bullet list while the model writes
  `SELECT T1.col FROM table AS T1`, so n-gram lookups rarely match; I did not test that explanation.
- **Live LoRA is slower than merged weights:** 18% lower offline throughput
  in vLLM, about 30% lower
  online, and 4.5x slower in HF; merge unless you need many adapters on one base.
- **FP8 KV cache broke the model** (accuracy collapsed to 0.28; outputs were malformed SQL,
  e.g. repeated keywords). Its speed number is meaningless. A likely cause is uncalibrated KV scales, but I did not test that.
- **Measurement noise.** The first of the 5 generation passes in each vLLM row is 10-20% slower than the rest (cold
  caches), and the same checkpoint scored 0.7611 and 0.7582 in two runs (3 questions) because greedy decoding is not
  bit-deterministic under different batching. Sub-point accuracy differences and few-percent speed differences are noise.
- **Not done:** the DPO stretch goal (`dpo/`) was not run, so there are no DPO results.

**Limits:** one GPU, one 1.5B adapter, one dataset; accuracy differences under about 1 point are not distinguishable from
noise at n=1,034; HF throughput rows are single passes; online numbers come from one sweep of 256 requests per level; the
retrained adapter is not the original checkpoint (the original is unavailable), though it reproduces the original's
accuracy (74.6% vs. 74.2%).

## Quick start

vLLM needs Linux. On Windows, use **WSL2 (Ubuntu)** with the NVIDIA driver installed on the Windows side. The CUDA toolkit does not need to be installed inside WSL.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt

# Spider: download from https://yale-lily.github.io/spider and unzip to data/spider
#   data/spider/{dev.json, train_spider.json, database/<db_id>/<db_id>.sqlite}

# The prompt and schema format are vendored from the text-to-sql-finetune repo (bench/train_format.py,
# configs/prompt_template.txt) and are the default, so serving matches how the adapter was trained.
# Spider's tables.json must sit next to dev.json. (BENCH_SCHEMA_FORMAT=ddl opts into a different format.)

BASE=Qwen/Qwen2.5-Coder-1.5B-Instruct \
ADAPTER=../text-to-sql-finetune/checkpoints/best \
LIMIT=200 bash scripts/run_matrix.sh    # quick pass; drop LIMIT for all 1,034
bash scripts/load_matrix.sh             # online latency sweep
```

Then open `results/REPORT.md`.

### How the published numbers were produced (Windows host, Docker)
vLLM does not run natively on Windows, so the results above were produced inside the `vllm/vllm-openai` image
(GPU passed through by Docker Desktop), with the repo, adapter and Spider copied into a long-lived container:

```bash
bash scripts/run_matrix_docker.sh                  # HF and vLLM rows (REPEATS=5 for the vLLM rows)
bash scripts/run_quantize_docker.sh                # llm-compressor FP8 / GPTQ / AWQ (isolated venv) + their vLLM rows
bash scripts/run_load_docker.sh                    # online latency sweep, one `vllm serve` per config
python scripts/rescore.py --run results/<name> --spider-dir /data/spider   # re-score saved predictions
```
Each script skips runs whose results already exist and logs failures to `results/logs/` instead of stopping.

### Getting the adapter
The original adapter file is not available, so it is **retrained** with the training repo
(https://github.com/rohilutture/text-to-sql-finetune: Qwen2.5-Coder-1.5B-Instruct, LoRA, 3 epochs, batch 8 x 4
accumulation, `--include-samples`), then evaluated there with `scripts/run_eval.py`. Use *that run's* execution accuracy
as this repo's reference: it will be close to, but not necessarily equal to, the 74.2% of the original adapter.

### Sanity check first
Run `hf-bf16-adapter` on the full dev set before anything else. It must reproduce the training repo's own
`run_eval.py` number for the same adapter to within about a point. If it doesn't, the prompt, schema format or SQL
extraction differs from training. Fix that before trusting any other row.

**What happened here:** the first run scored 71.6% vs. the training repo's 74.6%. The cause was a stricter matcher (the
benchmark required identical column order; the training evaluator tolerates permuted columns), found by re-scoring the
saved predictions with `scripts/rescore.py`: the corrected matcher gave 74.7%, so generation was identical and only the
scoring differed. The benchmark now scores exactly like the training repo and also reports the strict number.

### DPO stretch
```bash
python dpo/make_pairs.py --model models/merged-bf16 --spider-dir data/spider --k 8
python dpo/train_dpo.py  --model models/merged-bf16 --pairs dpo/pairs.jsonl
python scripts/merge_lora.py --base models/merged-bf16 --adapter models/dpo-lora --out models/dpo-merged
python -m bench.runner --backend vllm --model models/dpo-merged --name vllm-dpo --spider-dir data/spider
```

## Tests (CPU, no GPU needed)
```bash
pip install -r requirements-dev.txt && pytest -q
```
The tests cover execution matching (order rules, multisets, invalid SQL, timeouts, read-only DB), SQL extraction, the runner end to end with a mock backend, report/Pareto math, and DPO pair selection.

## Layout
```
bench/
  spider.py         dataset + schema DDL
  prompts.py        template, chat rendering, SQL extraction
  exec_match.py     execution-based correctness with error kinds
  runner.py         offline benchmark -> results/<run>/{summary,predictions,timings}
  load_test.py      online concurrency sweep against `vllm serve` (TTFT, p50/p95/p99)
  report.py         table + speed-vs-accuracy Pareto plot
  backends/         hf.py, vllm_backend.py, mock.py
scripts/            merge_lora.py, quantize.py (llm-compressor), run_matrix.sh, load_matrix.sh
dpo/                make_pairs.py, train_dpo.py (TRL)
```

See **[DESIGN.md](DESIGN.md)** for why each choice was made.
