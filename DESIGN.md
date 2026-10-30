# Design notes

These are the decisions in this repo and the reasoning behind each one, written so I can defend them in an interview.

## 1. Measure accuracy on every configuration
Quantization and speculative decoding are usually reported as speed numbers alone. For text-to-SQL, one changed token (a column name, a `>` vs `>=`) changes the answer. So every run goes through the same execution harness, and the report puts **Δ accuracy next to speed-up**. The Pareto plot marks the configurations that no other run beats on both axes at once. The rest are strictly worse choices.

## 2. Why the HF baseline is static-batched and sorted by length
`generate()` pads each batch to its longest prompt and keeps decoding until the longest output ends. Sorting by prompt length before batching gives the baseline its best fair chance. Otherwise the vLLM speed-up looks bigger than it really is. Any interviewer who knows the topic will check this.

## 3. What makes vLLM faster here
- **Continuous batching:** a finished sequence leaves the batch and a new one joins at the next decode step, so short requests don't wait on long ones.
- **PagedAttention:** the KV cache is allocated in blocks rather than one contiguous max-length slab per sequence. Less fragmentation means more sequences fit in VRAM, and a bigger batch means more throughput.
- **Prefix caching:** in Spider, many questions share a database, and the schema DDL is most of the prompt. KV blocks for a shared prefix are computed once and reused. The `noprefix` run isolates this effect.

## 4. Quantization choices
| Scheme | Stores | Calibration | Expectation |
|---|---|---|---|
| FP8 dynamic | 8-bit W+A | none | ~2x smaller, near-lossless, uses FP8 tensor cores (Ada/Blackwell) |
| W4A16 GPTQ | 4-bit W | 512 Spider *train* prompts | ~3.5x smaller; GPTQ fixes rounding error layer by layer using second-order info |
| W4A16 AWQ | 4-bit W | same | scales salient channels (by activation magnitude) before rounding; often holds accuracy better than GPTQ on small models |
| NF4 (bnb, HF) | 4-bit W | none | memory win only; dequantizes on the fly, often *slower* than bf16 |

On a 1.5B model, bf16 weights are only about 3 GB, and decode is memory-bandwidth bound. So W4A16 mainly speeds up *small-batch* decode. At large batch sizes the workload becomes compute bound and the gap shrinks. Showing that crossover across the concurrency sweep is part of the analysis.

Calibration data is **in-domain** (Spider train, same prompt format, including gold SQL so output activations are covered) and **never dev**. Calibrating on dev would leak the eval set into the weights.

## 5. N-gram speculative decoding
Speculative decoding drafts k tokens cheaply, then verifies them in one forward pass of the target model. Normally the drafter is a small model. Here it's **prompt lookup**: the draft is copied from an n-gram match in the prompt. SQL output is mostly table and column names copied from the schema, so the acceptance rate should be high and there's no second model to load. Greedy verification makes the output *identical* to plain greedy decoding, so accuracy should not move. If it does, that's a bug, and the benchmark would catch it.

## 6. Merged weights vs. live LoRA
Serving the adapter with `enable_lora` costs an extra low-rank matmul per layer, but it lets one base model serve many adapters (multi-tenant). Merging removes that cost and is required before quantizing. Both are benchmarked so the tradeoff shows up as a number.

## 7. Offline vs. online metrics
- **Offline** (`runner`): tokens/s and requests/s for a batch job. Latency per request is meaningless here, because everything is submitted at once.
- **Online** (`load_test`): streaming requests at fixed concurrency. TTFT is measured from the first content chunk, not estimated. p95/p99 matter because users feel the tail, not the mean.

## 8. Execution matching rules
- ORDER BY in gold → compare as lists; otherwise compare as multisets (`Counter`), so duplicates still count.
- Floats are rounded to 6 places.
- Read-only connection (`mode=ro`), so a generated `DELETE` can't corrupt the benchmark DB.
- A progress-handler timeout stops runaway queries (for example, a recursive CTE with no stop condition).
- Error kinds (`invalid_sql`, `wrong_rows`, `timeout`) are kept for every run, so you can see *how* a quantized model fails, not just how often.

## 9. DPO with execution rewards
After SFT, about half the remaining errors are "runs but returns wrong rows". DPO needs (chosen, rejected) pairs, and executing samples against the DB gives them for free. Rejected samples that run but return wrong rows are preferred over syntax errors as negatives: they sit close to the decision boundary, so they carry more signal. Questions where all k samples agree carry no preference signal and are dropped. The claim to test after training is that accuracy goes up **mostly through fewer `wrong_rows` errors**. The per-kind breakdown confirms or refutes that.

## 10. Honest-reporting rules
- Every number comes from my own runs on a stated GPU (one RTX 4090), with the vLLM version recorded (0.32.0, in the `vllm/vllm-openai` image).
- Accuracy is "execution accuracy on Spider dev (result-set match)" from this harness, not the official test-suite evaluator.
- A warm-up call runs before timing (CUDA graph capture and autotuning inflate the first call).

## 11. Where the measurements contradicted the plan
Written after the runs; the numbers are in the README.
- **N-gram speculation (section 5) did not help.** The bet was a high acceptance rate because SQL copies identifiers from
  the schema. The measured acceptance was ~0.3% of drafted tokens (mean acceptance length 1.02), and the run was slower at
  every concurrency level, including 1. Shrinking the lookup window helped a little but did not make it win.
- **AWQ did not beat GPTQ (section 4).** On this 1.5B adapter AWQ lost about 4 accuracy points and GPTQ about 2, with the same
  calibration data. FP8 was the only quantization that was close to lossless.
- **Quantization's speed benefit is a latency effect, not a throughput one.** Offline, FP8 and W4A16 were within a few
  percent of bf16 (a saturated GPU is compute-bound). The gain appears at concurrency 1 (p50 latency down 16-41%) and
  shrinks as concurrency rises, as section 4 predicted.
- **The decoding convention was a confounder (section 1).** HF `generate(do_sample=False)` still applies the model's
  generation_config repetition penalty; vLLM's explicit sampling parameters do not. Until decoding was pinned to pure
  greedy in both, the engines disagreed on 23% of predictions and vLLM looked 1.7 points "better".
- **The matcher is part of the measurement (section 8).** The first benchmark matcher required identical column order and
  scored the adapter 3 points below the training repo's evaluator, so it now follows the training repo's rules and also
  reports the strict number.
- **FP8 KV cache destroyed accuracy.** I had listed it as a memory lever; without calibrated scales the model produced
  malformed SQL (27% accuracy).
- **Timing needs repeats (section 10).** The first pass of each vLLM row is 10-20% slower than later ones, so single-run
  numbers were off by that much; vLLM rows are now the median of 5 passes.

