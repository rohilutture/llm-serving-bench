# Serving benchmark

Baseline: `hf-bf16`.

- Pareto-optimal on (accuracy, throughput): `vllm-bf16`, `vllm-bf16-kvfp8`, `vllm-bf16-lora`, `vllm-fp8-online`
- Pareto-optimal AND within 2 accuracy points of the baseline (the usable set): `vllm-bf16`, `vllm-bf16-lora`, `vllm-fp8-online`
- ⚠ Accuracy regression (more than 2 pts below baseline; fast but not usable): `hf-nf4`, `vllm-bf16-kvfp8`

`Exec acc` ignores column order (the training repo's rule); `Strict acc` requires identical column order. `Weights GB` is blank for runs that quantize at load time.

| Run | Exec acc | Strict acc | Out tok/s | Req/s | Speed-up | Δ acc (pts) | Weights GB | Peak mem GB |
|---|---|---|---|---|---|---|---|---|
| hf-bf16 | 0.7582 | 0.7321 | 291.8 | 9.472 | 1.00x | 0.0 | 3.087 | 5.845545472 |
| hf-bf16-adapter | 0.7602 | 0.7331 | 64.2 | 2.079 | 0.22x | 0.2 | — | 7.614075392 |
| hf-bf16-bs1 | 0.7592 | 0.7321 | 63.9 | 2.067 | 0.22x | 0.1 | 3.087 | 3.32680192 |
| hf-bf16-bs64 | 0.7602 | 0.7331 | 98.4 | 3.178 | 0.34x | 0.2 | 3.087 | 14.081639936 |
| hf-bf16-modeldefaults | 0.7437 | 0.7137 | 293.6 | 9.393 | 1.01x | -1.45 | 3.087 | 5.845545472 |
| hf-nf4 | 0.7253 | 0.6886 | 240.2 | 7.709 | 0.82x | -3.29 | 3.087 | 3.972848128 |
| vllm-bf16 | 0.7611 | 0.735 | 7723.4 | 249.728 | 26.47x | 0.29 | 3.087 | — |
| vllm-bf16-kvfp8 | 0.2698 | 0.2679 | 10770.3 | 358.767 | 36.91x | -48.84 | 3.087 | — |
| vllm-bf16-lora | 0.7621 | 0.736 | 6422.0 | 208.285 | 22.01x | 0.39 | — | — |
| vllm-bf16-ngram5 | 0.7611 | 0.734 | 3972.5 | 128.823 | 13.61x | 0.29 | 3.087 | — |
| vllm-bf16-noprefix | 0.7592 | 0.7331 | 1594.7 | 51.607 | 5.47x | 0.1 | 3.087 | — |
| vllm-fp8-online | 0.7524 | 0.7273 | 7831.9 | 255.294 | 26.84x | -0.58 | 3.087 | — |

![speed vs accuracy](REPORT.png)
