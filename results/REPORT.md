# Serving benchmark

Baseline: `hf-bf16`. Pareto-optimal: `vllm-bf16`, `vllm-bf16-kvfp8`, `vllm-bf16-lora`, `vllm-fp8-online`

| Run | Exec acc | Out tok/s | Req/s | Speed-up | Δ acc (pts) | Weights GB | Peak mem GB |
|---|---|---|---|---|---|---|---|
| hf-bf16 | 0.7437 | 296.1 | 9.472 | 1.00x | 0.0 | 3.087 | 5.845545472 |
| hf-bf16-adapter | 0.7157 | 65.5 | 2.084 | 0.22x | -2.8 | — | 7.614075392 |
| hf-nf4 | 0.707 | 241.0 | 7.482 | 0.81x | -3.67 | 3.087 | 3.972848128 |
| vllm-bf16 | 0.7611 | 7723.4 | 249.728 | 26.08x | 1.74 | 3.087 | — |
| vllm-bf16-kvfp8 | 0.2698 | 10770.3 | 358.767 | 36.37x | -47.39 | 3.087 | — |
| vllm-bf16-lora | 0.7621 | 6422.0 | 208.285 | 21.69x | 1.84 | — | — |
| vllm-bf16-ngram5 | 0.7611 | 3972.5 | 128.823 | 13.42x | 1.74 | 3.087 | — |
| vllm-bf16-noprefix | 0.7592 | 1594.7 | 51.607 | 5.39x | 1.55 | 3.087 | — |
| vllm-fp8-online | 0.7524 | 7831.9 | 255.294 | 26.45x | 0.87 | 3.087 | — |

![speed vs accuracy](REPORT.png)
