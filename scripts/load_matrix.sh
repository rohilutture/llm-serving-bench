#!/usr/bin/env bash
# Online latency sweep: start a vLLM server per config, sweep concurrency, stop it.
set -euo pipefail
SPIDER=${SPIDER:-data/spider}
TPL=${TPL:-configs/prompt_template.txt}

serve_and_test () {  # $1 name, $2 model dir, rest: extra serve flags
  local name=$1 model=$2; shift 2
  vllm serve "$model" --port 8000 --gpu-memory-utilization 0.85 "$@" > "results/$name.server.log" 2>&1 &
  local pid=$!
  until curl -sf localhost:8000/health >/dev/null; do sleep 2; done
  python -m bench.load_test --model "$model" --spider-dir "$SPIDER" --prompt-template "$TPL" \
      --concurrency 1 4 16 64 --n 256 --name "$name"
  kill $pid; wait $pid 2>/dev/null || true
}

mkdir -p results
serve_and_test vllm-bf16       models/merged-bf16 --enable-prefix-caching
serve_and_test vllm-w4a16-gptq models/w4a16-gptq  --enable-prefix-caching
serve_and_test vllm-fp8        models/fp8         --enable-prefix-caching
