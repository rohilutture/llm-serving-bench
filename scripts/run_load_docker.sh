#!/usr/bin/env bash
# Part 3: online latency sweep. For each config: start `vllm serve`, wait for /health (with a timeout), run the
# concurrency sweep (a discarded warm-up pass first), stop the server. A config whose results/<name>/load_test.json
# exists is skipped; a failure is logged and the script moves on.
#
#   docker exec -d bench bash -c "cd /work && bash scripts/run_load_docker.sh > results/load.log 2>&1"
set -u
cd "$(dirname "$0")/.."
SPIDER=${SPIDER:-/data/spider}
TPL=${TPL:-configs/prompt_template.txt}
BASE=${BASE:-Qwen/Qwen2.5-Coder-1.5B-Instruct}
ADAPTER=${ADAPTER:-/adapter}
PORT=8000
CONC=${CONC:-"1 4 16 64"}
N=${N:-256}
mkdir -p results/logs

serve_and_test() {  # serve_and_test <name> <model name to request> <vllm serve args...>
  local name=$1 req=$2; shift 2
  if [ -f "results/$name/load_test.json" ]; then echo "== online $name: already done, skipping"; return; fi
  echo "== online $name $(date +%H:%M:%S)"
  vllm serve "$@" --port "$PORT" --gpu-memory-utilization 0.85 --max-model-len 4096 > "results/logs/$name.server.log" 2>&1 &
  local pid=$! ok=0
  for _ in $(seq 1 180); do  # up to ~6 minutes for weights load + CUDA graph capture
    if curl -sf "localhost:$PORT/health" > /dev/null; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if [ $ok -eq 1 ]; then
    python -m bench.load_test --model "$req" --spider-dir "$SPIDER" --prompt-template "$TPL" \
        --concurrency $CONC --n "$N" --name "$name" > "results/logs/$name.load.log" 2>&1 \
      && python - "$name" <<'PY'
import json, sys
for r in json.load(open(f"results/{sys.argv[1]}/load_test.json")):
    print("   c=%-3d ttft p50 %.3fs | latency p50 %.3fs p95 %.3fs p99 %.3fs | %.0f out tok/s" % (
        r["concurrency"], r["ttft_p50"], r["latency_p50"], r["latency_p95"], r["latency_p99"], r["output_tok_per_s"]))
PY
    [ -f "results/$name/load_test.json" ] || { echo "   LOAD TEST FAILED; last lines of results/logs/$name.load.log:"; tail -n 4 "results/logs/$name.load.log" | cut -c1-200; }
  else
    echo "   SERVER FAILED to start; last lines of results/logs/$name.server.log:"; tail -n 5 "results/logs/$name.server.log" | cut -c1-220
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 5
}

SPEC='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
serve_and_test vllm-bf16          m models/merged-bf16 --served-model-name m --enable-prefix-caching
serve_and_test vllm-bf16-noprefix m models/merged-bf16 --served-model-name m --no-enable-prefix-caching
serve_and_test vllm-bf16-ngram5   m models/merged-bf16 --served-model-name m --enable-prefix-caching --speculative-config "$SPEC"
# Tuned n-gram variant: a shorter lookup window (1-2 tokens) so drafts can match inside the schema text
SPEC2='{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":2,"prompt_lookup_min":1}'
serve_and_test vllm-bf16-ngram3   m models/merged-bf16 --served-model-name m --enable-prefix-caching --speculative-config "$SPEC2"
serve_and_test vllm-bf16-lora     sql "$BASE" --served-model-name base --enable-lora --lora-modules sql="$ADAPTER" --max-lora-rank 64 --enable-prefix-caching
for s in fp8 w4a16-gptq w4a16-awq; do
  [ -f "models/$s/config.json" ] && serve_and_test "vllm-$s" m "models/$s" --served-model-name m --enable-prefix-caching
done
echo "== ONLINE DONE $(date +%H:%M:%S)"
