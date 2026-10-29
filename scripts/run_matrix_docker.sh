#!/usr/bin/env bash
# Part 1 of the experiment matrix, written for the vllm/vllm-openai container (paths: /adapter, /data/spider).
# Everything that does not need a pre-quantized checkpoint. A run whose results/<name>/summary.json exists is
# skipped; a failing run is logged to results/logs/<name>.log and the script moves on.
#
#   docker exec -d bench bash -c "cd /work && bash scripts/run_matrix_docker.sh > results/matrix.log 2>&1"
set -u
cd "$(dirname "$0")/.."
BASE=${BASE:-Qwen/Qwen2.5-Coder-1.5B-Instruct}
ADAPTER=${ADAPTER:-/adapter}
SPIDER=${SPIDER:-/data/spider}
TPL=${TPL:-configs/prompt_template.txt}
M=models/merged-bf16
COMMON=(--spider-dir "$SPIDER" --prompt-template "$TPL")
mkdir -p results/logs

python -m pip install -q bitsandbytes 2>&1 | tail -1

[ -d "$M" ] || python scripts/merge_lora.py --base "$BASE" --adapter "$ADAPTER" --out "$M"

run() {  # run <name> <runner args...>
  local name=$1; shift
  if [ -f "results/$name/summary.json" ]; then echo "== $name: already done, skipping"; return; fi
  echo "== $name $(date +%H:%M:%S)"
  python -m bench.runner "$@" --name "$name" "${COMMON[@]}" > "results/logs/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    python - "$name" <<'PY'
import json, sys
s = json.load(open(f"results/{sys.argv[1]}/summary.json"))
print("   exec_acc %.4f (strict %.4f) | %s out tok/s | %s s wall" % (
    s["exec_accuracy"], s.get("exec_accuracy_strict", float("nan")), s["output_tokens_per_s"], s["wall_seconds"]))
PY
  else
    echo "   FAILED (exit $rc); last lines of results/logs/$name.log:"; tail -n 4 "results/logs/$name.log" | cut -c1-200
  fi
}

# HF baselines (the adapter row hf-bf16-adapter is the sanity run)
run hf-bf16-adapter --backend hf --model "$BASE" --adapter "$ADAPTER"
run hf-bf16         --backend hf --model "$M"
run hf-nf4          --backend hf --model "$M" --bnb4
# vLLM: precision / serving knobs
run vllm-bf16          --backend vllm --model "$M"
run vllm-bf16-lora     --backend vllm --model "$BASE" --adapter "$ADAPTER"
run vllm-bf16-noprefix --backend vllm --model "$M" --no-prefix-cache
run vllm-bf16-ngram5   --backend vllm --model "$M" --spec-ngram 5
run vllm-bf16-kvfp8    --backend vllm --model "$M" --kv-cache-dtype fp8
run vllm-fp8-online    --backend vllm --model "$M" --quantization fp8   # vLLM's dynamic FP8, no calibration

python -m bench.report --results results --baseline hf-bf16 && echo "== report written"
echo "== PART 1 DONE $(date +%H:%M:%S)"
