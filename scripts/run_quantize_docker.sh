#!/usr/bin/env bash
# Part 2 of the matrix: quantized checkpoints (llm-compressor) and their vLLM rows, for the vllm/vllm-openai container.
# llm-compressor goes into its own venv (/opt/lc) so its torch/transformers pins cannot break the vLLM install.
# Resumable: existing checkpoints and finished runs are skipped; a failure is logged and the script moves on.
#
#   docker exec -d bench bash -c "cd /work && bash scripts/run_quantize_docker.sh > results/quantize.log 2>&1"
set -u
cd "$(dirname "$0")/.."
SPIDER=${SPIDER:-/data/spider}
TPL=${TPL:-configs/prompt_template.txt}
M=models/merged-bf16
COMMON=(--spider-dir "$SPIDER" --prompt-template "$TPL")
mkdir -p results/logs

if [ ! -x /opt/lc/bin/python ]; then
  echo "== creating llm-compressor venv $(date +%H:%M:%S)"
  python -m venv /opt/lc && /opt/lc/bin/pip install -q llmcompressor datasets > results/logs/lc-install.log 2>&1 \
    || { echo "   llm-compressor install FAILED, see results/logs/lc-install.log"; tail -n 5 results/logs/lc-install.log | cut -c1-200; }
fi
/opt/lc/bin/python -c "import llmcompressor, torch, transformers; print('llmcompressor', llmcompressor.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)" 2>&1 | tail -1

quant() {  # quant <scheme>  ->  models/<scheme>
  local s=$1
  if [ -f "models/$s/config.json" ]; then echo "== quantize $s: already done, skipping"; return; fi
  echo "== quantize $s $(date +%H:%M:%S)"
  /opt/lc/bin/python scripts/quantize.py --model "$M" --scheme "$s" "${COMMON[@]}" --out "models/$s" \
      > "results/logs/quantize-$s.log" 2>&1 || { echo "   FAILED; last lines of results/logs/quantize-$s.log:"; tail -n 5 "results/logs/quantize-$s.log" | cut -c1-220; }
}

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
print("   exec_acc %.4f (strict %.4f) | %s out tok/s | %s s wall | weights %s GB" % (
    s["exec_accuracy"], s.get("exec_accuracy_strict", float("nan")), s["output_tokens_per_s"], s["wall_seconds"], s.get("model_size_gb")))
PY
  else
    echo "   FAILED (exit $rc); last lines of results/logs/$name.log:"; tail -n 4 "results/logs/$name.log" | cut -c1-200
  fi
}

for s in fp8 w4a16-gptq w4a16-awq; do quant "$s"; done
for s in fp8 w4a16-gptq w4a16-awq; do
  [ -f "models/$s/config.json" ] && run "vllm-$s" --backend vllm --model "models/$s"
done

python -m bench.report --results results --baseline hf-bf16 > /dev/null && echo "== report written"
echo "== PART 2 DONE $(date +%H:%M:%S)"
