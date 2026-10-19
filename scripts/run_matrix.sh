#!/usr/bin/env bash
# Full experiment matrix. Run inside WSL2/Linux with an NVIDIA GPU (vLLM has no
# native Windows support). Expect ~1-2 h end to end on an RTX 4090/5090.
#
#   BASE=Qwen/Qwen2.5-Coder-1.5B-Instruct ADAPTER=/path/to/adapter bash scripts/run_matrix.sh
set -euo pipefail
: "${BASE:?set BASE to the base model id}"
: "${ADAPTER:?set ADAPTER to the trained LoRA adapter dir}"
SPIDER=${SPIDER:-data/spider}
TPL=${TPL:-configs/prompt_template.txt}   # the exact training template
LIMIT=${LIMIT:-}                          # e.g. LIMIT=200 for a quick pass
COMMON=(--spider-dir "$SPIDER" --prompt-template "$TPL" ${LIMIT:+--limit $LIMIT})

python scripts/merge_lora.py --base "$BASE" --adapter "$ADAPTER" --out models/merged-bf16
for s in w4a16-gptq w4a16-awq fp8; do
  [ -d "models/$s" ] || python scripts/quantize.py --model models/merged-bf16 --scheme "$s" \
      --spider-dir "$SPIDER" --prompt-template "$TPL" --out "models/$s"
done

# 1) Baselines
python -m bench.runner --backend hf   --model "$BASE" --adapter "$ADAPTER" --name hf-bf16-adapter "${COMMON[@]}"
python -m bench.runner --backend hf   --model models/merged-bf16 --name hf-bf16 "${COMMON[@]}"
python -m bench.runner --backend hf   --model models/merged-bf16 --bnb4 --name hf-nf4 "${COMMON[@]}"
# 2) vLLM, precision sweep
python -m bench.runner --backend vllm --model models/merged-bf16 --name vllm-bf16 "${COMMON[@]}"
python -m bench.runner --backend vllm --model models/fp8         --name vllm-fp8 "${COMMON[@]}"
python -m bench.runner --backend vllm --model models/w4a16-gptq  --name vllm-w4a16-gptq "${COMMON[@]}"
python -m bench.runner --backend vllm --model models/w4a16-awq   --name vllm-w4a16-awq "${COMMON[@]}"
# 3) vLLM serving knobs on the bf16 model
python -m bench.runner --backend vllm --model "$BASE" --adapter "$ADAPTER" --name vllm-bf16-lora "${COMMON[@]}"
python -m bench.runner --backend vllm --model models/merged-bf16 --no-prefix-cache --name vllm-bf16-noprefix "${COMMON[@]}"
python -m bench.runner --backend vllm --model models/merged-bf16 --spec-ngram 5 --name vllm-bf16-ngram5 "${COMMON[@]}"
python -m bench.runner --backend vllm --model models/merged-bf16 --kv-cache-dtype fp8 --name vllm-bf16-kvfp8 "${COMMON[@]}"

python -m bench.report --results results --baseline hf-bf16
echo "Done. See results/REPORT.md"
