#!/usr/bin/env bash
# Serve the base VLM with the tuned LoRA adapter behind one OpenAI-compatible endpoint.
#
#   serve/serve_vllm.sh /path/to/qwen25vl3b-r16-train/adapter
#   ADAPTER_NAME=slotfill-lora PORT=8000 serve/serve_vllm.sh <adapter dir>
#
# The OpenAI `model` field then selects the condition:
#
#   --model Qwen/Qwen2.5-VL-3B-Instruct   the untuned base
#   --model slotfill-lora                 the tuned adapter
#
# ...so baseline and adapter are benchmarked through identical code, which is the point.
# `eval/predict.py --backend openai --base-url http://localhost:8000/v1` talks to it.
#
# Notes that cost real time to rediscover:
#
# * vLLM applies LoRA to the language model only for Qwen2.5-VL; weights in the vision
#   tower are ignored. `train/train_vlm.py` freezes the vision layers for exactly this
#   reason, so this adapter loads cleanly. An adapter trained with --tune-vision would
#   load here and silently serve less than was trained.
# * A T4 (compute capability 7.5) has no bfloat16, so --dtype float16 is passed
#   explicitly rather than left to autodetection, and FlashAttention is unavailable
#   there - vLLM falls back to a supported backend on its own.
# * --max-model-len is deliberately small. A page is ~700 visual tokens and a record is
#   ~250, so 4096 is generous; the default (128k) would reserve KV cache for a context
#   this workload never uses and can refuse to start on a 15GB card.
set -euo pipefail

ADAPTER_PATH="${1:-}"
if [ -z "$ADAPTER_PATH" ]; then
  echo "usage: $0 <adapter dir>   (the folder holding adapter_config.json)" >&2
  exit 2
fi
if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
  echo "$ADAPTER_PATH has no adapter_config.json - point at the adapter/ folder" >&2
  exit 2
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
ADAPTER_NAME="${ADAPTER_NAME:-slotfill-lora}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

echo "serving $BASE_MODEL + $ADAPTER_NAME=$ADAPTER_PATH on :$PORT (dtype $DTYPE)"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$BASE_MODEL" \
  --enable-lora \
  --lora-modules "$ADAPTER_NAME=$ADAPTER_PATH" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --limit-mm-per-prompt '{"image":1}' \
  --port "$PORT" \
  "${@:2}"
