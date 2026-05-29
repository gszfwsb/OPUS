#!/bin/bash
# Convert FineWeb parquet files to OPUS-compatible .bin format using Qwen2.5-7B tokenizer.
set -euo pipefail

# Choose a python that has transformers installed.
# You can override with: PYTHON_BIN=/path/to/python bash data/run_tokenize_qwen25.sh
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "$PYTHON_BIN" -c "import transformers, pyarrow" >/dev/null 2>&1; then
  echo "[ERROR] Python ($PYTHON_BIN) missing deps (transformers, pyarrow). Activate your env or set PYTHON_BIN." 1>&2
  exit 1
fi

# HuggingFace cache (defaults to the standard HF location; override with HF_HOME)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Tokenizer: HuggingFace model id or a local directory (e.g. from cache_qwen25_7b_tokenizer.py)
TOKENIZER_DIR="${TOKENIZER_DIR:-Qwen/Qwen2.5-7B}"

# Input parquet directory (required)
PARQUET_DIR="${PARQUET_DIR:?'Set PARQUET_DIR to your FineWeb parquet directory'}"

# Output directories (override OUT_ROOT or the individual dirs)
OUT_ROOT="${OUT_ROOT:-./bins}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$OUT_ROOT/fineweb_qwen25_train}"
VAL_OUTPUT_DIR="${VAL_OUTPUT_DIR:-$OUT_ROOT/fineweb_qwen25_val}"

# Tokenization / sharding
VAL_TOKENS="${VAL_TOKENS:-200000000}"
TOKENS_PER_SHARD="${TOKENS_PER_SHARD:-100000000}"
NUM_WORKERS="${NUM_WORKERS:-64}"
MAX_TOKENS="${MAX_TOKENS:-}"          # optional, e.g. 50000000 for quick smoke

TEXT_COLUMN="${TEXT_COLUMN:-text}"
TOKENIZE_BATCH_TEXTS="${TOKENIZE_BATCH_TEXTS:-256}"
QUEUE_BATCH_DOCS="${QUEUE_BATCH_DOCS:-512}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"  # set to 1 if your tokenizer requires it
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"    # set to 0 to allow downloading if missing locally

mkdir -p "$HF_HOME" "$TRAIN_OUTPUT_DIR"
if [[ "$VAL_TOKENS" != "0" ]]; then
  mkdir -p "$VAL_OUTPUT_DIR"
fi

echo "============================================================"
echo "Parquet -> Qwen2.5 .bin (int32 payload)"
echo "============================================================"
echo "Input:         $PARQUET_DIR"
echo "Tokenizer dir: $TOKENIZER_DIR"
echo "Train out:     $TRAIN_OUTPUT_DIR"
if [[ "$VAL_TOKENS" != "0" ]]; then
  echo "Val out:       $VAL_OUTPUT_DIR"
  echo "Val tokens:    $(printf "%'d" "$VAL_TOKENS")"
else
  echo "Val:           disabled (VAL_TOKENS=0)"
fi
echo "Shard size:    $(printf "%'d" "$TOKENS_PER_SHARD") tokens"
echo "Workers:       $NUM_WORKERS"
if [[ -n "$MAX_TOKENS" ]]; then
  echo "Max tokens:    $(printf "%'d" "$MAX_TOKENS")"
fi
echo "HF_HOME:       $HF_HOME"
echo "============================================================"

cmd=("$PYTHON_BIN" -u data/qwen25tokenize.py
  --parquet_dir "$PARQUET_DIR"
  --output_dir "$TRAIN_OUTPUT_DIR"
  --tokens_per_shard "$TOKENS_PER_SHARD"
  --num_workers "$NUM_WORKERS"
  --tokenizer_dir "$TOKENIZER_DIR"
  --hf_home "$HF_HOME"
  --text_column "$TEXT_COLUMN"
  --queue_batch_docs "$QUEUE_BATCH_DOCS"
  --tokenize_batch_texts "$TOKENIZE_BATCH_TEXTS"
)

if [[ "$VAL_TOKENS" != "0" ]]; then
  cmd+=(--val_output_dir "$VAL_OUTPUT_DIR" --val_tokens "$VAL_TOKENS")
fi

if [[ -n "$MAX_TOKENS" ]]; then
  cmd+=(--max_tokens "$MAX_TOKENS")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust_remote_code)
fi

if [[ "$LOCAL_FILES_ONLY" == "0" ]]; then
  cmd+=(--no_local_files_only)
fi

echo
echo "[RUN] ${cmd[*]}"
echo
exec "${cmd[@]}"

