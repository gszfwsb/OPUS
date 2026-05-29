#!/bin/bash
# Convert FineWeb parquet files to GPT-2 .bin format for training.
set -euo pipefail

export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-.cache/tiktoken}"

PARQUET_DIR="${PARQUET_DIR:-fineweb_parquet}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-bins/fineweb_train}"
VAL_OUTPUT_DIR="${VAL_OUTPUT_DIR:-bins/fineweb_val}"
VAL_TOKENS=${VAL_TOKENS:-200000000}
TOKENS_PER_SHARD=${TOKENS_PER_SHARD:-100000000}
NUM_WORKERS=${NUM_WORKERS:-64}

echo "============================================================"
echo "Parquet -> GPT-2 .bin"
echo "============================================================"
echo "Input:      $PARQUET_DIR"
echo "Train out:  $TRAIN_OUTPUT_DIR"
echo "Val out:    $VAL_OUTPUT_DIR"
echo "Val tokens: $(printf "%'d" $VAL_TOKENS)"
echo "Workers:    $NUM_WORKERS"
echo "============================================================"

python -u data/gpt2tokenize.py \
    --parquet_dir "$PARQUET_DIR" \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --val_output_dir "$VAL_OUTPUT_DIR" \
    --val_tokens $VAL_TOKENS \
    --tokens_per_shard $TOKENS_PER_SHARD \
    --num_workers $NUM_WORKERS
