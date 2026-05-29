#!/usr/bin/env bash
#
# OPUS Proxy Construction Pipeline
#
# End-to-end pipeline that builds a high-quality proxy dataset for OPUS
# training data selection.  Three steps:
#
#   Step 1: Generate benchmark embeddings from evaluation tasks
#   Step 2: BETR-score every document in the training parquets
#   Step 3: Select the top-scored documents and write a proxy .bin shard
#
# Usage:
#   # Minimal (set required paths):
#   INPUT_ROOT=/path/to/fineweb_parquets \
#   OUTPUT_SCORED=/path/to/scored_parquets \
#   PROXY_OUTPUT=/path/to/proxy_bins \
#     bash data/run_build_proxy.sh
#
#   # Multi-GPU scoring:
#   INPUT_ROOT=... OUTPUT_SCORED=... PROXY_OUTPUT=... \
#   AUTO_MULTIGPU=1 GPU_LIST=0,1,2,3 \
#     bash data/run_build_proxy.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# ======================= Required paths =====================================

INPUT_ROOT="${INPUT_ROOT:?'Set INPUT_ROOT to your FineWeb parquet directory'}"
OUTPUT_SCORED="${OUTPUT_SCORED:?'Set OUTPUT_SCORED for BETR-scored parquet output'}"
PROXY_OUTPUT="${PROXY_OUTPUT:?'Set PROXY_OUTPUT for the final proxy .bin directory'}"

# ======================= Optional overrides =================================

BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-$ROOT_DIR/opencompass/data}"
BENCHMARK_NPZ="${BENCHMARK_NPZ:-$ROOT_DIR/benchmark_embeddings_arctic_l_v2.npz}"
ENCODER_MODEL="${ENCODER_MODEL:-Snowflake/snowflake-arctic-embed-l-v2.0}"
TARGET_TOKENS="${TARGET_TOKENS:-30000000}"

# Scoring settings
BATCH_SIZE="${BATCH_SIZE:-192}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

# Multi-GPU (passed through to Step 2)
AUTO_MULTIGPU="${AUTO_MULTIGPU:-0}"
GPU_LIST="${GPU_LIST:-}"

echo "============================================================"
echo "OPUS Proxy Construction Pipeline"
echo "============================================================"
echo "Input parquets:   $INPUT_ROOT"
echo "Scored parquets:  $OUTPUT_SCORED"
echo "Proxy output:     $PROXY_OUTPUT"
echo "Benchmark data:   $BENCHMARK_DATA_DIR"
echo "Benchmark NPZ:    $BENCHMARK_NPZ"
echo "Encoder:          $ENCODER_MODEL"
echo "Target tokens:    $TARGET_TOKENS"
echo "============================================================"
echo ""

# ======================== Step 1: Benchmark Embeddings ======================

if [[ -f "$BENCHMARK_NPZ" ]]; then
    echo "[Step 1] Benchmark embeddings already exist at $BENCHMARK_NPZ -- skipping."
else
    echo "[Step 1] Generating benchmark embeddings..."
    python -u data/embed_benchmarks.py \
        --data_dir "$BENCHMARK_DATA_DIR" \
        --encoder_model "$ENCODER_MODEL" \
        --output "$BENCHMARK_NPZ" \
        --batch_size 64 \
        --max_length "$MAX_LENGTH" \
        --device "$DEVICE" \
        --dtype "$DTYPE"
    echo "[Step 1] Done."
fi
echo ""

# ======================== Step 2: BETR Scoring ==============================

echo "[Step 2] BETR-scoring parquet files..."
export INPUT_ROOT OUTPUT_ROOT="$OUTPUT_SCORED" BENCHMARK_NPZ ENCODER_MODEL
export BATCH_SIZE MAX_LENGTH DTYPE DEVICE AUTO_MULTIGPU GPU_LIST
bash data/run_betr_score_fineweb_parquet.sh
echo "[Step 2] Done."
echo ""

# ======================== Step 3: Build Proxy .bin ==========================

echo "[Step 3] Building proxy .bin from top-scored documents..."
python -u data/build_betr_proxy.py \
    --input_root "$OUTPUT_SCORED" \
    --output_dir "$PROXY_OUTPUT" \
    --target_tokens "$TARGET_TOKENS"
echo "[Step 3] Done."
echo ""

echo "============================================================"
echo "Pipeline complete."
echo "Proxy .bin shards are in: $PROXY_OUTPUT"
echo "============================================================"
