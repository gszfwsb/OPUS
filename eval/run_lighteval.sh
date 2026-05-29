#!/bin/bash
# =============================================================================
# LightEval Offline Evaluation Runner (Multi-GPU + Batch Processing)
# =============================================================================
# 
# Usage:
#   ./run_lighteval.sh <checkpoint_path> [model_type] [tasks] [max_samples] [num_gpus] [batch_size]
#
# Examples:
#   # Single GPU evaluation
#   ./run_lighteval.sh logs/experiment/state_step010000.pt gpt2-xl mmlu
#
#   # Multi-GPU evaluation (2 GPUs, batch_size=128 for H200)
#   ./run_lighteval.sh logs/experiment/state_step010000.pt gpt2-xl mmlu "" 2 128
#
#   # Quick test with limited samples
#   ./run_lighteval.sh logs/experiment/state_step010000.pt gpt2-xl all 100
#
#   # High throughput (2x H200, batch_size=256)
#   ./run_lighteval.sh logs/experiment/state_step010000.pt gpt2-xl mmlu "" 2 256
#
# Environment:
#   GPU_IDS: Comma-separated GPU IDs (e.g., "0,1"). Overrides num_gpus.
#   BATCH_SIZE: Override batch size via environment variable.
#
# =============================================================================

set -e

# Set offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/OPUS:${PYTHONPATH}"
export TIKTOKEN_CACHE_DIR="${SCRIPT_DIR}/.cache/tiktoken"

# Parse arguments
CHECKPOINT="${1:-}"
MODEL_TYPE="${2:-gpt2-xl}"
TASKS="${3:-mmlu}"
MAX_SAMPLES="${4:-}"
NUM_GPUS="${5:-2}"
# Use env var BATCH_SIZE if set, otherwise use arg or default to 128 for H200
BATCH_SIZE="${BATCH_SIZE:-${6:-16}}"

# Validation
if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 <checkpoint_path> [model_type] [tasks] [max_samples] [num_gpus] [batch_size]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path  Path to .pt checkpoint file"
    echo "  model_type       Model architecture (gpt2, gpt2-medium, gpt2-large, gpt2-xl)"
    echo "  tasks            Comma-separated tasks or 'all'"
    echo "  max_samples      Max samples per task (for quick testing)"
    echo "  num_gpus         Number of GPUs for parallel evaluation (default: 2)"
    echo "  batch_size       Batch size for inference (default: 128, use 256+ for H200)"
    echo ""
    echo "Available tasks:"
    echo "  hellaswag, winogrande, piqa, siqa, arc_easy, arc_challenge,"
    echo "  commonsenseqa, openbookqa, mmlu"
    echo ""
    echo "Examples:"
    echo "  # Single GPU with default batch size"
    echo "  $0 logs/my_exp/state_step005000.pt gpt2-xl mmlu"
    echo ""
    echo "  # 2x H200 GPUs with batch_size=256"
    echo "  $0 logs/my_exp/state_step005000.pt gpt2-xl mmlu \"\" 2 256"
    echo ""
    echo "  # Via environment variables"
    echo "  GPU_IDS=0,1 BATCH_SIZE=256 $0 logs/my_exp/state_step005000.pt gpt2-xl mmlu"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Convert to absolute path
CHECKPOINT=$(realpath "$CHECKPOINT")

# Determine GPU configuration
if [ -n "$GPU_IDS" ]; then
    # Use explicit GPU IDs from environment
    GPU_ARG="--gpu_ids $GPU_IDS"
    NUM_GPUS_DISPLAY="$GPU_IDS"
elif [ "$NUM_GPUS" -gt 1 ]; then
    # Use num_gpus argument
    GPU_ARG="--num_gpus $NUM_GPUS"
    NUM_GPUS_DISPLAY="$NUM_GPUS GPUs (0-$((NUM_GPUS-1)))"
else
    # Single GPU mode
    GPU_ARG="--num_gpus 1"
    NUM_GPUS_DISPLAY="1 (single GPU)"
fi

echo "============================================================"
echo "LightEval Offline Evaluation (Multi-GPU + Batch Processing)"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Model type: $MODEL_TYPE"
echo "Tasks: $TASKS"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "GPUs: $NUM_GPUS_DISPLAY"
echo "Batch size: $BATCH_SIZE"
echo "============================================================"

# Build command
CMD="python ${SCRIPT_DIR}/lighteval.py \
    --checkpoint \"$CHECKPOINT\" \
    --model_type $MODEL_TYPE \
    --tasks \"$TASKS\" \
    --batch_size $BATCH_SIZE \
    $GPU_ARG"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Extract experiment name for output
CKPT_NAME=$(basename "$CHECKPOINT" .pt)
EXP_DIR=$(dirname "$CHECKPOINT")
EXP_NAME=$(basename "$EXP_DIR")
CMD="$CMD --output_dir \"${SCRIPT_DIR}/lighteval_results/${EXP_NAME}\""

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "Results: ${SCRIPT_DIR}/lighteval_results/${EXP_NAME}/"
echo "============================================================"

