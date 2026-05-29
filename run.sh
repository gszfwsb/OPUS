#!/bin/bash
# ============================================================================
# Train GPT-2 XL from scratch with OPUS data selection.
# Supports both single-node and multi-node torchrun launches.
#
# Single node example:
#   bash run_main.sh
#
# Multi-node example (platform launches the same command on all nodes):
#   export OUTPUT_BASE=/mnt/data/your_shared_dir
#   export RUN_NAME=opus_gpt2xl_32gpu_port29501
#   export NNODES=4 GPUS_PER_NODE=8 MASTER_PORT=29501
#   # Preferred: platform injects MASTER_ADDR / NODE_RANK
#   bash run_main.sh
#   # If the platform does not inject them, set them manually.
# ============================================================================

unset EXPERIMENT_NAME
set -euo pipefail
trap 'pkill -TERM -P $$ >/dev/null 2>&1 || true; sleep 1; pkill -KILL -P $$ >/dev/null 2>&1 || true' EXIT INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/OPUS:${PYTHONPATH:-}"

MODEL_TYPE="${MODEL_TYPE:-gpt2-xl}"
DATASET="fineweb_edu3plus_custom"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-muon_hybrid}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-$((2*1024))}
VAL_SEQ_LEN=${VAL_SEQ_LEN:-$((8*1024))}
TOTAL_TOKENS_B=${TOTAL_TOKENS_B:-30.0}
EVAL_EVERY_TOKENS_B=${EVAL_EVERY_TOKENS_B:-1.0}
CHECKPOINT_EVERY_TOKENS_B=${CHECKPOINT_EVERY_TOKENS_B:-$EVAL_EVERY_TOKENS_B}

# Launch configuration.
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NUM_GPUS:-8}}"
NUM_GPUS="${NUM_GPUS:-$GPUS_PER_NODE}"
OUTPUT_BASE="${OUTPUT_BASE:-}"
RUN_NAME="${RUN_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
NODE_RANK="${NODE_RANK:-}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [ -z "${OUTPUT_ROOT}" ]; then
    if [ -n "${OUTPUT_BASE}" ] && [ -n "${RUN_NAME}" ]; then
        OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_NAME}"
    elif [ -n "${OUTPUT_BASE}" ]; then
        OUTPUT_ROOT="${OUTPUT_BASE}"
    else
        OUTPUT_ROOT="."
    fi
fi

get_first_nonempty_env() {
    local env_name=""
    local env_value=""
    for env_name in "$@"; do
        env_value="${!env_name:-}"
        if [ -n "${env_value}" ]; then
            printf '%s' "${env_value}"
            return 0
        fi
    done
    return 1
}

if [ -z "${GPU_DEVICES:-}" ]; then
    GPU_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
else
    GPU_DEVICES="${GPU_DEVICES}"
fi

if [ -z "${NODE_RANK}" ]; then
    NODE_RANK="$(get_first_nonempty_env GROUP_RANK SLURM_NODEID OMPI_COMM_WORLD_NODE_RANK OMPI_MCA_orte_ess_node_rank VC_TASK_INDEX RANK_INDEX NODE_INDEX JOB_COMPLETION_INDEX 2>/dev/null || true)"
fi

if [ -z "${NODE_RANK}" ] && [ -n "${RANK:-}" ] && [ "${RANK}" -lt "${NNODES}" ]; then
    NODE_RANK="${RANK}"
fi

if [ -z "${NODE_RANK}" ]; then
    HOSTNAME_VALUE="$(hostname)"
    if [[ "${HOSTNAME_VALUE}" =~ (worker|node)-([0-9]+)$ ]]; then
        NODE_RANK="${BASH_REMATCH[2]}"
        echo "Auto-detected NODE_RANK=${NODE_RANK} from hostname=${HOSTNAME_VALUE}"
    fi
fi

if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR="$(get_first_nonempty_env MASTER_HOST PET_MASTER_ADDR CHIEF_IP CHIEF_HOST 2>/dev/null || true)"
fi

if [ "${NNODES}" -gt 1 ] && { [ -z "${MASTER_ADDR}" ] || [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; }; then
    HOSTNAME_VALUE="$(hostname)"
    if [[ "${HOSTNAME_VALUE}" =~ ^(.+-(worker|node)-)[0-9]+$ ]]; then
        MASTER_ADDR="${BASH_REMATCH[1]}0"
        echo "Auto-detected MASTER_ADDR=${MASTER_ADDR} from hostname pattern"
    fi
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

if [ "${NNODES}" -gt 1 ]; then
    if [ -z "${NODE_RANK}" ]; then
        echo "ERROR: failed to infer NODE_RANK for multi-node launch." >&2
        echo "Please rely on platform injection or export NODE_RANK manually." >&2
        exit 1
    fi
    if [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; then
        echo "ERROR: failed to infer MASTER_ADDR for multi-node launch." >&2
        echo "Please rely on platform injection or export MASTER_ADDR=<node0_ip> manually." >&2
        exit 1
    fi
fi

if [ "${NNODES}" -gt 1 ]; then
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-23}"
    export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
    export NCCL_ALGO="${NCCL_ALGO:-Tree}"
    export NCCL_PROTO="${NCCL_PROTO:-Simple}"
    export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"
    export NCCL_NTHREADS="${NCCL_NTHREADS:-512}"
    export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
    export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
    export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
    export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
        if ip link show eth0 >/dev/null 2>&1; then
            export NCCL_SOCKET_IFNAME=eth0
        elif ip link show bond0 >/dev/null 2>&1; then
            export NCCL_SOCKET_IFNAME=bond0
        fi
    fi
fi

# Adam hyperparams (defaults match train.py); required because set -u is enabled
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}
ADAM_WEIGHT_DECAY=${ADAM_WEIGHT_DECAY:-0.1}

# OPUS knobs
OPUS_SCORE_LEN=${OPUS_SCORE_LEN:-512}
OPUS_PROXY_BATCH=${OPUS_PROXY_BATCH:-8}
OPUS_SELECTION_RATIO=${OPUS_SELECTION_RATIO:-0.25}
OPUS_BUFFER_MULTIPLIER=${OPUS_BUFFER_MULTIPLIER:-16}
OPUS_PRECONDITIONER=${OPUS_PRECONDITIONER:-"auto"}
OPUS_TEMPERATURE=${OPUS_TEMPERATURE:-0.9}

# Random Projection (optional)
USE_RANDOM_PROJECTION=${USE_RANDOM_PROJECTION:-1}
PROJECTION_DIM=${PROJECTION_DIM:-8192}
PROJECTION_SEED=${PROJECTION_SEED:-42}

RATIO_PCT=$(python3 -c "print(int(${OPUS_SELECTION_RATIO} * 100))")
if [ -n "${RUN_NAME}" ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${RUN_NAME}"}
elif [ "$USE_RANDOM_PROJECTION" = "1" ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-"OPUS_${OPTIMIZER_TYPE}_${MODEL_TYPE}_S${OPUS_SCORE_LEN}_B${OPUS_BUFFER_MULTIPLIER}_R${RATIO_PCT}_PB${OPUS_PROXY_BATCH}_RP${PROJECTION_DIM}_T${OPUS_TEMPERATURE}_${TOTAL_TOKENS_B}B"}
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-"OPUS_${OPTIMIZER_TYPE}_${MODEL_TYPE}_S${OPUS_SCORE_LEN}_B${OPUS_BUFFER_MULTIPLIER}_R${RATIO_PCT}_PB${OPUS_PROXY_BATCH}_T${OPUS_TEMPERATURE}_${TOTAL_TOKENS_B}B"}
fi

# Data
DATA_ROOT="${DATA_ROOT:-./bins}"
TRAIN_SCORES="${DATA_ROOT}/fineweb_200B_train"
VAL_SCORES="${DATA_ROOT}/fineweb_200B_val"
PROXY_SCORES="${DATA_ROOT}/fineweb_betr_proxy_top30M"
DATA_SHUFFLE_SEED=42

echo "============================================================"
echo "Train from Scratch Configuration"
echo "============================================================"
echo "Model: $MODEL_TYPE (custom architecture)"
echo "Optimizer: $OPTIMIZER_TYPE"
echo "Learning Rates: default (adam=0.002, muon=0.010)"
echo "Adam betas: ($ADAM_BETA1, $ADAM_BETA2)  weight_decay=$ADAM_WEIGHT_DECAY"
echo "Train Seq Len: $((TRAIN_SEQ_LEN/1024))K"
echo "Total Tokens: ${TOTAL_TOKENS_B}B"
echo "Checkpoint Every: ${CHECKPOINT_EVERY_TOKENS_B}B"
echo "Output Root: ${OUTPUT_ROOT}"
echo "Run Name: ${RUN_NAME:-<auto>}"
echo "Train Script: ${TRAIN_SCRIPT}"
echo "GPU Devices: ${GPU_DEVICES}"
if [ "${NNODES}" -eq 1 ]; then
    echo "Distributed: single-node (torchrun --standalone), GPUs this node: ${GPUS_PER_NODE}"
else
    echo "Distributed: multi-node"
    echo "Node Rank: ${NODE_RANK}/${NNODES}"
    echo "GPUs per node: ${GPUS_PER_NODE}"
    echo "World Size: ${WORLD_SIZE}"
    echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
    if [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; then
        echo "WARNING: set MASTER_ADDR to node0 private IP for multi-node training" >&2
    fi
fi
echo "============================================================"

mkdir -p "${OUTPUT_ROOT}/logs"

if [ "${NNODES}" -eq 1 ]; then
    LAUNCH_ARGS="--standalone --nproc_per_node=${GPUS_PER_NODE}"
else
    LAUNCH_ARGS="--nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU_DEVICES torchrun ${LAUNCH_ARGS} ${TRAIN_SCRIPT} \
    --model_type $MODEL_TYPE \
    --dataset $DATASET \
    --optimizer_type $OPTIMIZER_TYPE \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --adam_weight_decay $ADAM_WEIGHT_DECAY \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --total_tokens_b $TOTAL_TOKENS_B \
    --eval_every_tokens_b $EVAL_EVERY_TOKENS_B \
    --checkpoint_every_tokens_b $CHECKPOINT_EVERY_TOKENS_B \
    --eval_mode inline \
    --train_files \"${TRAIN_SCORES}/*.bin\" \
    --val_files \"${VAL_SCORES}/*.bin\" \
    --data_shuffle_seed $DATA_SHUFFLE_SEED \
    --output_root \"$OUTPUT_ROOT\" \
    --use_opus \
    --selection_strategy opus \
    --opus_selection_method stochastic \
    --opus_preconditioner $OPUS_PRECONDITIONER \
    --opus_buffer_size_multiplier $OPUS_BUFFER_MULTIPLIER \
    --opus_selection_ratio $OPUS_SELECTION_RATIO \
    --opus_temperature $OPUS_TEMPERATURE \
    --opus_score_len $OPUS_SCORE_LEN \
    --opus_proxy_batch $OPUS_PROXY_BATCH \
    --opus_proxy_dir \"${PROXY_SCORES}/*.bin\" \
    --opus_proxy_tokens 30000000"

if [ "$USE_RANDOM_PROJECTION" = "1" ]; then
    CMD="$CMD \
    --use_random_projection \
    --projection_dim $PROJECTION_DIM \
    --projection_seed $PROJECTION_SEED"
    echo "Random Projection ENABLED: dim=$PROJECTION_DIM, seed=$PROJECTION_SEED"
else
    echo "Random Projection DISABLED (full-dim gradients)"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    CMD="$CMD --experiment_name \"$EXPERIMENT_NAME\""
fi

# ============================================================
# Auto Resume from Latest Checkpoint
# ============================================================
CHECKPOINT_DIR="${OUTPUT_ROOT}/logs/${EXPERIMENT_NAME}"
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -1 ${CHECKPOINT_DIR}/state_step*.pt 2>/dev/null | sort -t'p' -k2 -n | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        STEP_NUM=$(basename "$LATEST_CHECKPOINT" | sed 's/state_step\([0-9]*\)\.pt/\1/')
        echo "============================================================"
        echo "AUTO RESUME DETECTED"
        echo "Found: $LATEST_CHECKPOINT (step $STEP_NUM)"
        echo "============================================================"
        CMD="$CMD --resume_from_checkpoint \"$LATEST_CHECKPOINT\""
    fi
fi

echo "Running: $CMD"
eval $CMD
