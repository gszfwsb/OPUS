#!/bin/bash
# Qwen3-8B CPT with OPUS data selection (Muon-Pretrain framework).
set -euo pipefail
unset EXPERIMENT_NAME
export PYTHONPATH=$PWD:$PWD/OPUS:$PYTHONPATH
trap 'pkill -TERM -P $$ >/dev/null 2>&1 || true; sleep 1; pkill -KILL -P $$ >/dev/null 2>&1 || true' EXIT INT TERM

# Model and Training
MODEL_TYPE="qwen3-8b"
OPTIMIZER_TYPE="muon_hybrid"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"

TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-4096}
VAL_SEQ_LEN=${VAL_SEQ_LEN:-4096}
ATTENTION_MODE="${ATTENTION_MODE:-flex}"

TOTAL_TOKENS_B=${TOTAL_TOKENS_B:-2}
WARMUP_FRAC=${WARMUP_FRAC:-0.01}
EVAL_EVERY_TOKENS_B=${EVAL_EVERY_TOKENS_B:-0.5}
CHECKPOINT_EVERY_TOKENS_B=${CHECKPOINT_EVERY_TOKENS_B:-$EVAL_EVERY_TOKENS_B}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}

NUM_GPUS=${NUM_GPUS:-8}
GPU_DEVICES=${GPU_DEVICES:-"0,1,2,3,4,5,6,7"}
OUTPUT_ROOT="${OUTPUT_ROOT:-.}"

# Loss Mask
USE_LOSS_MASK="${USE_LOSS_MASK:-1}"
LOSS_MASK_SUFFIX="${LOSS_MASK_SUFFIX:-.lossmask}"

# Optimizer and Schedule
ADAM_LR=${ADAM_LR:-1e-6}
MUON_LR=${MUON_LR:-1e-5}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}
ADAM_WEIGHT_DECAY=${ADAM_WEIGHT_DECAY:-0.01}
LR_SCHEDULE=${LR_SCHEDULE:-legacy}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}

# Data
DATA_ROOT_BASE=${DATA_ROOT_BASE:-"cpt_scipedia_qwen3_8b_bins_lossmask"}
PART0_DIR="${PART0_DIR:-${DATA_ROOT_BASE}/CPT_scipedia}"
PART1_DIR="${PART1_DIR:-${DATA_ROOT_BASE}/CPT_scipedia_part1}"
PART2_DIR="${PART2_DIR:-${DATA_ROOT_BASE}/CPT_scipedia_part2}"

DOMAIN_ROOT_DIR="${DOMAIN_ROOT_DIR:-${PART0_DIR},${PART1_DIR},${PART2_DIR}}"
VAL_DIRS="${VAL_DIRS:-${PART0_DIR}/val,${PART1_DIR}/val,${PART2_DIR}/val}"

INIT_MODEL=${INIT_MODEL:-"Qwen/Qwen3-8B-Base"}

# OPUS Hyperparameters
USE_OPUS="${USE_OPUS:-1}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-opus}"
OPUS_SELECTION_METHOD="${OPUS_SELECTION_METHOD:-stochastic}"
OPUS_PRECONDITIONER="${OPUS_PRECONDITIONER:-auto}"
OPUS_BUFFER_MULTIPLIER="${OPUS_BUFFER_MULTIPLIER:-16}"
OPUS_SELECTION_RATIO="${OPUS_SELECTION_RATIO:-0.5}"
OPUS_TEMPERATURE="${OPUS_TEMPERATURE:-0.8}"
OPUS_SCORE_LEN="${OPUS_SCORE_LEN:-512}"
OPUS_PROXY_BATCH="${OPUS_PROXY_BATCH:-4}"
OPUS_N_WINDOWS="${OPUS_N_WINDOWS:-1}"
OPUS_PROXY_MODE="${OPUS_PROXY_MODE:-refresh}"
OPUS_PROXY_REFRESH_INTERVAL="${OPUS_PROXY_REFRESH_INTERVAL:-1}"
OPUS_PROXY_TOKENS="${OPUS_PROXY_TOKENS:-30000000}"
OPUS_GLOBAL_SELECTION="${OPUS_GLOBAL_SELECTION:-0}"

PROXY_PATTERN_DEFAULT="${PART0_DIR}/val/*.bin,${PART1_DIR}/val/*.bin,${PART2_DIR}/val/*.bin"
PROXY_PATTERN="${PROXY_PATTERN:-${PROXY_PATTERN_DEFAULT}}"

USE_RANDOM_PROJECTION="${USE_RANDOM_PROJECTION:-0}"
PROJECTION_DIM="${PROJECTION_DIM:-8192}"
PROJECTION_SEED="${PROJECTION_SEED:-42}"

# Experiment Name
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"Qwen3_8B_CPT_OPUS_${SELECTION_STRATEGY}_${OPUS_SELECTION_METHOD}_R${OPUS_SELECTION_RATIO}_T${OPUS_TEMPERATURE}_S${TRAIN_SEQ_LEN}_GA${GRAD_ACCUM_STEPS}_ALR${ADAM_LR}_MLR${MUON_LR}_TOK${TOTAL_TOKENS_B}B"}

# Print Config
echo "============================================================"
echo " Qwen3-8B CPT | OPUS Data Selection"
echo "============================================================"
echo "model=$MODEL_TYPE  optimizer=$OPTIMIZER_TYPE  init=$INIT_MODEL"
echo "seq_len=$TRAIN_SEQ_LEN  attention=$ATTENTION_MODE  tokens=${TOTAL_TOKENS_B}B  warmup=$WARMUP_FRAC"
echo "adam_lr=$ADAM_LR  muon_lr=$MUON_LR  schedule=$LR_SCHEDULE  clip=$GRAD_CLIP_NORM  ga=$GRAD_ACCUM_STEPS"
echo "gpus=$NUM_GPUS  loss_mask=$USE_LOSS_MASK"
echo "output_root=$OUTPUT_ROOT  checkpoint_every=${CHECKPOINT_EVERY_TOKENS_B}B"
if [[ "${USE_OPUS}" == "1" ]]; then
  echo "opus: strategy=$SELECTION_STRATEGY  method=$OPUS_SELECTION_METHOD  precon=$OPUS_PRECONDITIONER"
  echo "      buffer=${OPUS_BUFFER_MULTIPLIER}x  ratio=$OPUS_SELECTION_RATIO  temp=$OPUS_TEMPERATURE  score_len=$OPUS_SCORE_LEN"
  echo "      proxy_bs=$OPUS_PROXY_BATCH  n_win=$OPUS_N_WINDOWS  global=$OPUS_GLOBAL_SELECTION  rand_proj=$USE_RANDOM_PROJECTION"
fi
echo "experiment: $EXPERIMENT_NAME"
echo "============================================================"

mkdir -p "${OUTPUT_ROOT}/logs"

# Build Extra Args
EXTRA_ARGS=""
if [[ "${USE_LOSS_MASK}" == "1" ]]; then
  EXTRA_ARGS=" --use_loss_mask --loss_mask_suffix ${LOSS_MASK_SUFFIX}"
fi

if [[ "${USE_OPUS}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --use_opus"
  EXTRA_ARGS="${EXTRA_ARGS} --selection_strategy ${SELECTION_STRATEGY}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_selection_method ${OPUS_SELECTION_METHOD}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_preconditioner ${OPUS_PRECONDITIONER}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_buffer_size_multiplier ${OPUS_BUFFER_MULTIPLIER}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_selection_ratio ${OPUS_SELECTION_RATIO}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_temperature ${OPUS_TEMPERATURE}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_score_len ${OPUS_SCORE_LEN}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_proxy_batch ${OPUS_PROXY_BATCH}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_n_windows ${OPUS_N_WINDOWS}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_proxy_dir \"${PROXY_PATTERN}\""
  EXTRA_ARGS="${EXTRA_ARGS} --opus_proxy_tokens ${OPUS_PROXY_TOKENS}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_proxy_mode ${OPUS_PROXY_MODE}"
  EXTRA_ARGS="${EXTRA_ARGS} --opus_proxy_refresh_interval ${OPUS_PROXY_REFRESH_INTERVAL}"

  if [[ "${OPUS_GLOBAL_SELECTION}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --opus_global_selection"
  fi

  if [[ "${USE_RANDOM_PROJECTION}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_random_projection --projection_dim ${PROJECTION_DIM} --projection_seed ${PROJECTION_SEED}"
  fi
fi

# Launch
CMD="CUDA_VISIBLE_DEVICES=$GPU_DEVICES torchrun --standalone --nproc_per_node=$NUM_GPUS ${TRAIN_SCRIPT} \
  --model_type $MODEL_TYPE \
  --dataset fineweb_edu3plus_custom \
  --optimizer_type $OPTIMIZER_TYPE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --attention_mode $ATTENTION_MODE \
  --grad_accum_steps $GRAD_ACCUM_STEPS \
  --total_tokens_b $TOTAL_TOKENS_B \
  --eval_every_tokens_b $EVAL_EVERY_TOKENS_B \
  --checkpoint_every_tokens_b $CHECKPOINT_EVERY_TOKENS_B \
  --warmup_frac $WARMUP_FRAC \
  --grad_clip_norm $GRAD_CLIP_NORM \
  --lr_schedule $LR_SCHEDULE \
  --min_lr_ratio $MIN_LR_RATIO \
  --adam_lr $ADAM_LR \
  --muon_lr $MUON_LR \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --adam_weight_decay $ADAM_WEIGHT_DECAY \
  --eval_mode inline \
  --domain_root_dir \"${DOMAIN_ROOT_DIR}\" \
  --val_dir \"${VAL_DIRS}\" \
  --output_root \"$OUTPUT_ROOT\" \
  --init_model $INIT_MODEL \
  --experiment_name \"$EXPERIMENT_NAME\"${EXTRA_ARGS}"

# Auto Resume
CHECKPOINT_DIR="${OUTPUT_ROOT}/logs/${EXPERIMENT_NAME}"
if [ -d "$CHECKPOINT_DIR" ]; then
  LATEST_CHECKPOINT=$(ls -1 ${CHECKPOINT_DIR}/state_step*.pt 2>/dev/null | sort -t'p' -k2 -n | tail -1)
  if [ -n "${LATEST_CHECKPOINT:-}" ]; then
    echo "Auto-resuming from: $LATEST_CHECKPOINT"
    CMD="$CMD --resume_from_checkpoint \"$LATEST_CHECKPOINT\""
  fi
fi

echo "$CMD"
eval $CMD
