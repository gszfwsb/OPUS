#!/bin/bash
# Download HuggingFaceFW/fineweb dataset with resume support.
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-fineweb_parquet}"
TARGET_TOKENS_B=${TARGET_TOKENS_B:-200.0}
BATCH_FILES=${BATCH_FILES:-10}
SUBSET="${SUBSET:-default}"

python -u data/download_fineweb.py \
    --repo_id "HuggingFaceFW/fineweb" \
    --output_dir "$OUTPUT_DIR" \
    --target_tokens_b $TARGET_TOKENS_B \
    --batch_files $BATCH_FILES \
    --subset "$SUBSET" \
    --continuous \
    --safe_download \
    --download_chunk_kb 512 \
    --download_fsync_mb 16 \
    --skip_in_progress

echo "Output: $OUTPUT_DIR"
echo "To resume, run this script again."
