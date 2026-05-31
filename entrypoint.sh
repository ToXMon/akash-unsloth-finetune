#!/usr/bin/env bash
# Akash Unsloth Fine-tune Demo - Main Entrypoint
# Orchestrates: prepare_data → train → upload
set -euo pipefail

# Error handler
on_error() {
    local exit_code=$?
    local line_no=$1
    echo "ERROR: Script failed at line ${line_no} with exit code ${exit_code}" >&2
    echo "ERROR: Check logs at /app/logs/training.log" >&2
    exit ${exit_code}
}
trap 'on_error ${LINENO}' ERR

# Timestamps
START_TIME=$(date +%s)

log() {
    echo "[$(date -Iseconds)] $*"
}

log "=========================================="
log "Akash Unsloth Fine-tune Demo Starting"
log "=========================================="

# Validate required environment variables
MISSING_VARS=""
for var in HF_TOKEN HF_REPO_ID; do
    if [ -z "${!var:-}" ]; then
        MISSING_VARS="${MISSING_VARS}  - ${var}\n"
    fi
done

if [ -n "${MISSING_VARS}" ]; then
    echo "ERROR: Missing required environment variables:" >&2
    echo -e "${MISSING_VARS}" >&2
    exit 1
fi

log "Environment validated: HF_REPO_ID=${HF_REPO_ID}"

# Create directories
mkdir -p /app/data /app/output/lora_adapters /app/output/merged_model /app/logs

# GPU Check
log "Checking GPU availability..."
python -c "from src.utils import check_gpu; import logging; logging.basicConfig(level=logging.INFO); info = check_gpu(); print(f'GPU: {info}')" 2>&1 || {
    echo "ERROR: GPU check failed. Ensure NVIDIA GPU and drivers are available." >&2
    exit 1
}

# Phase 1: Data Preparation
log "=========================================="
log "Phase 1/3: Data Preparation"
log "=========================================="
python -m src.prepare_data \
    --dataset "${DATASET_NAME:-OpenBMB/OlympiadBench}" \
    --subset-size "${SUBSET_SIZE:-5000}" \
    --output "${OUTPUT_PATH:-/app/data/train.jsonl}" \
    --seed "${SEED:-42}"

DATA_EXIT=$?
if [ ${DATA_EXIT} -ne 0 ]; then
    log "ERROR: Data preparation failed (exit ${DATA_EXIT})"
    exit ${DATA_EXIT}
fi

# Verify training data exists
if [ ! -f "/app/data/train.jsonl" ]; then
    log "ERROR: Training data not found at /app/data/train.jsonl"
    exit 1
fi
SAMPLE_COUNT=$(wc -l < /app/data/train.jsonl)
log "Training data ready: ${SAMPLE_COUNT} samples"

# Phase 2: Training
log "=========================================="
log "Phase 2/3: QLoRA Fine-tuning"
log "=========================================="
python -m src.train \
    --config "configs/train_config.yaml" \
    ${MAX_STEPS:+--max-steps ${MAX_STEPS}}

TRAIN_EXIT=$?
if [ ${TRAIN_EXIT} -ne 0 ]; then
    log "ERROR: Training failed (exit ${TRAIN_EXIT})"
    exit ${TRAIN_EXIT}
fi

log "Training complete. Adapters saved to /app/output/lora_adapters/"

# Phase 3: Upload
log "=========================================="
log "Phase 3/3: Model Upload"
log "=========================================="
python -m src.upload --config "configs/train_config.yaml"

UPLOAD_EXIT=$?
if [ ${UPLOAD_EXIT} -ne 0 ]; then
    log "WARNING: Upload failed (exit ${UPLOAD_EXIT}), artifacts saved locally"
fi

# Summary
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))
SECS=$ELAPSED
ESTIMATED_COST=$(python3 -c "print(round($SECS / 3600 * 0.75, 2))")

log "=========================================="
log "Pipeline Complete!"
log "=========================================="
log "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log "Estimated cost: ~\$${ESTIMATED_COST}"
log "Artifacts:"
log "  LoRA adapters: /app/output/lora_adapters/"
log "  Merged model:  /app/output/merged_model/"
log "  Logs:          /app/logs/training.log"
log "=========================================="

exit 0
