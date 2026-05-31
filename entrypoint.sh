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

# Sanitize env vars: strip literal ${VAR:-default} patterns that may come from
# misconfigured deploy.yaml. Container runtimes do NOT expand bash substitution
# syntax, so we detect and strip it here as a safety net.
sanitize_env() {
    local varname="$1"
    local default="$2"
    local val="${!varname:-}"

    # If the value looks like a literal ${...:-...} pattern, replace with default
    if [[ "$val" =~ ^\$\{.*:-.*\}$ ]]; then
        log "WARNING: $varname contains unexpanded bash syntax '$val', using default '$default'"
        export "$varname=$default"
    fi
}

# Sanitize all configurable env vars
sanitize_env BASE_MODEL        "unsloth/qwen3.5-4b-instruct-bnb-4bit"
sanitize_env MAX_SEQ_LENGTH    "2048"
sanitize_env DATA_SOURCE       "lmms-lab/OlympiadBench"
sanitize_env DATASET_NAME      "lmms-lab/OlympiadBench"
sanitize_env DATASET_SUBSET_SIZE "5000"
sanitize_env SUBSET_SIZE       "5000"
sanitize_env DATASET_DOMAIN    "medical_qa"
sanitize_env LORA_RANK         "16"
sanitize_env LORA_ALPHA        "32"
sanitize_env LORA_DROPOUT      "0.05"
sanitize_env LEARNING_RATE     "2e-4"
sanitize_env NUM_EPOCHS        "3"
sanitize_env BATCH_SIZE        "4"
sanitize_env GRADIENT_ACCUMULATION_STEPS "4"
sanitize_env WARMUP_STEPS      "50"
sanitize_env SEED              "42"
sanitize_env PUSH_MERGED       "true"
sanitize_env PUSH_ADAPTERS     "true"
sanitize_env HOURLY_RATE       "0.75"
sanitize_env MAX_BUDGET        "3.00"
sanitize_env CONFIG_PATH       "configs/train_config.yaml"
sanitize_env OUTPUT_PATH       "/app/data/train.jsonl"

# Validate/defaults for HF upload variables (optional — training proceeds without them)
if [ -z "${HF_TOKEN:-}" ]; then
    log "WARNING: HF_TOKEN not set — model upload will be skipped"
    export HF_TOKEN=""
fi
if [ -z "${HF_REPO_ID:-}" ]; then
    log "WARNING: HF_REPO_ID not set — using default"
    export HF_REPO_ID="ToXMon/qwen3.5-4b-finetuned"
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
    --dataset "${DATASET_NAME}" \
    --subset-size "${SUBSET_SIZE}" \
    --output "${OUTPUT_PATH}" \
    --seed "${SEED}"

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
    --config "${CONFIG_PATH}" \
    ${MAX_STEPS:+--max-steps ${MAX_STEPS}}

TRAIN_EXIT=$?
if [ ${TRAIN_EXIT} -ne 0 ]; then
    log "ERROR: Training failed (exit ${TRAIN_EXIT})"
    exit ${TRAIN_EXIT}
fi

log "Training complete. Adapters saved to /app/output/lora_adapters/"

# Phase 3: Upload (skip if no HF_TOKEN)
log "=========================================="
log "Phase 3/3: Model Upload"
log "=========================================="
if [ -n "${HF_TOKEN:-}" ]; then
    python -m src.upload --config "${CONFIG_PATH}"
    UPLOAD_EXIT=$?
    if [ ${UPLOAD_EXIT} -ne 0 ]; then
        log "WARNING: Upload failed (exit ${UPLOAD_EXIT}), artifacts saved locally"
    fi
else
    log "WARNING: HF_TOKEN not set — skipping model upload"
    log "Artifacts available locally at /app/output/"
fi

# Summary
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))
SECS=$ELAPSED
ESTIMATED_COST=$(python3 -c "print(round($SECS / 3600 * ${HOURLY_RATE}, 2))")

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
