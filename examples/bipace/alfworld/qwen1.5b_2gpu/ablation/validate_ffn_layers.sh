#!/usr/bin/env bash
# FFN-layer validation for Qwen2.5-1.5B-Instruct (single GPU; same as 4xH100 dir).
#
# Identical to examples/bipace/alfworld/qwen1.5b_2gpu/ablation/validate_ffn_layers.sh.
# Re-staged here so this 2-GPU 1.5B folder is self-contained.

set -e

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/val_outputs/ffn_layers_1.5b}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/validate_ffn_1.5b_${TS}.log"
ln -sf "$LOG_FILE" "$LOG_DIR/validate_ffn_1.5b.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] FFN-layer validation (1.5B, 2gpu folder) started, logging to $LOG_FILE"

set -x

OBS_SOURCE=${OBS_SOURCE:-synthetic}

python3 scripts/validate_ffn_layers.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --layers 0 4 8 12 16 20 24 27 \
    --pools last_token mean_prompt attn_weighted \
    --eps-grid 0.05 0.10 0.15 0.20 \
    --n-per-class 40 \
    --max-len 512 \
    --micro-bsz 8 \
    --dtype bfloat16 \
    --obs-source "$OBS_SOURCE" "$@"

set +x
echo "[$(date)] FFN-layer validation (1.5B) done. CSV at $OUTPUT_DIR/ffn_layer_validation.csv"
