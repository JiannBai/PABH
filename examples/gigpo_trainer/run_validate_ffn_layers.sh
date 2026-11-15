#!/usr/bin/env bash
# P0 (4) — FFN-layer validation experiment for BiGPO.
#
# Standalone single-GPU run. Loads Qwen2.5-7B-Instruct in eval mode,
# forwards a synthetic ALFWorld audit set covering all 6 task families,
# and reports per-(layer, pool) clustering quality + linear-probe
# task-id accuracy. Answers: which hidden layer / which pooling
# position is the best fingerprint for BiGPO?
#
# Cost: ~2–3 minutes on one H100. No FSDP, no Ray, no env workers.

set -e

# ---- User paths (mirrors the BiGPO training script) -------------------
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/val_outputs/ffn_layers}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# Single-GPU only — pick the first free device.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Minimal logging (mirrors training-script convention).
LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/validate_ffn_${TS}.log"
ln -sf "$LOG_FILE" "$LOG_DIR/validate_ffn.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] FFN-layer validation started, logging to $LOG_FILE"

set -x

# OBS_SOURCE:
#   "synthetic" (default) — built-in 6 task families × 40 templates.
#   <path>                 — directory of qualitative_*.txt or one such file
#                            (real anchor obs dumped during BiGPO training).
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
echo "[$(date)] FFN-layer validation done. CSV at $OUTPUT_DIR/ffn_layer_validation.csv"
