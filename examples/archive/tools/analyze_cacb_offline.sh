#!/usr/bin/env bash
# One-shot offline CACB partition analysis.
# Generates G rollouts per prompt and reports cacb/rloo/singleton mix
# under both first_n and action_tag key modes — the question the
# in-trainer CACB_VERBOSE log would answer, but offline.

set -e

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
# Default = empty → script uses internal synthetic ALFWorld-style prompts.
# To use the verl test set, run with:
#   PROMPTS=$HOME/data/verl-agent/text/test.parquet bash run_analyze_cacb_offline.sh
# (Run examples/data_preprocess/prepare_offline.py first if it doesn't exist.)
PROMPTS=${PROMPTS:-}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/val_outputs/cacb_offline_1.5b}
N_PROMPTS=${N_PROMPTS:-16}
GROUP_SIZE=${GROUP_SIZE:-8}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/analyze_cacb_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] CACB offline analysis started, logging to $LOG_FILE"

set -x

PROMPTS_ARG=()
if [ -n "$PROMPTS" ]; then
    PROMPTS_ARG=(--prompts "$PROMPTS")
fi

python3 scripts/analyze_cacb_offline.py \
    --model-path "$MODEL_PATH" \
    "${PROMPTS_ARG[@]}" \
    --n-prompts "$N_PROMPTS" \
    --group-size "$GROUP_SIZE" \
    --max-new-tokens 256 \
    --temperature 1.0 \
    --top-p 0.95 \
    --output "$OUTPUT_DIR" "$@"

set +x
echo "[$(date)] Done. See $OUTPUT_DIR/summary.txt"
