#!/usr/bin/env bash
# Multi-seed wrapper for TextCraft GRPO 1.5B on 4xH100.
#
# Runs the single-seed script sequentially with seeds 1 and 2 by default.
# Override with:
#   SEEDS="3 4" bash examples/bipace/textcraft/qwen1.5b_4xh100/ablation/grpo_3seeds.sh
#
# Any arguments are forwarded to run_textcraft_grpo_qwen1.5b_4xh100.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_SEED_SCRIPT=./examples/bipace/textcraft/qwen1.5b_4xh100/train_grpo.sh
SEEDS=${SEEDS:-"0 1 2"}


echo "[$(date)] Starting TextCraft GRPO 1.5B multi-seed run: $SEEDS"

for seed in $SEEDS; do
    echo "============================================================"
    echo "[$(date)] Running seed=${seed}"
    echo "============================================================"

    SEED=$seed "$SINGLE_SEED_SCRIPT" "$@"

    echo "[$(date)] seed=${seed} done."
done

echo "[$(date)] All seeds finished: $SEEDS"
