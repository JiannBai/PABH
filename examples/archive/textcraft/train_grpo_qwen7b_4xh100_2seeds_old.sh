#!/usr/bin/env bash
# Multi-seed wrapper for TextCraft GRPO 7B on 4xH100.
#
# Runs the single-seed script sequentially with seeds 1 and 2 by default.
# Override with:
#   SEEDS="3 4" bash examples/archive/textcraft/train_grpo_qwen7b_4xh100_2seeds_old.sh
#
# Any arguments are forwarded to run_textcraft_grpo_qwen7b_4xh100.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_SEED_SCRIPT="$SCRIPT_DIR/run_textcraft_grpo_qwen7b_4xh100.sh"
SEEDS=${SEEDS:-"1 2"}

if [[ ! -x "$SINGLE_SEED_SCRIPT" ]]; then
    echo "Missing or non-executable single-seed script: $SINGLE_SEED_SCRIPT" >&2
    exit 1
fi

echo "[$(date)] Starting TextCraft GRPO 7B multi-seed run: $SEEDS"

for seed in $SEEDS; do
    echo "============================================================"
    echo "[$(date)] Running seed=${seed}"
    echo "============================================================"

    SEED=$seed "$SINGLE_SEED_SCRIPT" "$@"

    echo "[$(date)] seed=${seed} done."
done

echo "[$(date)] All seeds finished: $SEEDS"
