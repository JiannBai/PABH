#!/usr/bin/env bash
# Sequential launcher for the full embedder-ablation grid:
# {hash_ngram, frozen_st} x seeds {0,1,2} on ALFWorld/Qwen2.5-7B.
# Each run is ~the same wall-clock as the reference 7B Q-style run.
#
# Calibrate frozen_st eps first and pass it via FROZEN_ST_EPS:
#   python3 scripts/calibrate_text_embedder_eps.py --embedder frozen_st
#   FROZEN_ST_EPS=0.08 bash examples/bipace/alfworld/qwen7b_4xh100/ablation/embedder_all.sh

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

HASH_EPS=${HASH_EPS:-0.25}
FROZEN_ST_EPS=${FROZEN_ST_EPS:?set FROZEN_ST_EPS after running scripts/calibrate_text_embedder_eps.py --embedder frozen_st}

for seed in 0 1 2; do
  EMBEDDER=hash_ngram EPS=$HASH_EPS SEED=$seed \
    bash "$HERE/run_alfworld_bipace_embedder_ablation_7b.sh"
done

for seed in 0 1 2; do
  EMBEDDER=frozen_st EPS=$FROZEN_ST_EPS SEED=$seed \
    bash "$HERE/run_alfworld_bipace_embedder_ablation_7b.sh"
done
