#!/usr/bin/env bash
# Shared defaults for new BiPACE scripts. Existing scripts are self-contained;
# source this file when adding new runs to keep paths and naming consistent.

set -euo pipefail

BIPACE_EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BIPACE_EXAMPLES_DIR}/../.." && pwd)"

export DATA_ROOT="${DATA_ROOT:-$HOME/data/verl-agent}"
export MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
export LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/train_logs}"
export SWANLAB_ROOT="${SWANLAB_ROOT:-$REPO_ROOT}"
export PROJECT_PREFIX="${PROJECT_PREFIX:-verl_agent_bipace}"

mkdir -p "$LOG_ROOT"

