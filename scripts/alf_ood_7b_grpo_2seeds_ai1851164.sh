#!/usr/bin/env bash
# ALFWorld OOD 7B GRPO 2 seeds（seed0 → seed1，GPU4-7）
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/alfworld/qwen7b_4xh100/train_ood_all_7b_thisnode.sh
LOG=$BASE_DIR/verl_log/alf_ood_7b_grpo_2seeds_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_alfworld_bipace_thisnode
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
done7() { [ -d "$CKPT_ROOT/$1/global_step_150" ]; }

mkdir -p "$BASE_DIR/verl_log"
log "OOD 7B GRPO 2-seed relay started"

pkill -TERM -f 'gpu_idle_monitor[.]py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress[.]py' 2>/dev/null || true
sleep 10

launch() {
  local seed=$1
  local exp=grpo_alfworld_ood_7b_4gpu_seed${seed}_thisnode
  if done7 "$exp"; then log "$exp already done"; return 0; fi
  if alive "$exp"; then log "$exp already running"; return 0; fi
  log "launch $exp"
  cd "$BASE_DIR" || exit 1
  env CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_PATH="$MODEL" MODE=grpo SEED="$seed" \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$exp launched pid=$!"
}

run_seed() {
  local seed=$1
  local exp=grpo_alfworld_ood_7b_4gpu_seed${seed}_thisnode dead=0
  launch "$seed"
  while true; do
    if done7 "$exp"; then log "$exp DONE"; return 0; fi
    if ! alive "$exp"; then
      dead=$((dead + 1))
      [ "$dead" -gt 8 ] && { log "$exp give up"; return 1; }
      log "$exp dead (check #$dead), relaunch"
      sleep 20
      launch "$seed"
    else
      dead=0
    fi
    sleep 180
  done
}

run_seed 0
run_seed 1

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
