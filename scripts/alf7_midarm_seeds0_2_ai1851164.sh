#!/usr/bin/env bash
# ALFWorld 7B midarm(g_text only) ×3 补种（本机 ai-1851164）
# 等 ALF 7B PABH seed1 完赛后，在 GPU4-7 顺序跑 seed0 → seed1 → seed2。
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/alfworld/qwen7b_4xh100/train_midarm_g_actorhidden_7b_thisnode.sh
LOG=$BASE_DIR/verl_log/alf7_midarm_seeds0_2_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_alfworld_bipace_thisnode
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct

GATE_EXP=bisimhead_alfworld_7b_4gpu_L-8_eps0.10_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_seed1_thisnode

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
done7() { [ -d "$CKPT_ROOT/$1/global_step_150" ]; }

mkdir -p "$BASE_DIR/verl_log"
log "ALF 7B midarm seed0/1/2 relay started; waiting for ALF 7B PABH seed1 gate"

while [ ! -d "$CKPT_ROOT/$GATE_EXP/global_step_150" ]; do
  sleep 180
done
log "gate ALF 7B PABH seed1 done; launching 7B midarm seed0/1/2 on GPU4-7"

pkill -TERM -f 'gpu_idle_monitor[.]py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress[.]py' 2>/dev/null || true
sleep 10

launch7() {  # $1=SEED
  local seed=$1 exp=midarm_alfworld_7b_4gpu_L-8_eps0.10_g_actorhidden_seed${seed}_thisnode
  if done7 "$exp"; then log "$exp already done"; return 0; fi
  if alive "$exp"; then log "$exp already running"; return 0; fi
  log "launch $exp"
  cd "$BASE_DIR" || exit 1
  env CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_PATH="$MODEL" SEED="$seed" \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$exp launched pid=$!"
}

run_seed() {  # $1=SEED
  local seed=$1 exp=midarm_alfworld_7b_4gpu_L-8_eps0.10_g_actorhidden_seed${seed}_thisnode dead=0
  launch7 "$seed"
  while true; do
    if done7 "$exp"; then log "$exp DONE"; return 0; fi
    if ! alive "$exp"; then
      dead=$((dead + 1))
      [ "$dead" -gt 8 ] && { log "$exp give up after $dead dead checks"; return 1; }
      log "$exp appears dead (check #$dead), relaunching"
      sleep 20
      launch7 "$seed"
    else
      dead=0
    fi
    sleep 180
  done
}

run_seed 0
run_seed 1
run_seed 2

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
