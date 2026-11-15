#!/usr/bin/env bash
# ALFWorld OOD relay · ai-1845503 (new container) · GPU 0-3
#
# Top-priority CVPR generalization evidence.  Runs, serially on GPU 0-3:
#   1. ALFWorld OOD 方法臂 s0  (bisimhead_g_prmanchor, ~28h)
#   2. ALFWorld OOD baseline s0 (official BiPACE-Q, ~28h)
# Each run: restart on death (grace 180s), completion = global_step_200 dir.
# At the very end the full 8-GPU idle monitor supervisor is restarted so the
# idle cards (0-3) are protected from reclamation again.
#
# Env note: both training scripts self-activate the shared `bipace` env via
# the conda shim at /root/miniconda3/etc/profile.d/conda.sh.

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
ALF_DIR=$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100
# JuiceFS shared volume is 100% full; keep this run's checkpoints on local disk.
# Sync-back to the durable repo dir is done by hand after the run / once space is freed.
VERL_CKPT_BASE=/root/verl_checkpoints
ALF_PROJ=$VERL_CKPT_BASE/qwen2.5_1.5b_alfworld_bipace_thisnode
LOG=$BASE_DIR/verl_log/alfood_relay_ai1845503.log
TOTAL_STEPS=200
MAX_RESTARTS=60

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
# Args: $1=exp_name, $2=proj_root.  The checkpoint lives under
# $proj_root/$exp_name/ (trainer.default_local_dir), so the marker is read from
# there.  Completion is detected via latest_checkpointed_iteration.txt
# (>= TOTAL_STEPS), NOT via the presence of global_step_<TOTAL_STEPS>, because
# keep_best_and_latest prunes that exact folder once the run moves past it.
is_done() {
  local latest=""
  [ -f "$2/$1/latest_checkpointed_iteration.txt" ] && latest=$(cat "$2/$1/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$latest" ] && [ "$latest" -ge "$TOTAL_STEPS" ] 2>/dev/null
}

run_one() { # $1=script $2=exp_name $3=proj_root $4=seed
  local SCRIPT=$1 EXP=$2 PROJ=$3 SEED=$4
  if is_done "$EXP" "$PROJ"; then log "$EXP already done, skip"; return 0; fi
  cd "$BASE_DIR" || return 1
  env VERL_CKPT_BASE=$VERL_CKPT_BASE SEED=$SEED CUDA_VISIBLE_DEVICES=0,1,2,3 \
    setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  local t=$(date +%s) restarts=0
  log "$EXP launched pid=$!"
  while true; do
    local now=$(date +%s)
    if is_done "$EXP" "$PROJ"; then log "$EXP DONE"; return 0; fi
    if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
      restarts=$((restarts + 1))
      if [ $restarts -gt $MAX_RESTARTS ]; then
        log "$EXP exceeded MAX_RESTARTS ($MAX_RESTARTS), aborting relay"
        return 1
      fi
      log "$EXP not alive, restart #$restarts in 100s"
      sleep 100
      cd "$BASE_DIR" || return 1
      env VERL_CKPT_BASE=$VERL_CKPT_BASE SEED=$SEED CUDA_VISIBLE_DEVICES=0,1,2,3 \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
      t=$(date +%s); log "$EXP relaunched pid=$!"
    fi
    sleep 60
  done
}

mkdir -p "$VERL_CKPT_BASE"
log "alfood relay started (ai-1845503), GPU 0-3, ckpt base=$VERL_CKPT_BASE"

# 1. ALFWorld OOD 方法臂 s0
run_one "$ALF_DIR/train_bisimhead_g_prmanchor_ood_thisnode.sh" \
  "bisimhead_alfworld_ood_1.5b_4gpu_L-12_eps0.05_g_prmanchor_lr1e-4_tau0.005_ups3_seed0_thisnode" \
  "$ALF_PROJ" 0 || exit 1
sleep 100

# 2. ALFWorld OOD baseline s0（严格单变量对照）
run_one "$ALF_DIR/train_official_bipaceq_ood_thisnode.sh" \
  "official_bipaceq_alfworld_ood_1.5b_4gpu_L-12_eps0.05_firstn_seed0_thisnode" \
  "$ALF_PROJ" 0 || exit 1

log "all ALF OOD queue DONE, restarting full idle monitor"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "relay exit"
exit 0
