#!/usr/bin/env bash
# ALFWorld OOD 完整接力 · ai-1845503 · GPU 0-3
#
# CVPR 泛化证据：方法臂(PABH) vs baseline(官方 BiPACE-Q)，各 3 seeds。
#   方法臂 seed0 已完成；baseline seed0 可能正在跑（由旧 relay 启动）。
# 本 relay 支持三种状态：已 done -> skip；已在跑 -> 等待其完成；否则启动+守护。
#
# 队列（串行，每 run ~28h）：
#   方法 s0 -> baseline s0 -> 方法 s1 -> baseline s1 -> 方法 s2 -> baseline s2
#
# 完成判据：$PROJ/$EXP/latest_checkpointed_iteration.txt >= 200

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
ALF_DIR=$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100
VERL_CKPT_BASE=/root/verl_checkpoints
ALF_PROJ=$VERL_CKPT_BASE/qwen2.5_1.5b_alfworld_bipace_thisnode
LOG=$BASE_DIR/verl_log/alfood_full_relay_ai1845503.log
TOTAL_STEPS=200
MAX_RESTARTS=60

METHOD_SCRIPT=$ALF_DIR/train_bisimhead_g_prmanchor_ood_thisnode.sh
BASE_SCRIPT=$ALF_DIR/train_official_bipaceq_ood_thisnode.sh

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() {
  local latest=""
  [ -f "$2/$1/latest_checkpointed_iteration.txt" ] && latest=$(cat "$2/$1/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$latest" ] && [ "$latest" -ge "$TOTAL_STEPS" ] 2>/dev/null
}

launch() { # $1=script $2=seed
  local SCRIPT=$1 SEED=$2
  cd "$BASE_DIR" || return 1
  env VERL_CKPT_BASE=$VERL_CKPT_BASE SEED=$SEED CUDA_VISIBLE_DEVICES=0,1,2,3 \
    setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$(basename $SCRIPT) SEED=$SEED launched pid=$!"
}

run_one() { # $1=script $2=exp_name $3=seed
  local SCRIPT=$1 EXP=$2 SEED=$3
  if is_done "$EXP" "$ALF_PROJ"; then log "$EXP already done, skip"; return 0; fi

  if ! is_alive "$EXP"; then
    log "$EXP not running, launching"
    launch "$SCRIPT" "$SEED"
  else
    log "$EXP already running, monitoring it"
  fi

  local t=$(date +%s) restarts=0
  while true; do
    local now=$(date +%s)
    if is_done "$EXP" "$ALF_PROJ"; then log "$EXP DONE"; return 0; fi
    if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
      restarts=$((restarts + 1))
      if [ $restarts -gt $MAX_RESTARTS ]; then
        log "$EXP exceeded MAX_RESTARTS ($MAX_RESTARTS), aborting relay"
        return 1
      fi
      log "$EXP not alive, restart #$restarts in 100s"
      sleep 100
      launch "$SCRIPT" "$SEED"
      t=$(date +%s)
    fi
    sleep 60
  done
}

mkdir -p "$VERL_CKPT_BASE"
log "alfood full relay started (ai-1845503), GPU 0-3, ckpt base=$VERL_CKPT_BASE"

for seed in 0 1 2; do
  run_one "$METHOD_SCRIPT" \
    "bisimhead_alfworld_ood_1.5b_4gpu_L-12_eps0.05_g_prmanchor_lr1e-4_tau0.005_ups3_seed${seed}_thisnode" \
    "$seed" || exit 1
  sleep 100
  run_one "$BASE_SCRIPT" \
    "official_bipaceq_alfworld_ood_1.5b_4gpu_L-12_eps0.05_firstn_seed${seed}_thisnode" \
    "$seed" || exit 1
  sleep 100
done

log "all ALF OOD queue (3 seeds x 2 arms) DONE, restarting full idle monitor"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "relay exit"
exit 0
