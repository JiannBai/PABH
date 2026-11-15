#!/usr/bin/env bash
# ALFWorld 3-seed 补齐接力（修正版）· ai-1845503
#   slotA(GPU0-3): 1.5B BiPACE-Q seed2（已在跑，监控）
#   slotB(GPU4-7): 7B BiPACE-Q(rawnchor) seed2 -> 7B PABH seed2
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
LOG=$BASE_DIR/verl_log/alf_supplement_relay2_ai1845503.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() {
  local v=""; [ -f "$1/$2/latest_checkpointed_iteration.txt" ] && v=$(cat "$1/$2/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$v" ] && [ "$v" -ge "$3" ] 2>/dev/null
}
stop_stress() {
  for pid in $(ps aux | grep '[g]pu_idle_monitor.py --gpu-index' | awk '{print $2}'); do
    idx=$(ps -o args= -p "$pid" 2>/dev/null | grep -oE 'gpu-index [0-9]+' | awk '{print $2}')
    for g in $1; do [ "$idx" = "$g" ] && kill -TERM "$pid" 2>/dev/null; done
  done
  pkill -TERM -f '[g]pu_stress.py' 2>/dev/null
  sleep 12
}

run_one() {
  local SCRIPT=$1 ENVS=$2 PROJ=$3 EXP=$4 STEPS=$5
  if is_done "$PROJ" "$EXP" "$STEPS"; then log "$EXP done"; return 0; fi
  if ! is_alive "$EXP"; then
    log "$EXP launching"
    local gpus=$(echo "$ENVS" | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | cut -d= -f2)
    stop_stress "$(echo $gpus | tr ',' ' ')"
    cd "$BASE_DIR" || return 1
    env $ENVS setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    log "$EXP launched pid=$!"
  else
    log "$EXP running, monitoring"
  fi
  local t=$(date +%s) restarts=0
  while true; do
    is_done "$PROJ" "$EXP" "$STEPS" && { log "$EXP DONE"; return 0; }
    if [ $(( $(date +%s) - t )) -gt 180 ] && ! is_alive "$EXP"; then
      restarts=$((restarts+1)); [ $restarts -gt 40 ] && { log "$EXP gave up"; return 1; }
      log "$EXP restart #$restarts in 100s"; sleep 100
      local gpus=$(echo "$ENVS" | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | cut -d= -f2)
      stop_stress "$(echo $gpus | tr ',' ' ')"
      cd "$BASE_DIR" || return 1
      env $ENVS setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
      t=$(date +%s)
    fi
    sleep 60
  done
}

PROJ7=$BASE_DIR/verl_checkpoints/qwen2.5_7b_alfworld_bipace_thisnode
PROJ15=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode

log "alf supplement relay2 started"

(
  # 1.5B BiPACE-Q seed0 已在跑；跑完后切到我们的方法 PABH(q_style)，再补 baseline seed1/2
  run_one "$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100/train_official_bipaceq_thisnode.sh" \
    "SEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3" "$PROJ15" \
    "official_bipaceq_alfworld_1.5b_4gpu_L-12_eps0.05_qstyle_actiontag_seed0_thisnode" 200
  for s in 0 1 2; do
    run_one "$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100/train_pabh_qstyle_1.5b_thisnode.sh" \
      "SEED=$s CUDA_VISIBLE_DEVICES=0,1,2,3" "$PROJ15" \
      "pabh_alfworld_1.5b_4gpu_L-12_eps0.05_qstyle_actiontag_g_prmanchor_lr1e-4_tau0.005_ups3_seed${s}_thisnode" 200
  done
  for s in 1 2; do
    run_one "$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100/train_official_bipaceq_thisnode.sh" \
      "SEED=$s CUDA_VISIBLE_DEVICES=0,1,2,3" "$PROJ15" \
      "official_bipaceq_alfworld_1.5b_4gpu_L-12_eps0.05_qstyle_actiontag_seed${s}_thisnode" 200
  done
) >> "$LOG" 2>&1 &

(
  # 7B PABH seed2 已在跑；完成后补 seed0（我们的方法优先），再补 7B BiPACE-Q seed0/1
  run_one "$BASE_DIR/examples/bipace/alfworld/qwen7b_4xh100/train_bisimhead_g_prmanchor_thisnode.sh" \
    "SEED=2 CUDA_VISIBLE_DEVICES=4,5,6,7" "$PROJ7" \
    "bisimhead_alfworld_7b_4gpu_L-8_eps0.10_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_seed2_thisnode" 150
  run_one "$BASE_DIR/examples/bipace/alfworld/qwen7b_4xh100/train_bisimhead_g_prmanchor_thisnode.sh" \
    "SEED=0 CUDA_VISIBLE_DEVICES=4,5,6,7" "$PROJ7" \
    "bisimhead_alfworld_7b_4gpu_L-8_eps0.10_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_seed0_thisnode" 150
  # 7B PABH seed0 / 7B BiPACE-Q seed0/1 暂时让位给 ScienceWorld PABH eps 寻优
  # （sw_pabh_eps_search_ai1845503.sh），等寻优短跑完成后再回来补。
) >> "$LOG" 2>&1 &

wait
log "alf supplement relay2 all DONE"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "relay2 exit"
