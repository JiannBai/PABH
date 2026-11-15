#!/usr/bin/env bash
# ScienceWorld PABH eps 寻优 · ai-1845503 · GPU 4-7（bipace 环境，已装 scienceworld）
# 1.5B / 7B 都做：EPS ∈ {0.05, 0.10, 0.15}，各 1 seed × 100 步短跑。
# 之后人工选最优 eps 再补 3 seeds（本脚本只做短跑筛选）。
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
LOG=$BASE_DIR/verl_log/sw_pabh_eps_search_ai1845503.log
SCR15=$BASE_DIR/examples/bipace/sciworld/qwen1.5b/train_sciworld_1.5b_node1613826.sh
SCR7=$BASE_DIR/examples/bipace/sciworld/qwen7b/train_sciworld_7b_node1613826.sh
PROJ15=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
PROJ7=$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_bipace_node1613826
SFT15=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
SFT7=$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_sft_node1613826/sft_full/global_step_483

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() {
  local v=""; [ -f "$1/$2/latest_checkpointed_iteration.txt" ] && v=$(cat "$1/$2/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$v" ] && [ "$v" -ge "$3" ] 2>/dev/null
}
stop_stress() {
  for pid in $(ps aux | grep '[g]pu_idle_monitor.py --gpu-index' | awk '{print $2}'); do
    idx=$(ps -o args= -p "$pid" 2>/dev/null | grep -oE 'gpu-index [0-9]+' | awk '{print $2}')
    { [ "$idx" -ge 4 ] && [ "$idx" -le 7 ]; } 2>/dev/null && kill -TERM "$pid" 2>/dev/null
  done
  pkill -TERM -f '[g]pu_stress.py' 2>/dev/null
  sleep 12
}
run_one() { # $1=script $2=envs $3=proj $4=exp $5=steps
  local SCRIPT=$1 ENVS=$2 PROJ=$3 EXP=$4 STEPS=$5
  if is_done "$PROJ" "$EXP" "$STEPS"; then log "$EXP done"; return 0; fi
  if ! is_alive "$EXP"; then
    log "$EXP launching"; stop_stress
    cd "$BASE_DIR" || return 1
    env $ENVS setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    log "$EXP launched pid=$!"
  else
    log "$EXP running"
  fi
  local t=$(date +%s) restarts=0
  while true; do
    is_done "$PROJ" "$EXP" "$STEPS" && { log "$EXP DONE"; return 0; }
    if [ $(( $(date +%s) - t )) -gt 180 ] && ! is_alive "$EXP"; then
      restarts=$((restarts+1)); [ $restarts -gt 20 ] && { log "$EXP gave up"; return 1; }
      log "$EXP restart #$restarts in 100s"; sleep 100; stop_stress
      cd "$BASE_DIR" || return 1
      env $ENVS setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
      t=$(date +%s)
    fi
    sleep 60
  done
}

log "SW PABH eps search started (GPU 4-7)"
# 等 GPU 4-7 空闲（7B PABH seed2 与 seed0 都跑完后）
while true; do
  d2=$(is_done "$PROJ7" "bisimhead_alfworld_7b_4gpu_L-8_eps0.10_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_seed2_thisnode" 150 && echo y || echo n)
  d0=$(is_done "$PROJ7" "bisimhead_alfworld_7b_4gpu_L-8_eps0.10_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_seed0_thisnode" 150 && echo y || echo n)
  [ "$d2" = "y" ] && [ "$d0" = "y" ] && break
  sleep 120
done
log "GPU 4-7 free (7B PABH seed2/seed0 done)"

for EPS in 0.05 0.10 0.15; do
  run_one "$SCR7" "MODE=bisimhead_prm START=sft EPS=$EPS STEPS=100 SEED=0 MODEL_PATH=$SFT7 CUDA_VISIBLE_DEVICES=4,5,6,7" \
    "$PROJ7" "bisimhead_prm_sftstart_sciworld_7b_4gpu_L-8_eps${EPS}_qstyle_actiontag_seed0_node1613826" 100
done
for EPS in 0.05 0.10 0.15; do
  run_one "$SCR15" "MODE=bisimhead_prm START=sft EPS=$EPS STEPS=100 SEED=0 MODEL_PATH=$SFT15 CUDA_VISIBLE_DEVICES=4,5,6,7" \
    "$PROJ15" "bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps${EPS}_qstyle_actiontag_seed0_node1613826" 100
done

log "SW PABH eps short sweep DONE (6 runs)"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "relay exit"
