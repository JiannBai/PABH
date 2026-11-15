#!/usr/bin/env bash
# 7B 补充实验接力 · ai-1845503 · GPU 0-3（ALF OOD 收官后接管）
# 顺序：raw-7B AppWorld val-only 诊断 -> ALFWorld 7B GiGPO ×3 seeds -> WebShop 7B GiGPO ×3 seeds
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
ALF_CKPT=/root/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode/official_bipaceq_alfworld_ood_1.5b_4gpu_L-12_eps0.05_firstn_seed2_thisnode
RAW7B_EVAL=$BASE_DIR/examples/bipace/appworld/qwen7b/eval_raw7b_valonly.sh
ALF_LADDER=$BASE_DIR/examples/bipace/alfworld/qwen7b_4xh100/train_ladder_repro_7b_thisnode.sh
WS_SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen7b_4xh100/train_webshop7b_node1613826.sh
ALF_PROJ=$BASE_DIR/verl_checkpoints/qwen2.5_7b_alfworld_ladder_thisnode
WS_PROJ=$BASE_DIR/verl_checkpoints/qwen2.5_7b_webshop_bipace_node1613826
LOG=$BASE_DIR/verl_log/sevenb_supplement_relay.log
TOTAL_STEPS=150

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alf_done() {
  local latest=""
  [ -f "$ALF_CKPT/latest_checkpointed_iteration.txt" ] && latest=$(cat "$ALF_CKPT/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$latest" ] && [ "$latest" -ge 200 ] 2>/dev/null
}
is_done() { # $1=proj $2=exp
  local latest=""
  [ -f "$1/$2/latest_checkpointed_iteration.txt" ] && latest=$(cat "$1/$2/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$latest" ] && [ "$latest" -ge "$TOTAL_STEPS" ] 2>/dev/null
}
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
gpu_free_03() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$1<=3 && $2>500 {f=1} END {print (f?"busy":"free")}'
}

stop_stress_03() {
  for pid in $(ps aux | grep 'gpu_idle_monitor.py --gpu-index' | grep -v grep | awk '{print $2}'); do
    idx=$(ps -o args= -p "$pid" 2>/dev/null | grep -oE 'gpu-index [0-9]+' | awk '{print $2}')
    [ "$idx" -le 3 ] && kill -TERM "$pid" 2>/dev/null
  done
  sleep 12
}

launch_alf_gigpo() { # $1=seed
  local SEED=$1
  cd "$BASE_DIR" || return 1
  env EST=gigpo SEED=$SEED CUDA_VISIBLE_DEVICES=0,1,2,3 \
    setsid nohup bash "$ALF_LADDER" >> /dev/null 2>&1 &
  log "ALF 7B gigpo SEED=$SEED launched pid=$!"
}

launch_ws_gigpo() { # $1=seed
  local SEED=$1
  cd "$BASE_DIR" || return 1
  env MODE=gigpo SEED=$SEED CUDA_VISIBLE_DEVICES=0,1,2,3 \
    setsid nohup bash "$WS_SCRIPT" >> /dev/null 2>&1 &
  log "WebShop 7B gigpo SEED=$SEED launched pid=$!"
}

run_one() { # $1=kind(alf|ws) $2=seed $3=exp
  local KIND=$1 SEED=$2 EXP=$3 PROJ
  [ "$KIND" = "alf" ] && PROJ=$ALF_PROJ || PROJ=$WS_PROJ
  if is_done "$PROJ" "$EXP"; then log "$EXP already done, skip"; return 0; fi
  if ! is_alive "$EXP"; then
    [ "$KIND" = "alf" ] && launch_alf_gigpo "$SEED" || launch_ws_gigpo "$SEED"
  else
    log "$EXP already running, monitoring"
  fi
  local t=$(date +%s)
  while true; do
    is_done "$PROJ" "$EXP" && { log "$EXP DONE"; return 0; }
    if [ $(( $(date +%s) - t )) -gt 180 ] && ! is_alive "$EXP"; then
      log "$EXP not alive, restarting in 100s"
      sleep 100
      [ "$KIND" = "alf" ] && launch_alf_gigpo "$SEED" || launch_ws_gigpo "$SEED"
      t=$(date +%s)
    fi
    sleep 60
  done
}

log "7B supplement relay started; waiting for ALF OOD baseline s2"
while ! alf_done; do sleep 120; done
log "ALF OOD done; waiting GPU 0-3 free"
while [ "$(gpu_free_03)" != "free" ]; do sleep 60; done
stop_stress_03

# 1. raw-7B AppWorld val-only（一次性诊断，退出后继续）
log "launching raw-7B val-only"
env CUDA_VISIBLE_DEVICES=0,1,2,3 APPWORLD_PORT_FILE=/tmp/codewiz/appworld_ports.ports \
  setsid nohup bash "$RAW7B_EVAL" >> "$BASE_DIR/verl_log/raw7b_valonly.out" 2>&1 &
while pgrep -f 'eval_raw7b_valonly.sh' > /dev/null 2>&1; do sleep 60; done
log "raw-7B val-only finished"

# 2. ALFWorld 7B GiGPO ×3 seeds
for s in 0 1 2; do
  run_one alf "$s" "gigpo_alfworld_7b_4gpu_seed${s}_thisnode" || exit 1
  sleep 100
done

# 3. WebShop 7B GiGPO ×3 seeds
for s in 0 1 2; do
  run_one ws "$s" "gigpo_webshop_7b_4gpu_L-8_eps0.08_qstyle_actiontag_seed${s}_node1613826" || exit 1
  sleep 100
done

log "all 7B supplement DONE, restarting idle monitor"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "relay exit"
exit 0
