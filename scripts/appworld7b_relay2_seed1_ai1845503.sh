#!/usr/bin/env bash
# AppWorld 7B seed1 三臂接力 · ai-1845503 · GPU 0-3（与 seed0 的 GPU 4-7 并行）
# 队列：baseline -> bisimhead_prm(我们方法) -> gigpo
# 端口表：/tmp/codewiz/appworld_ports2.ports（第二套 server 集群）
# 完成判据：latest_checkpointed_iteration.txt >= 100

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/appworld/qwen7b/train_appworld_7b_node1797956.sh
SFT_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_appworld_sft_multistep_node1797956/sft_full/global_step_78
PROJ=$BASE_DIR/verl_checkpoints/qwen2.5_7b_appworld_bipace_node1797956
LOG=$BASE_DIR/verl_log/appworld7b_relay2_seed1.log
TOTAL_STEPS=100
MAX_RESTARTS=30
PORT_FILE=/tmp/codewiz/appworld_ports2.ports

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() {
  local latest=""
  [ -f "$2/$1/latest_checkpointed_iteration.txt" ] && latest=$(cat "$2/$1/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$latest" ] && [ "$latest" -ge "$TOTAL_STEPS" ] 2>/dev/null
}

stop_stress_03() {
  for pid in $(ps aux | grep '[g]pu_idle_monitor.py --gpu-index' | awk '{print $2}'); do
    idx=$(ps -o args= -p "$pid" 2>/dev/null | grep -oE 'gpu-index [0-9]+' | awk '{print $2}')
    { [ "$idx" -ge 0 ] && [ "$idx" -le 3 ]; } 2>/dev/null && kill -TERM "$pid" 2>/dev/null
  done
  pkill -TERM -f '[g]pu_stress.py' 2>/dev/null
  sleep 12
}

launch() { # $1=mode $2=seed
  local MODE=$1 SEED=$2
  cd "$BASE_DIR" || return 1
  stop_stress_03
  env MODE=$MODE SEED=$SEED TAG=sftms7b MODEL_PATH=$SFT_CKPT CUDA_VISIBLE_DEVICES=0,1,2,3 \
    APPWORLD_PORT_FILE=$PORT_FILE \
    setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "7B-seed1 MODE=$MODE SEED=$SEED launched pid=$!"
}

run_one() { # $1=mode
  local MODE=$1
  local EXP="${MODE}_appworld_7b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_sftms7b_node1797956"
  if is_done "$EXP" "$PROJ"; then log "$EXP already done, skip"; return 0; fi
  if ! is_alive "$EXP"; then
    log "$EXP not running, launching"
    launch "$MODE" 1
  else
    log "$EXP already running, monitoring it"
  fi
  local t=$(date +%s) restarts=0
  while true; do
    local now=$(date +%s)
    is_done "$EXP" "$PROJ" && { log "$EXP DONE"; return 0; }
    if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
      restarts=$((restarts + 1))
      [ $restarts -gt $MAX_RESTARTS ] && { log "$EXP exceeded MAX_RESTARTS, aborting"; return 1; }
      log "$EXP not alive, restart #$restarts in 100s"
      sleep 100
      launch "$MODE" 1
      t=$(date +%s)
    fi
    sleep 60
  done
}

log "appworld 7B seed1 relay started (GPU 0-3, port file $PORT_FILE); waiting for second server cluster"
while [ "$(wc -l < "$PORT_FILE" 2>/dev/null || echo 0)" -lt 160 ]; do sleep 30; done
log "second server cluster ready"

for MODE in baseline bisimhead_prm gigpo; do
  run_one "$MODE" || exit 1
  sleep 100
done

log "all AppWorld 7B seed1 3-arm DONE"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "relay exit"
exit 0
