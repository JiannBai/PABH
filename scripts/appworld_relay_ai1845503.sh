#!/usr/bin/env bash
# AppWorld 三臂接力 · ai-1845503 · GPU 4-7
# 验证当前 PABH 相对 GiGPO / BiPACE-Q 在 AppWorld 上是否有效（各 100 步试水）。
# 队列（串行，GPU 4-7）：
#   1. baseline       (BiPACE-Q：actor_hidden 聚类)
#   2. gigpo          (GiGPO：精确锚分组，bisim_grouping=False)
#   3. bisimhead_prm  (我们的方法：g_text + head + api_prm 锚)
# 完成判据：$PROJ/$EXP/latest_checkpointed_iteration.txt >= 100

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/appworld/qwen1.5b/train_appworld_1.5b_node1797956.sh
PROJ=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_appworld_bipace_node1797956
SFT_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_appworld_sft_multistep_node1797956/sft_full/global_step_78
LOG=$BASE_DIR/verl_log/appworld_relay_ai1845503.log
TOTAL_STEPS=100
MAX_RESTARTS=30

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() {
  local latest=""
  [ -f "$2/$1/latest_checkpointed_iteration.txt" ] && latest=$(cat "$2/$1/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$latest" ] && [ "$latest" -ge "$TOTAL_STEPS" ] 2>/dev/null
}

stop_stress_47() {
  for pid in $(ps aux | grep 'gpu_idle_monitor.py --gpu-index' | grep -v grep | awk '{print $2}'); do
    idx=$(ps -o args= -p "$pid" 2>/dev/null | grep -oE 'gpu-index [0-9]+' | awk '{print $2}')
    { [ "$idx" -ge 4 ] && [ "$idx" -le 7 ]; } 2>/dev/null && kill -TERM "$pid" 2>/dev/null
  done
  sleep 12
}

launch() { # $1=mode $2=seed
  local MODE=$1 SEED=$2
  cd "$BASE_DIR" || return 1
  stop_stress_47
  env MODE=$MODE SEED=$SEED TAG=sftms MODEL_PATH=$SFT_CKPT CUDA_VISIBLE_DEVICES=4,5,6,7 \
    APPWORLD_PORT_FILE=/tmp/codewiz/appworld_ports.ports \
    setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "MODE=$MODE SEED=$SEED launched pid=$!"
}

run_one() { # $1=mode $2=seed
  local MODE=$1 SEED=$2
  local EXP="${MODE}_appworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${SEED}_sftms_node1797956"
  if is_done "$EXP" "$PROJ"; then log "$EXP already done, skip"; return 0; fi
  if ! is_alive "$EXP"; then
    log "$EXP not running, launching"
    launch "$MODE" "$SEED"
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
      launch "$MODE" "$SEED"
      t=$(date +%s)
    fi
    sleep 60
  done
}

log "appworld relay started (ai-1845503), GPU 4-7"
for MODE in baseline gigpo bisimhead_prm; do
  run_one "$MODE" 0 || exit 1
  sleep 100
done
log "all AppWorld 3-arm DONE, restarting idle monitor"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "relay exit"
exit 0
