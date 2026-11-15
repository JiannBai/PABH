#!/usr/bin/env bash
# AppWorld 7B 三臂接力 · ai-1845503 · GPU 4-7（SFT multistep 冷启动）
# 队列：baseline(BiPACE-Q) -> bisimhead_prm(我们方法) -> gigpo(GiGPO 精确锚)
# 完成判据：$PROJ/$EXP/latest_checkpointed_iteration.txt >= 100
# 前置：7B multistep SFT 已训练完（脚本内等待 sft_full/global_step_* 出现）

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/appworld/qwen7b/train_appworld_7b_node1797956.sh
SFT_DIR=$BASE_DIR/verl_checkpoints/qwen2.5_7b_appworld_sft_multistep_node1797956/sft_full
PROJ=$BASE_DIR/verl_checkpoints/qwen2.5_7b_appworld_bipace_node1797956
LOG=$BASE_DIR/verl_log/appworld7b_relay_ai1845503.log
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
  for pid in $(ps aux | grep '[g]pu_idle_monitor.py --gpu-index' | awk '{print $2}'); do
    idx=$(ps -o args= -p "$pid" 2>/dev/null | grep -oE 'gpu-index [0-9]+' | awk '{print $2}')
    { [ "$idx" -ge 4 ] && [ "$idx" -le 7 ]; } 2>/dev/null && kill -TERM "$pid" 2>/dev/null
  done
  pkill -TERM -f '[g]pu_stress.py' 2>/dev/null
  sleep 12
}

find_sft_ckpt() {
  local d
  d=$(ls -d "$SFT_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
  [ -n "$d" ] && echo "$d"
}

launch() { # $1=mode $2=seed $3=model
  local MODE=$1 SEED=$2 MODEL=$3
  cd "$BASE_DIR" || return 1
  stop_stress_47
  env MODE=$MODE SEED=$SEED TAG=sftms7b MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=4,5,6,7 \
    APPWORLD_PORT_FILE=/tmp/codewiz/appworld_ports.ports \
    setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "7B MODE=$MODE SEED=$SEED launched pid=$!"
}

run_one() { # $1=mode $2=seed $3=model
  local MODE=$1 SEED=$2 MODEL=$3
  local EXP="${MODE}_appworld_7b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${SEED}_sftms7b_node1797956"
  if is_done "$EXP" "$PROJ"; then log "$EXP already done, skip"; return 0; fi
  if ! is_alive "$EXP"; then
    log "$EXP not running, launching"
    launch "$MODE" "$SEED" "$MODEL"
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
      launch "$MODE" "$SEED" "$MODEL"
      t=$(date +%s)
    fi
    sleep 60
  done
}

log "appworld 7B relay started (ai-1845503), GPU 4-7; waiting for 7B multistep SFT"
while true; do
  SFT_CKPT=$(find_sft_ckpt)
  if [ -n "$SFT_CKPT" ] && ! pgrep -f 'run_sft_coldstart_appworld_multistep.sh' > /dev/null 2>&1; then
    log "7B SFT ckpt ready: $SFT_CKPT"
    break
  fi
  sleep 60
done

for MODE in baseline bisimhead_prm gigpo; do
  run_one "$MODE" 0 "$SFT_CKPT" || exit 1
  sleep 100
done

log "all AppWorld 7B 3-arm DONE, restarting idle monitor"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "relay exit"
exit 0
