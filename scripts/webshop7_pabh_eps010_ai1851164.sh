#!/usr/bin/env bash
# WebShop 7B PABH eps=0.10 s0 · GPU 4-7
BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen7b_4xh100/train_webshop7b_eps010_node1613826.sh
LOG=$BASE_DIR/verl_log/webshop7_pabh_eps010_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_webshop_bipace_node1613826
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct
EXP=bisimhead_webshop_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed0_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$CKPT_ROOT/$1/global_step_150" ]; }
mkdir -p "$BASE_DIR/verl_log" /root/ws7_pabh_eps010_s0
log "relay started"
pkill -TERM -f 'gpu_idle_monitor.py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress.py' 2>/dev/null || true
sleep 10
if is_done "$EXP"; then log "$EXP already done"; exit 0; fi
if ! is_alive "$EXP"; then
  cd "$BASE_DIR" || exit 1
  env HOME=/root/ws7_pabh_eps010_s0 CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_PATH="$MODEL" MODE=bisimhead SEED=0 \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$EXP launched pid=$!"
fi
t=$(date +%s); restarts=0
while true; do
  now=$(date +%s)
  if is_done "$EXP"; then log "$EXP DONE"; break; fi
  if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
    restarts=$((restarts + 1))
    [ $restarts -gt 40 ] && { log "$EXP give up"; break; }
    log "$EXP restart #$restarts"
    sleep 120
    cd "$BASE_DIR" || exit 1
    env HOME=/root/ws7_pabh_eps010_s0 CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_PATH="$MODEL" MODE=bisimhead SEED=0 \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    t=$(date +%s)
  fi
  sleep 60
done
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
