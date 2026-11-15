#!/usr/bin/env bash
# WebShop 7B PABH eps0.15 s0 接力（本机 ai-1851164）
# 等 7B PABH eps0.10 s0 完赛后，在 GPU 4-7 上启动 eps0.15 s0。
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen7b_4xh100/train_webshop7b_eps015_node1613826.sh
LOG=$BASE_DIR/verl_log/webshop7_pabh_eps015_relay_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_webshop_bipace_node1613826
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct

GATE_EXP=bisimhead_webshop_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed0_node1613826
EXP=bisimhead_webshop_7b_4gpu_L-8_eps0.15_qstyle_actiontag_seed0_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
done7() { [ -d "$CKPT_ROOT/$1/global_step_150" ]; }

mkdir -p "$BASE_DIR/verl_log" /root/ws7_pabh_eps015_s0
log "7B PABH eps0.15 relay started; waiting for eps0.10 gate"

while [ ! -d "$CKPT_ROOT/$GATE_EXP/global_step_150" ]; do
  sleep 180
done
log "gate 7B eps0.10 done, freeing GPU4-7 for eps0.15"

pkill -TERM -f 'gpu_idle_monitor.py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress.py' 2>/dev/null || true
sleep 10

if done7 "$EXP"; then log "$EXP already done"; exit 0; fi
if ! alive "$EXP"; then
  cd "$BASE_DIR" || exit 1
  env HOME=/root/ws7_pabh_eps015_s0 CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_PATH="$MODEL" MODE=bisimhead SEED=0 \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$EXP launched pid=$!"
fi

t=0
while true; do
  if done7 "$EXP"; then log "$EXP DONE"; break; fi
  if ! alive "$EXP"; then
    t=$((t+1))
    [ $t -gt 8 ] && { log "$EXP give up after $t dead checks"; break; }
    log "$EXP appears dead (check #$t), relaunching"
    sleep 120
    cd "$BASE_DIR" || exit 1
    env HOME=/root/ws7_pabh_eps015_s0 CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_PATH="$MODEL" MODE=bisimhead SEED=0 \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    log "$EXP relaunched pid=$!"
  else
    t=0
  fi
  sleep 180
done

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
