#!/usr/bin/env bash
# Sokoban 1.5B method gtext s2 · machine ai-1851164 · GPU 0-3
# MODE=bisimhead_prm + box_prm + g_text, 200 steps, completion = global_step_200.

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/sokoban/qwen1.5b/train_sokoban_1.5b_node1613826.sh
LOG=$BASE_DIR/verl_log/sokoban_s2_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sokoban_bipace_node1613826
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct
EXP=bisimhead_prm_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed2_gtext_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$CKPT_ROOT/$1/global_step_200" ]; }

mkdir -p "$BASE_DIR/verl_log" /root/sok_s2
log "sokoban_s2 relay started"

pkill -TERM -f 'gpu_idle_monitor.py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress.py' 2>/dev/null || true
sleep 10

if is_done "$EXP"; then log "$EXP already done, skip"; exit 0; fi
if is_alive "$EXP"; then log "$EXP already running, watching"; else
  cd "$BASE_DIR" || exit 1
  env HOME=/root/sok_s2 CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH="$MODEL" MODE=bisimhead_prm SEED=2 TAG=gtext \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$EXP launched pid=$!"
fi

t=$(date +%s)
restarts=0
while true; do
  now=$(date +%s)
  if is_done "$EXP"; then log "$EXP DONE"; break; fi
  if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
    restarts=$((restarts + 1))
    if [ $restarts -gt 40 ]; then log "$EXP exceeded 40 restarts, giving up"; break; fi
    log "$EXP not alive, restart #$restarts in 120s"
    sleep 120
    cd "$BASE_DIR" || exit 1
    env HOME=/root/sok_s2 CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH="$MODEL" MODE=bisimhead_prm SEED=2 TAG=gtext \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    t=$(date +%s)
    log "$EXP relaunched pid=$!"
  fi
  sleep 60
done

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "sokoban_s2 relay exit"
