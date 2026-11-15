#!/usr/bin/env bash
# SW 7B method gtext s1 · machine ai-1851164 · GPU 0-3
# SFT cold-start, 150 steps, completion = global_step_150.

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/sciworld/qwen7b/train_sciworld_7b_node1613826.sh
LOG=$BASE_DIR/verl_log/sw7_s1_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_bipace_node1613826
SFT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_sft_node1613826/sft_full/global_step_483
EXP=bisimhead_prm_sftstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed1_gtext_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$CKPT_ROOT/$1/global_step_150" ]; }

mkdir -p "$BASE_DIR/verl_log" /root/sw7_s1
log "sw7_s1 relay started"
if is_done "$EXP"; then log "$EXP already done, skip"; exit 0; fi
if is_alive "$EXP"; then log "$EXP already running, watching"; else
  cd "$BASE_DIR" || exit 1
  env HOME=/root/sw7_s1 CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH="$SFT" MODE=bisimhead_prm SEED=1 START=sft TAG=gtext \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$EXP launched pid=$!"
fi

t=$(date +%s)
restarts=0
while true; do
  now=$(date +%s)
  if is_done "$EXP"; then log "$EXP DONE"; exit 0; fi
  if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
    restarts=$((restarts + 1))
    if [ $restarts -gt 40 ]; then log "$EXP exceeded 40 restarts, giving up"; exit 1; fi
    log "$EXP not alive, restart #$restarts in 120s"
    sleep 120
    cd "$BASE_DIR" || exit 1
    env HOME=/root/sw7_s1 CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH="$SFT" MODE=bisimhead_prm SEED=1 START=sft TAG=gtext \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    t=$(date +%s)
    log "$EXP relaunched pid=$!"
  fi
  sleep 60
done
