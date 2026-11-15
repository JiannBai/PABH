#!/usr/bin/env bash
# ALFWorld 1.5B midarm(g_text only) seed2 补种（本机 ai-1851164）
# 等 WebShop 7B BiPACE-Q eps0.10 seed2 完赛后，在 GPU0-3 启动。
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100/train_midarm_g_actorhidden_thisnode.sh
LOG=$BASE_DIR/verl_log/alf15_midarm_seed2_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct

GATE_EXP=baseline_webshop_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed2_node1613826
GATE_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_webshop_bipace_node1613826/$GATE_EXP
EXP=midarm_alfworld_1.5b_4gpu_L-12_eps0.05_g_actorhidden_gtext_seed2_thisnode

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
done1() { [ -d "$CKPT_ROOT/$1/global_step_200" ]; }

mkdir -p "$BASE_DIR/verl_log"
log "ALF 1.5B midarm seed2 relay started; waiting for WebShop BiPACE-Q seed2 gate"

while [ ! -d "$GATE_CKPT/global_step_150" ]; do
  sleep 180
done
log "gate WebShop BiPACE-Q seed2 done; launching 1.5B midarm seed2 on GPU0-3"

pkill -TERM -f 'gpu_idle_monitor[.]py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress[.]py' 2>/dev/null || true
sleep 10

launch() {
  if done1 "$EXP"; then log "$EXP already done"; return 0; fi
  if alive "$EXP"; then log "$EXP already running"; return 0; fi
  log "launch $EXP"
  cd "$BASE_DIR" || exit 1
  env CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH="$MODEL" SEED=2 GTEXT=1 \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$EXP launched pid=$!"
}

launch

dead=0
while true; do
  if done1 "$EXP"; then log "$EXP DONE"; break; fi
  if ! alive "$EXP"; then
    dead=$((dead + 1))
    [ "$dead" -gt 8 ] && { log "$EXP give up after $dead dead checks"; break; }
    log "$EXP appears dead (check #$dead), relaunching"
    sleep 20
    launch
  else
    dead=0
  fi
  sleep 180
done

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
