#!/usr/bin/env bash
# WebShop 7B BiPACE-Q 固定 eps=0.10 补种（本机 ai-1851164）
# GPU 0-3 顺序跑 seed1 → seed2；seed0 已有，补齐 {0,1,2} 三 seed。
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen7b_4xh100/train_webshop7b_eps010_node1613826.sh
LOG=$BASE_DIR/verl_log/webshop7_bipaceq_eps010_seeds1_2_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_webshop_bipace_node1613826
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct

S1_EXP=baseline_webshop_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed1_node1613826
S2_EXP=baseline_webshop_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed2_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
done7() { [ -d "$CKPT_ROOT/$1/global_step_150" ]; }

mkdir -p "$BASE_DIR/verl_log" /root/ws7_bipaceq_eps010_s1 /root/ws7_bipaceq_eps010_s2
log "WebShop 7B BiPACE-Q eps0.10 seed1/2 relay started"

pkill -TERM -f 'gpu_idle_monitor[.]py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress[.]py' 2>/dev/null || true
sleep 10

launch7() {  # $1=EXP $2=SEED $3=HOME_DIR
  local exp=$1 seed=$2 home_dir=$3
  if done7 "$exp"; then log "$exp already done"; return 0; fi
  if alive "$exp"; then log "$exp already running"; return 0; fi
  log "launch $exp (seed=$seed)"
  cd "$BASE_DIR" || exit 1
  env HOME="$home_dir" CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH="$MODEL" MODE=baseline SEED="$seed" \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$exp launched pid=$!"
}

run_to_done() {  # $1=EXP $2=SEED $3=HOME_DIR
  local exp=$1 seed=$2 home_dir=$3 dead=0
  launch7 "$exp" "$seed" "$home_dir"
  while true; do
    if done7 "$exp"; then log "$exp DONE"; return 0; fi
    if ! alive "$exp"; then
      dead=$((dead + 1))
      [ "$dead" -gt 8 ] && { log "$exp give up after $dead dead checks"; return 1; }
      log "$exp appears dead (check #$dead), relaunching"
      sleep 20
      launch7 "$exp" "$seed" "$home_dir"
    else
      dead=0
    fi
    sleep 180
  done
}

run_to_done "$S1_EXP" 1 /root/ws7_bipaceq_eps010_s1
run_to_done "$S2_EXP" 2 /root/ws7_bipaceq_eps010_s2

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
