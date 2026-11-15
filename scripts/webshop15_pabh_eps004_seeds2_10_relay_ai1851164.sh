#!/usr/bin/env bash
# WebShop 1.5B PABH eps=0.04 补种接力（本机 ai-1851164）
# 等 7B PABH eps0.03 完赛后，在 GPU 0-3 上并行跑 seed2（GPU0,1）与 seed10（GPU2,3）。
# seed0 已在 eps 寻优阶段完成（峰值 0.777）。
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen1.5b_2gpu_official/train_webshop15b_pabh_eps_node1613826.sh
LOG=$BASE_DIR/verl_log/webshop15_pabh_eps004_seeds2_10_relay_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct

GATE_EXP=bisimhead_webshop_7b_4gpu_L-8_eps0.03_qstyle_actiontag_seed0_node1613826
GATE_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_webshop_bipace_node1613826/$GATE_EXP

S2_EXP=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.04_qstyle_g_prmanchor_seed2_node1613826
S10_EXP=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.04_qstyle_g_prmanchor_seed10_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
done1() { [ -d "$CKPT_ROOT/$1/global_step_200" ]; }

mkdir -p "$BASE_DIR/verl_log" /root/ws15_pabh_eps004_s2 /root/ws15_pabh_eps004_s10
log "1.5B PABH eps0.04 seed2/10 relay started; waiting for 7B eps0.03 gate"

while [ ! -d "$GATE_CKPT/global_step_150" ]; do
  sleep 180
done
log "gate 7B eps0.03 done; launching 1.5B eps0.04 seed2/seed10 on GPU0-3"

pkill -TERM -f 'gpu_idle_monitor.py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress.py' 2>/dev/null || true
sleep 10

launch() {  # $1=EXP $2=SEED $3=HOME_DIR $4=GPU_PAIR
  local exp=$1 seed=$2 home_dir=$3 gpus=$4
  if done1 "$exp"; then log "$exp already done"; return 0; fi
  if alive "$exp"; then log "$exp already running"; return 0; fi
  log "launch $exp (seed=$seed, gpus=$gpus)"
  cd "$BASE_DIR" || exit 1
  env HOME="$home_dir" CUDA_VISIBLE_DEVICES="$gpus" MODEL_PATH="$MODEL" EPS=0.04 SEED="$seed" \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$exp launched pid=$!"
}

launch "$S2_EXP" 2 /root/ws15_pabh_eps004_s2 0,1
launch "$S10_EXP" 10 /root/ws15_pabh_eps004_s10 2,3

dead2=0
dead10=0
while true; do
  if done1 "$S2_EXP" && done1 "$S10_EXP"; then
    log "1.5B PABH eps0.04 seed2/seed10 both done"
    break
  fi
  if ! alive "$S2_EXP" && ! done1 "$S2_EXP"; then
    dead2=$((dead2 + 1))
    if [ "$dead2" -gt 8 ]; then
      log "$S2_EXP dead $dead2 checks, give up"
    else
      log "$S2_EXP dead (check #$dead2), relaunch"
      launch "$S2_EXP" 2 /root/ws15_pabh_eps004_s2 0,1
    fi
  else
    dead2=0
  fi
  if ! alive "$S10_EXP" && ! done1 "$S10_EXP"; then
    dead10=$((dead10 + 1))
    if [ "$dead10" -gt 8 ]; then
      log "$S10_EXP dead $dead10 checks, give up"
    else
      log "$S10_EXP dead (check #$dead10), relaunch"
      launch "$S10_EXP" 10 /root/ws15_pabh_eps004_s10 2,3
    fi
  else
    dead10=0
  fi
  sleep 180
done

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
