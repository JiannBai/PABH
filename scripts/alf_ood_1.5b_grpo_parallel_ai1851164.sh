#!/usr/bin/env bash
# ALFWorld OOD 1.5B GRPO ×3（本机 ai-1851164）
# 两槽并行：GPU0-3=seed0，GPU4-7=seed1；任一完赛在空槽接 seed2。
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100/train_ood_all_1.5b_thisnode.sh
LOG=$BASE_DIR/verl_log/alf_ood_1.5b_grpo_parallel_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct

S0_EXP=grpo_alfworld_ood_1.5b_4gpu_seed0_thisnode
S1_EXP=grpo_alfworld_ood_1.5b_4gpu_seed1_thisnode
S2_EXP=grpo_alfworld_ood_1.5b_4gpu_seed2_thisnode

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
done1() { [ -d "$CKPT_ROOT/$1/global_step_200" ]; }

mkdir -p "$BASE_DIR/verl_log"
log "OOD 1.5B GRPO parallel relay started"

pkill -TERM -f 'gpu_idle_monitor[.]py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress[.]py' 2>/dev/null || true
sleep 10

launch() {  # $1=EXP $2=SEED $3=GPUS
  local exp=$1 seed=$2 gpus=$3
  if done1 "$exp"; then log "$exp already done"; return 0; fi
  if alive "$exp"; then log "$exp already running"; return 0; fi
  log "launch $exp (gpus=$gpus)"
  cd "$BASE_DIR" || exit 1
  env CUDA_VISIBLE_DEVICES="$gpus" MODEL_PATH="$MODEL" MODE=grpo SEED="$seed" \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$exp launched pid=$!"
}

launch "$S0_EXP" 0 0,1,2,3
launch "$S1_EXP" 1 4,5,6,7

s2_started=0
dead0=0
dead1=0
while true; do
  if [ "$s2_started" -eq 1 ] && done1 "$S2_EXP"; then
    log "OOD 1.5B GRPO seed0/1/2 all done"
    break
  fi

  if [ "$s2_started" -eq 0 ]; then
    if done1 "$S0_EXP" && ! alive "$S0_EXP"; then
      launch "$S2_EXP" 2 0,1,2,3
      s2_started=1
    elif done1 "$S1_EXP" && ! alive "$S1_EXP"; then
      launch "$S2_EXP" 2 4,5,6,7
      s2_started=1
    fi
  fi

  if ! alive "$S0_EXP" && ! done1 "$S0_EXP"; then
    dead0=$((dead0 + 1))
    if [ "$dead0" -gt 8 ]; then log "$S0_EXP give up"; else
      log "$S0_EXP dead (check #$dead0), relaunch"
      launch "$S0_EXP" 0 0,1,2,3
    fi
  else
    dead0=0
  fi

  if ! alive "$S1_EXP" && ! done1 "$S1_EXP"; then
    dead1=$((dead1 + 1))
    if [ "$dead1" -gt 8 ]; then log "$S1_EXP give up"; else
      log "$S1_EXP dead (check #$dead1), relaunch"
      launch "$S1_EXP" 1 4,5,6,7
    fi
  else
    dead1=0
  fi

  sleep 180
done

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
