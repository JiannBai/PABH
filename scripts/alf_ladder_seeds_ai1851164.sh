#!/usr/bin/env bash
# ALFWorld 1.5B GiGPO ladder seeds s1/s2 · machine ai-1851164
#
# Fills the 3-seed ALFWorld estimator ladder (GiGPO s0 is already done):
#   GPU 0-3 -> GiGPO seed1 (gigpo_alfworld_1.5b_4gpu_seed1_thisnode)
#   GPU 4-7 -> GiGPO seed2 (gigpo_alfworld_1.5b_4gpu_seed2_thisnode)
# Completion criterion = global_step_200 directory.  On death, restart with
# a 180s grace window and resume from the latest checkpoint automatically.
#
# Each run gets its own HOME so the two prepare_offline steps never race on
# the same $HOME/data/verl-agent/text/*.parquet files.

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
ALF_DIR=$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100
LOG=$BASE_DIR/verl_log/alf_ladder_seeds_ai1851164.log
ALFWORLD_DATA=/home/work/.cache/alfworld
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_alfworld_ladder_thisnode
TOTAL_STEPS=200
MAX_RESTARTS=40

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$CKPT_ROOT/$1/global_step_$TOTAL_STEPS" ]; }

run_one() { # $1=seed $2=gpus $3=home
  local SEED=$1 GPUS=$2 HOME_DIR=$3
  local EXP="gigpo_alfworld_1.5b_4gpu_seed${SEED}_thisnode"
  mkdir -p "$HOME_DIR" "$BASE_DIR/verl_log"
  if is_done "$EXP"; then log "$EXP already done, skip"; return 0; fi
  cd "$BASE_DIR" || return 1
  env HOME="$HOME_DIR" ALFWORLD_DATA="$ALFWORLD_DATA" CUDA_VISIBLE_DEVICES="$GPUS" \
      SEED="$SEED" EST=gigpo \
      setsid nohup bash "$ALF_DIR/train_ladder_repro_thisnode.sh" >> /dev/null 2>&1 &
  local t restarts=0
  t=$(date +%s)
  log "$EXP launched (GPU $GPUS, HOME=$HOME_DIR) pid=$!"
  while true; do
    local now
    now=$(date +%s)
    if is_done "$EXP"; then log "$EXP DONE"; return 0; fi
    if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
      restarts=$((restarts + 1))
      if [ $restarts -gt $MAX_RESTARTS ]; then
        log "$EXP exceeded MAX_RESTARTS ($MAX_RESTARTS), giving up"
        return 1
      fi
      log "$EXP not alive, restart #$restarts in 120s"
      sleep 120
      cd "$BASE_DIR" || return 1
      env HOME="$HOME_DIR" ALFWORLD_DATA="$ALFWORLD_DATA" CUDA_VISIBLE_DEVICES="$GPUS" \
          SEED="$SEED" EST=gigpo \
          setsid nohup bash "$ALF_DIR/train_ladder_repro_thisnode.sh" >> /dev/null 2>&1 &
      t=$(date +%s)
      log "$EXP relaunched pid=$!"
    fi
    sleep 60
  done
}

mkdir -p "$BASE_DIR/verl_log"
log "alf_ladder_seeds relay started (ai-1851164), ALFWORLD_DATA=$ALFWORLD_DATA"

run_one 1 "0,1,2,3" /root/alf_ladder_s1 &
PID1=$!
run_one 2 "4,5,6,7" /root/alf_ladder_s2 &
PID2=$!

wait "$PID1"
R1=$?
wait "$PID2"
R2=$?
log "relay finished seed1_rc=$R1 seed2_rc=$R2"
exit $((R1 != 0 || R2 != 0))
