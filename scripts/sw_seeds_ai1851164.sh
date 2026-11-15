#!/usr/bin/env bash
# ScienceWorld continuation · machine ai-1851164
#
# GPU 0-3: finish SW 1.5B method gtext s3 (resume from s150, completion = s200)
# GPU 4-7: start SW 7B method gtext s0 (SFT cold-start, completion = s150)
#
# Each run gets its own HOME so prepare_offline outputs never race.

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SW15_DIR=$BASE_DIR/examples/bipace/sciworld/qwen1.5b
SW7_DIR=$BASE_DIR/examples/bipace/sciworld/qwen7b
LOG=$BASE_DIR/verl_log/sw_seeds_ai1851164.log
SW15_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
SW7_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_sft_node1613826/sft_full/global_step_483

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { # $1=exp $2=ckpt_root $3=total_steps
  local _exp=$1 _root=$2 _total=$3 _d _n
  for _d in "$_root/$_exp"/global_step_*; do
    [ -d "$_d" ] || continue
    _n=${_d##*global_step_}
    case "$_n" in
      ''|*[!0-9]*) continue ;;
    esac
    if [ "$_n" -ge "$_total" ]; then return 0; fi
  done
  return 1
}

run_one() { # $1=script $2=exp $3=ckpt_root $4=total_steps $5=gpus $6=home $7=model $8=mode $9=seed ${10}=start ${11}=tag
  local SCRIPT=$1 EXP=$2 CKPT_ROOT=$3 TOTAL=$4 GPUS=$5 HOME_DIR=$6 MODEL=$7 MODE=$8 SEED=$9 START=${10} TAG=${11}
  mkdir -p "$HOME_DIR" "$BASE_DIR/verl_log"
  if is_done "$EXP" "$CKPT_ROOT" "$TOTAL"; then log "$EXP already done, skip"; return 0; fi
  if is_alive "$EXP"; then
    log "$EXP already running, watching existing process"
    local t=$(date +%s) restarts=0
    while true; do
      local now
      now=$(date +%s)
      if is_done "$EXP" "$CKPT_ROOT" "$TOTAL"; then log "$EXP DONE"; return 0; fi
      if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
        restarts=$((restarts + 1))
        if [ $restarts -gt 40 ]; then log "$EXP exceeded 40 restarts, giving up"; return 1; fi
        log "$EXP not alive, restart #$restarts in 120s"
        sleep 120
        cd "$BASE_DIR" || return 1
        env HOME="$HOME_DIR" CUDA_VISIBLE_DEVICES="$GPUS" MODEL_PATH="$MODEL" MODE="$MODE" SEED="$SEED" START="$START" TAG="$TAG" \
            setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
        t=$(date +%s)
        log "$EXP relaunched pid=$!"
      fi
      sleep 60
    done
  fi
  cd "$BASE_DIR" || return 1
  env HOME="$HOME_DIR" CUDA_VISIBLE_DEVICES="$GPUS" MODEL_PATH="$MODEL" MODE="$MODE" SEED="$SEED" START="$START" TAG="$TAG" \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  local t restarts=0
  t=$(date +%s)
  log "$EXP launched (GPU $GPUS, HOME=$HOME_DIR, model=$MODEL) pid=$!"
  while true; do
    local now
    now=$(date +%s)
    if is_done "$EXP" "$CKPT_ROOT" "$TOTAL"; then log "$EXP DONE"; return 0; fi
    if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
      restarts=$((restarts + 1))
      if [ $restarts -gt 40 ]; then log "$EXP exceeded 40 restarts, giving up"; return 1; fi
      log "$EXP not alive, restart #$restarts in 120s"
      sleep 120
      cd "$BASE_DIR" || return 1
      env HOME="$HOME_DIR" CUDA_VISIBLE_DEVICES="$GPUS" MODEL_PATH="$MODEL" MODE="$MODE" SEED="$SEED" START="$START" TAG="$TAG" \
          setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
      t=$(date +%s)
      log "$EXP relaunched pid=$!"
    fi
    sleep 60
  done
}

mkdir -p "$BASE_DIR/verl_log"
log "sw_seeds relay started (ai-1851164)"

# Avoid the idle-monitor stress racing vLLM's memory profiling at launch.
pkill -TERM -f 'gpu_idle_monitor.py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress.py' 2>/dev/null || true
sleep 10

# 1) SW 1.5B method gtext s3: fresh run from scratch to s200 (head save/load enabled)
run_one "$SW15_DIR/train_sciworld_1.5b_node1613826.sh" \
  "bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed3_gtext_node1613826" \
  "$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826" 200 \
  "0,1,2,3" /root/sw15_s3 "$SW15_CKPT" bisimhead_prm 3 sft gtext &
PID1=$!

# 2) SW 7B method gtext s0: full run to s150
run_one "$SW7_DIR/train_sciworld_7b_node1613826.sh" \
  "bisimhead_prm_sftstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed0_gtext_node1613826" \
  "$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_bipace_node1613826" 150 \
  "4,5,6,7" /root/sw7_s0 "$SW7_CKPT" bisimhead_prm 0 sft gtext &
PID2=$!

wait "$PID1"; R1=$?
wait "$PID2"; R2=$?
log "relay finished 15b_s3_rc=$R1 7b_s0_rc=$R2"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
exit $((R1 != 0 || R2 != 0))
