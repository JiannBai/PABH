#!/usr/bin/env bash
# exp4_launcher.sh — 自动启动器（用户已授权 2026-09-04）
# 等待 实验2 (pid 828953, GPU 4-7) 或 实验3 (pid 877461, GPU 0-3) 任一退出，
# 确认对应 GPU 显存释放后，在该组 GPU 上启动实验4
# （d_g锚 + resample + min_dt=3）。
#
# 日志: verl_log/exp4_launcher.log

BASE=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
LOG=$BASE/verl_log/exp4_launcher.log
SCRIPT=$BASE/examples/bipace/textcraft/qwen1.5b_4xh100/train_bisimhead_dg_anchor_resample_mindt3_thisnode.sh

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "watcher started (pid $$): waiting for exp2(pid 828953,GPU4-7) or exp3(pid 877461,GPU0-3) to exit"

while true; do
  GPUS=""
  if ! kill -0 828953 2>/dev/null; then
    GPUS=4,5,6,7
    log "exp2 (pid 828953) exited -> GPU 4-7 candidate"
  elif ! kill -0 877461 2>/dev/null; then
    GPUS=0,1,2,3
    log "exp3 (pid 877461) exited -> GPU 0-3 candidate"
  else
    sleep 300
    continue
  fi

  # 等待 GPU 显存完全释放（vllm/FSDP 清理可能需要几分钟；上限 60 分钟）
  FREED=0
  for i in $(seq 1 60); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPUS | awk '{s+=$1} END {print s}')
    if [ "${USED:-99999}" -lt 2000 ]; then FREED=1; break; fi
    sleep 60
  done
  if [ "$FREED" != "1" ]; then
    log "WARNING: GPU $GPUS memory not freed after 60min (used=${USED}MiB), launching anyway"
  fi
  sleep 60  # 额外缓冲，等 Ray/vllm 完全退出

  log "launching exp4 on CUDA_VISIBLE_DEVICES=$GPUS"
  cd "$BASE" || { log "FATAL: cd $BASE failed"; exit 1; }
  CUDA_VISIBLE_DEVICES=$GPUS nohup bash "$SCRIPT" >> "$LOG" 2>&1 &
  LAUNCHER_PID=$!
  disown $LAUNCHER_PID 2>/dev/null
  log "exp4 launched (bash pid $LAUNCHER_PID, GPUs $GPUS); watcher exiting"
  break
done
