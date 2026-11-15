#!/usr/bin/env bash
# 等 ALF OOD baseline s2 收官后，在 GPU 0-3 跑 raw-7B AppWorld val-only 诊断。
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
ALF_CKPT=/root/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode/official_bipaceq_alfworld_ood_1.5b_4gpu_L-12_eps0.05_firstn_seed2_thisnode
EVAL_SCRIPT=$BASE_DIR/examples/bipace/appworld/qwen7b/eval_raw7b_valonly.sh
LOG=$BASE_DIR/verl_log/raw7b_val_watcher.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alf_done() {
  local latest=""
  [ -f "$ALF_CKPT/latest_checkpointed_iteration.txt" ] && latest=$(cat "$ALF_CKPT/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$latest" ] && [ "$latest" -ge 200 ] 2>/dev/null
}

log "raw7b val watcher started; waiting for ALF OOD baseline s2"
while ! alf_done; do sleep 120; done
log "ALF OOD baseline s2 done; waiting GPU 0-3 free"

# 等 ALF OOD relay 收尾 + idle monitor 可能给 0-3 上 stress；等 0-3 显存回落到 <500MB 后
# 停掉 0-3 上的 stress 再上卡。
stable=0
while [ $stable -lt 3 ]; do
  mem=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1<=3 && $2>500 {f=1} END {print (f?"busy":"free")}')
  [ "$mem" = "free" ] && stable=$((stable+1)) || stable=0
  sleep 30
done

# 停掉 0-3 上的 idle-monitor 子进程（让出显存）
for pid in $(ps aux | grep 'gpu_idle_monitor.py --gpu-index' | grep -v grep | awk '{print $2}'); do
  idx=$(ps -o args= -p "$pid" | grep -oE 'gpu-index [0-9]+' | awk '{print $2}')
  [ "$idx" -le 3 ] && kill -TERM "$pid" 2>/dev/null
done
sleep 12

log "GPU 0-3 free, launching raw-7B val-only"
env CUDA_VISIBLE_DEVICES=0,1,2,3 APPWORLD_PORT_FILE=/tmp/codewiz/appworld_ports.ports \
  setsid nohup bash "$EVAL_SCRIPT" >> "$BASE_DIR/verl_log/raw7b_valonly.out" 2>&1 &
log "eval launched pid=$!"
log "watcher exit"
exit 0
