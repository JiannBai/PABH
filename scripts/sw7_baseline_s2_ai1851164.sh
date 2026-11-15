#!/usr/bin/env bash
# SW 7B BiPACE-Q baseline s2 · machine ai-1851164 · GPU 0-3
# 等 baseline s1 完赛（global_step_150）+ GPU 0-3 空闲后自动接力。
# MODE=baseline (actor_hidden, no g_text, no head), SFT cold-start, 150 steps.

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/sciworld/qwen7b/train_sciworld_7b_node1613826.sh
LOG=$BASE_DIR/verl_log/sw7_baseline_s2_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_bipace_node1613826
SFT=$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_sft_node1613826/sft_full/global_step_483
S1_EXP=baseline_sftstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed1_node1613826
EXP=baseline_sftstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed2_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$CKPT_ROOT/$1/global_step_150" ]; }
gpu_free_03() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$1<=3 && $2>500 {f=1} END {print (f?"busy":"free")}'
}

mkdir -p "$BASE_DIR/verl_log" /root/sw7_bl_s2
log "sw7_baseline_s2 relay started; waiting for baseline s1 (global_step_150)"

# 等 s1 完赛，再等 s1 relay 收尾退出（它会重启 idle-monitor），避免抢卡竞争。
while ! is_done "$S1_EXP"; do sleep 120; done
log "baseline s1 done; waiting for s1 relay to exit"
while pgrep -f 'sw7_baseline_s1_ai1851164.sh' > /dev/null 2>&1; do sleep 60; done

# 停掉 idle-monitor（s1 relay 退出前会重启它，此时 GPU 0-3 可能被 stress 占住）。
log "stopping idle monitor before launch"
pkill -TERM -f 'gpu_idle_monitor.py' 2>/dev/null || true
pkill -TERM -f 'gpu_stress.py' 2>/dev/null || true
sleep 10

log "waiting for GPU 0-3 free"
stable=0
while [ "$stable" -lt 3 ]; do
  if [ "$(gpu_free_03)" = "free" ]; then
    stable=$((stable+1))
  else
    stable=0
    pkill -TERM -f 'gpu_idle_monitor.py' 2>/dev/null || true
    pkill -TERM -f 'gpu_stress.py' 2>/dev/null || true
  fi
  sleep 30
done

if is_done "$EXP"; then log "$EXP already done, skip"; exit 0; fi
if is_alive "$EXP"; then log "$EXP already running, watching"; else
  cd "$BASE_DIR" || exit 1
  env HOME=/root/sw7_bl_s2 CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH="$SFT" MODE=baseline SEED=2 START=sft \
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
    if [ "$restarts" -gt 40 ]; then log "$EXP exceeded 40 restarts, giving up"; break; fi
    log "$EXP not alive, restart #$restarts in 120s"
    sleep 120
    cd "$BASE_DIR" || exit 1
    env HOME=/root/sw7_bl_s2 CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH="$SFT" MODE=baseline SEED=2 START=sft \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    t=$(date +%s)
    log "$EXP relaunched pid=$!"
  fi
  sleep 60
done

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "sw7_baseline_s2 relay exit"
