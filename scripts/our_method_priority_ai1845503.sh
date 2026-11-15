#!/usr/bin/env bash
# ScienceWorld 优先接力 · ai-1845503（g_text 已修正为 goal-conditioned + 全参数事件）
# slotA(GPU0-3): 收尾 1.5B PABH eps0.15(旧 g_text) -> SW GRPO 1.5B x3 -> SW PABH eps(gtextv2)
#                -> ALF BiPACE-Q 1.5B first_n x3
# slotB(GPU4-7): SW GiGPO 7B x3 -> SW GRPO 7B x3 -> SW PABH eps(gtextv2) -> ALF 7B PABH 收尾
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
LOG=$BASE_DIR/verl_log/our_method_priority_ai1845503.log

SCR_ALF15=$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100/train_pabh_qstyle_1.5b_thisnode.sh
SCR_ALF7=$BASE_DIR/examples/bipace/alfworld/qwen7b_4xh100/train_bisimhead_g_prmanchor_thisnode.sh
SCR_BIPACEQ15=$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100/train_official_bipaceq_thisnode.sh
SCR_SW15=$BASE_DIR/examples/bipace/sciworld/qwen1.5b/train_sciworld_1.5b_node1613826.sh
SCR_SW7=$BASE_DIR/examples/bipace/sciworld/qwen7b/train_sciworld_7b_node1613826.sh
SCR_OOD15=$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100/train_ood_all_1.5b_thisnode.sh
SCR_OOD7=$BASE_DIR/examples/bipace/alfworld/qwen7b_4xh100/train_ood_all_7b_thisnode.sh

PROJ_ALF15=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode
PROJ_ALF7=$BASE_DIR/verl_checkpoints/qwen2.5_7b_alfworld_bipace_thisnode
PROJ_SW15=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
PROJ_SW7=$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_bipace_node1613826
SFT15=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
SFT7=$BASE_DIR/verl_checkpoints/qwen2.5_7b_sciworld_sft_node1613826/sft_full/global_step_483

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() {
  local v=""; [ -f "$1/$2/latest_checkpointed_iteration.txt" ] && v=$(cat "$1/$2/latest_checkpointed_iteration.txt" 2>/dev/null)
  [ -n "$v" ] && [ "$v" -ge "$3" ] 2>/dev/null
}
IDLE_LOCK=$BASE_DIR/verl_log/.idle_monitor_launch_lock

# 启动训练前临时停掉 idle monitor（否则 vLLM 内存 profiling 期间 stress 释放
# 会触发 "Initial free memory ... current free memory" 断言）。训练上卡后立刻恢复。
pause_idle_monitor() { # $1 = 目标 GPU 列表（逗号分隔，可选）
  local gpus=$1
  while ! mkdir "$IDLE_LOCK" 2>/dev/null; do sleep 5; done
  pkill -TERM -f '[g]pu_idle_monitor.py' 2>/dev/null
  pkill -TERM -f '[g]pu_stress.py' 2>/dev/null
  sleep 15
  if [ -n "$gpus" ]; then
    for _ in $(seq 1 30); do
      local m
      m=$(gpu_mem "$gpus")
      [ -n "$m" ] && [ "$m" -le 500 ] 2>/dev/null && break
      sleep 10
    done
  fi
}

resume_idle_monitor() {
  bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
  rmdir "$IDLE_LOCK" 2>/dev/null
}

gpu_mem() { # $1=comma list of gpus -> max memory used among them (MiB)
  local mx=0 m
  for m in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v gpus="$1" 'BEGIN{split(gpus,a,",")} {for(i in a) if($1==a[i]) print $2}'); do
    [ "$m" -gt "$mx" ] 2>/dev/null && mx=$m
  done
  echo "$mx"
}

wait_gpu_allocated() { # $1=gpus  -> 0 if all got >= 2GB, 1 on timeout
  local gpus=$1 i
  for i in $(seq 1 40); do
    if ! pgrep -f "trainer.experiment_name=$EXP" > /dev/null 2>&1; then return 1; fi
    if [ "$(gpu_mem "$gpus")" -ge 2000 ] 2>/dev/null; then return 0; fi
    sleep 15
  done
  return 1
}
run_one() {
  local SCRIPT=$1 ENVS=$2 PROJ=$3 EXP=$4 STEPS=$5
  if is_done "$PROJ" "$EXP" "$STEPS"; then log "$EXP done"; return 0; fi
  if ! is_alive "$EXP"; then
    log "$EXP launching"
    local gpus=$(echo "$ENVS" | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | cut -d= -f2)
    pause_idle_monitor "$gpus"
    cd "$BASE_DIR" || return 1
    env $ENVS setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    log "$EXP launched pid=$!"
    wait_gpu_allocated "$gpus"
    resume_idle_monitor
  else
    log "$EXP running, monitoring"
  fi
  local t=$(date +%s) restarts=0
  while true; do
    is_done "$PROJ" "$EXP" "$STEPS" && { log "$EXP DONE"; return 0; }
    if [ $(( $(date +%s) - t )) -gt 180 ] && ! is_alive "$EXP"; then
      restarts=$((restarts+1)); [ "$restarts" -gt 40 ] && { log "$EXP gave up"; return 1; }
      log "$EXP restart #$restarts in 100s"; sleep 100
      local gpus=$(echo "$ENVS" | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | cut -d= -f2)
      pause_idle_monitor "$gpus"
      cd "$BASE_DIR" || return 1
      env $ENVS setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
      wait_gpu_allocated "$gpus"
      resume_idle_monitor
      t=$(date +%s)
    fi
    sleep 60
  done
}

log "our-method priority relay started"

(
  # 剩余队列（已完成的 seed0/seed1/eps 寻优不再重复排）：先收尾正在跑的 BiPACE-Q seed2，
  # 再补跑之前被跳过、卡在 50/200 的 GRPO 1.5B seed0。
  run_one "$SCR_BIPACEQ15" "SEED=2 CUDA_VISIBLE_DEVICES=0,1,2,3" "$PROJ_ALF15" \
    "official_bipaceq_alfworld_1.5b_4gpu_L-12_eps0.05_firstn_counterfactual_seed2_thisnode" 200
  run_one "$SCR_SW15" "MODE=grpo START=sft SEED=0 MODEL_PATH=$SFT15 CUDA_VISIBLE_DEVICES=0,1,2,3" \
    "$PROJ_SW15" "grpo_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826" 200
  # ALF 7B 我们的方法 seed0 全新重跑（与 seed1 同用 _ai1845503 后缀，旧的无效 seed0 已删）。
  run_one "$SCR_ALF7" "SEED=0 TAG=ai1845503 CUDA_VISIBLE_DEVICES=0,1,2,3" "$PROJ_ALF7" \
    "bisimhead_alfworld_7b_4gpu_L-8_eps0.10_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_seed0_thisnode_ai1845503" 150
  for s in 0 1; do
    run_one "$SCR_OOD15" "MODE=gigpo SEED=$s CUDA_VISIBLE_DEVICES=0,1,2,3" "$PROJ_ALF15" \
      "gigpo_alfworld_ood_1.5b_4gpu_seed${s}_thisnode" 200
  done
  for s in 0 1; do
    run_one "$SCR_OOD7" "MODE=pabh SEED=$s CUDA_VISIBLE_DEVICES=0,1,2,3" "$PROJ_ALF7" \
      "bisimhead_alfworld_ood_7b_4gpu_L-8_eps0.10_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_seed${s}_thisnode" 150
  done
  for s in 0 1; do
    run_one "$SCR_OOD15" "MODE=grpo SEED=$s CUDA_VISIBLE_DEVICES=0,1,2,3" "$PROJ_ALF15" \
      "grpo_alfworld_ood_1.5b_4gpu_seed${s}_thisnode" 200
  done
  for s in 0 1; do
    run_one "$SCR_OOD7" "MODE=grpo SEED=$s CUDA_VISIBLE_DEVICES=0,1,2,3" "$PROJ_ALF7" \
      "grpo_alfworld_ood_7b_4gpu_seed${s}_thisnode" 150
  done
) >> "$LOG" 2>&1 &

(
  # seed0b 已按用户要求停止并删除；GPU4-7 直接补 7B OOD BiPACE-Q/GiGPO。
  for s in 0 1; do
    run_one "$SCR_OOD7" "MODE=bipaceq SEED=$s CUDA_VISIBLE_DEVICES=4,5,6,7" "$PROJ_ALF7" \
      "official_bipaceq_alfworld_ood_7b_4gpu_L-8_eps0.10_qstyle_seed${s}_thisnode" 150
  done
  for s in 0 1; do
    run_one "$SCR_OOD7" "MODE=gigpo SEED=$s CUDA_VISIBLE_DEVICES=4,5,6,7" "$PROJ_ALF7" \
      "gigpo_alfworld_ood_7b_4gpu_seed${s}_thisnode" 150
  done
) >> "$LOG" 2>&1 &

wait
log "our-method priority relay all DONE"
bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
log "relay exit"
