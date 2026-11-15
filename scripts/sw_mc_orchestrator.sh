#!/usr/bin/env bash
# ScienceWorld 机器控制三臂编排器（2026-10-08，ai-1753446）
# 目的：排除机器差异混杂——GiGPO/baseline/方法 三臂全部在当前容器同机各补 1 seed。
# 触发：GiGPO s0（pid $1）退出后——
#   lane A（0-3）：方法 seed3（bisimhead_prm, SFT 起点）-> GiGPO seed1
#   lane B（4-7）：抢占 7B raw-方法（pid $2，ckpt 保留）-> baseline seed2 -> 恢复 7B raw-方法
set -u
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
SCI7=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen7b
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/sw_mc_orchestrator.log
GIGPO_PID=$1
RAW7_PID=$2

echo "[$(date)] sw-mc orchestrator started (gigpo_pid=$GIGPO_PID raw7_pid=$RAW7_PID)" >> "$LOG"

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 120; done; }

wait_free_gpus () {
    local gpus=$1 deadline=$(( $(date +%s) + 3600 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local busy=0
        IFS=',' read -ra arr <<< "$gpus"
        for g in "${arr[@]}"; do
            local mem
            mem=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
            [ "$mem" -gt 2000 ] && busy=1
        done
        [ "$busy" = "0" ] && return 0
        sleep 60
    done
    return 1
}

launch15 () {  # $1=MODE $2=SEED $3=gpus
    cd "$SCI15" || return 1
    setsid nohup env MODE=$1 SEED=$2 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=$3 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null &
    echo $!
}
launch7raw () {  # 恢复 7B raw-方法（START=raw，目录已有 ckpt 自动 resume）
    cd "$SCI7" || return 1
    setsid nohup env MODE=bisimhead_prm SEED=0 START=raw CUDA_VISIBLE_DEVICES=$1 bash train_sciworld_7b_node1613826.sh > /dev/null 2>&1 < /dev/null &
    echo $!
}

# ---- lane A：GiGPO s0 退出 -> 方法 seed3 -> GiGPO seed1 ----
(
    wait_pid "$GIGPO_PID"
    echo "[$(date)] gigpo s0 exited" >> "$LOG"
    wait_free_gpus 0,1,2,3
    P=$(launch15 bisimhead_prm 3 0,1,2,3)
    echo "[$(date)] method seed3 launched pid=$P (GPU 0-3)" >> "$LOG"
    wait_pid "$P"
    echo "[$(date)] method seed3 exited" >> "$LOG"
    wait_free_gpus 0,1,2,3
    P=$(launch15 gigpo 1 0,1,2,3)
    echo "[$(date)] gigpo seed1 launched pid=$P (GPU 0-3)" >> "$LOG"
) &

# ---- lane B：等 GiGPO s0 退出后，抢占 7B raw-方法 -> baseline seed2 -> 恢复 ----
(
    wait_pid "$GIGPO_PID"
    sleep 60
    # 抢占：杀 7B raw-方法（其 scheduler v3 的 lane B watcher 会尝试重启它——先杀 v3 lane B  watcher 防冲突）
    pkill -f "sw_matrix_scheduler3" 2>/dev/null
    kill -9 "$RAW7_PID" 2>/dev/null
    sleep 30
    # 清其 ray 集群（按 GPU 侧限定：此时 4-7 上只有它）
    for p in $(ps -ef | grep -E "raylet|gcs_server|ray::|default_worker" | grep -v grep | awk '{print $2}'); do kill -9 "$p" 2>/dev/null; done
    wait_free_gpus 4,5,6,7
    sleep 100
    P=$(launch15 baseline 2 4,5,6,7)
    echo "[$(date)] baseline seed2 launched pid=$P (GPU 4-7; 7B raw-method preempted, ckpt retained)" >> "$LOG"
    wait_pid "$P"
    echo "[$(date)] baseline seed2 exited; resuming 7B raw-method" >> "$LOG"
    wait_free_gpus 4,5,6,7
    sleep 100
    P=$(launch7raw 4,5,6,7)
    echo "[$(date)] 7B raw-method resumed pid=$P (GPU 4-7)" >> "$LOG"
    # 重新挂上 ALFWorld 阶梯调度（scheduler v3 lane B：等 7B 完 -> alf_gigpo -> alf_grpo）
    if [ -x /mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/scripts/sw_matrix_scheduler3.sh ]; then
        setsid nohup /mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/scripts/sw_matrix_scheduler3.sh 999999999 "$P" > /dev/null 2>&1 < /dev/null &
        echo "[$(date)] ALF ladder scheduler re-armed pid=$! (watch 7B pid=$P)" >> "$LOG"
    fi
) &

wait
echo "[$(date)] sw-mc orchestrator done" >> "$LOG"
