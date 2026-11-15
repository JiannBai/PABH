#!/usr/bin/env bash
# 估计器阶梯复现（GiGPO/GRPO）排队启动器（2026-10-07，ai-1753446）
# lane A：等 1.5B raw-baseline 退出 -> EST=gigpo seed0（GPU 0-3）
# lane B：等 scheduler v2 全队列清空（raw-方法 + 1.5B raw-方法 跑完）-> EST=grpo seed0（GPU 4-7）
set -u
ALF_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/alfworld/qwen1.5b_4xh100
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/ladder_launcher.log
CUR_PID_A=$1   # 当前 lane A 任务 pid（1.5B raw-baseline）
SCHED_PID=$2   # scheduler v2 pid（其 lane B 队列含 raw-方法、1.5B raw-方法）

echo "[$(date)] ladder launcher started (watchA=$CUR_PID_A sched=$SCHED_PID)" >> "$LOG"

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

launch () {  # $1=EST $2=gpus
    cd "$ALF_DIR" || return 1
    setsid nohup env EST=$1 SEED=0 CUDA_VISIBLE_DEVICES=$2 bash train_ladder_repro_thisnode.sh > /dev/null 2>&1 < /dev/null &
    echo $!
}

# lane A：当前任务退出即上 gigpo
(
    wait_pid "$CUR_PID_A"
    wait_free_gpus 0,1,2,3
    P=$(launch gigpo 0,1,2,3)
    echo "[$(date)] ladder gigpo launched pid=$P (GPU 0-3)" >> "$LOG"
) &

# lane B：scheduler v2 整个队列跑完（进程退出）且显存释放后上 grpo
(
    wait_pid "$SCHED_PID"
    wait_free_gpus 4,5,6,7
    sleep 100
    wait_free_gpus 4,5,6,7
    P=$(launch grpo 4,5,6,7)
    echo "[$(date)] ladder grpo launched pid=$P (GPU 4-7)" >> "$LOG"
) &

wait
echo "[$(date)] ladder launcher done" >> "$LOG"
