#!/usr/bin/env bash
# ALFWorld 阶梯看门狗（2026-10-09，lane B 专用版——1.5B raw-方法已按用户拍板取消）
# 等 7B raw-方法（pid $1, 4-7）收官 -> ALFWorld GiGPO（EST=gigpo）-> ALFWorld GRPO（EST=grpo）
# 每个：死亡->错峰重启（<=3 次）->ckpt 续->完成（global_step_200 判据）->下一个
set -u
ALF=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/alfworld/qwen1.5b_4xh100
CKPT_LAD=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_alfworld_ladder_thisnode
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/alf_ladder_watchdog.log
echo "[$(date)] alf-ladder watchdog started (watch 7B pid=$1)" >> "$LOG"

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 180; done; }
wait_free () {
    local deadline=$(( $(date +%s) + 3600 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local busy=0
        for g in 4 5 6 7; do
            local mem; mem=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
            [ "$mem" -gt 2000 ] && busy=1
        done
        [ "$busy" = "0" ] && return 0
        sleep 60
    done
    return 1
}
launch_alf () {  # $1=EST
    cd "$ALF" || return 1
    setsid nohup env EST=$1 SEED=0 CUDA_VISIBLE_DEVICES=4,5,6,7 bash train_ladder_repro_thisnode.sh > /dev/null 2>&1 < /dev/null &
    echo $!
}
run_one () {  # $1=EST
    local est=$1 restarts=0 pid dir
    dir=$CKPT_LAD/${est}_alfworld_1.5b_4gpu_seed0_thisnode
    [ -d "$dir/global_step_200" ] && { echo "[$(date)] $est already done, skip" >> "$LOG"; return 0; }
    wait_free
    pid=$(launch_alf "$est")
    echo "[$(date)] $est launched pid=$pid" >> "$LOG"
    while true; do
        sleep 300
        kill -0 "$pid" 2>/dev/null && continue
        [ -d "$dir/global_step_200" ] && { echo "[$(date)] $est COMPLETED" >> "$LOG"; return 0; }
        [ "$restarts" -ge 3 ] && { echo "[$(date)] ALARM: $est died x$restarts; give up" >> "$LOG"; return 1; }
        wait_free
        sleep 100
        pid=$(launch_alf "$est")
        restarts=$(( restarts + 1 ))
        echo "[$(date)] $est restarted pid=$pid (#$restarts)" >> "$LOG"
    done
}

wait_pid "$1"
echo "[$(date)] 7B raw-method exited" >> "$LOG"
run_one gigpo
run_one grpo
echo "[$(date)] alf-ladder watchdog done" >> "$LOG"
