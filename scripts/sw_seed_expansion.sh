#!/usr/bin/env bash
# ScienceWorld 1.5B 多 seed 扩样计划（2026-10-12，本机 lane A 部分）
# SW 7B method s1（pid $1）+ baseline s1（finisher 链接续）全部收官后：
#   0-3 卡依次跑：GiGPO seed3 → GiGPO seed4 → baseline seed3 → baseline seed4
# 每个 run：守护（死亡续跑 ≤3 次），完成 = global_step_200
set -u
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
CKPT15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/sw_seed_expansion.log
MAXR=3
echo "[$(date)] sw-seed-expansion watcher started (wait for sw7 finisher pid=$1)" >> "$LOG"

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 300; done; }
wait_free_gpus () {
    local deadline=$(( $(date +%s) + 7200 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local busy=0
        for g in 0 1 2 3; do
            local mem; mem=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
            [ "$mem" -gt 2000 ] && busy=1
        done
        [ "$busy" = "0" ] && return 0
        sleep 120
    done
    return 1
}

launch15 () {  # $1=MODE $2=SEED
    cd "$SCI15" || return 1
    setsid nohup env MODE=$1 SEED=$2 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null &
    echo $!
}

run_task () {  # $1=MODE $2=SEED $3=stagger
    local mode=$1 seed=$2 stagger=$3 restarts=0 pid
    local dir=$CKPT15/${mode}_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${seed}_node1613826
    [ -d "$dir/global_step_200" ] && { echo "[$(date)] $mode s$seed already done, skip" >> "$LOG"; return 0; }
    sleep "$stagger"
    wait_free_gpus
    pid=$(launch15 "$mode" "$seed")
    echo "[$(date)] $mode seed$seed launched pid=$pid" >> "$LOG"
    while true; do
        sleep 300
        kill -0 "$pid" 2>/dev/null && continue
        [ -d "$dir/global_step_200" ] && { echo "[$(date)] $mode seed$seed COMPLETED" >> "$LOG"; return 0; }
        [ "$restarts" -ge $MAXR ] && { echo "[$(date)] ALARM: $mode s$seed died x$restarts" >> "$LOG"; return 1; }
        wait_free_gpus
        sleep 100
        pid=$(launch15 "$mode" "$seed")
        restarts=$(( restarts + 1 ))
        echo "[$(date)] $mode seed$seed restarted pid=$pid (#$restarts)" >> "$LOG"
    done
}

# 等 sw7_seed1_finisher 链整体结束（其进程退出 = method s1 + baseline s1 都完成）
wait_pid "$1"
echo "[$(date)] sw7 seed1 finisher exited; starting SW 1.5B seed expansion" >> "$LOG"
run_task gigpo 3 0
run_task gigpo 4 100
run_task baseline 3 100
run_task baseline 4 100
echo "[$(date)] sw-seed-expansion all done" >> "$LOG"
