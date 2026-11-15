#!/usr/bin/env bash
# ScienceWorld 三个 segfault 死亡 run 的续跑链（2026-10-09）
# 等 method seed3（pid $1, 0-3）退出后依次 resume：
#   GiGPO s1（s100->200）-> GiGPO s2（s100->200）-> baseline s2（s100->200）
# 每个：死亡再重启（<=3 次，自动从最新 ckpt 续），完成=global_step_200
set -u
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
CKPT15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/sw_resume_chain.log
echo "[$(date)] sw-resume chain started (watch method s3 pid=$1)" >> "$LOG"

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 180; done; }
wait_free () {
    local deadline=$(( $(date +%s) + 3600 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local busy=0
        for g in 0 1 2 3; do
            local mem; mem=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
            [ "$mem" -gt 2000 ] && busy=1
        done
        [ "$busy" = "0" ] && return 0
        sleep 60
    done
    return 1
}
launch15 () {  # $1=MODE $2=SEED
    cd "$SCI15" || return 1
    setsid nohup env MODE=$1 SEED=$2 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null &
    echo $!
}
marker () {
    case "$1" in
        gigpo_1) echo "$CKPT15/gigpo_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826" ;;
        gigpo_2) echo "$CKPT15/gigpo_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed2_node1613826" ;;
        baseline_2) echo "$CKPT15/baseline_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed2_node1613826" ;;
    esac
}
run_resume () {  # $1=key $2=MODE $3=SEED
    local key=$1 mode=$2 seed=$3 restarts=0 pid dir
    dir=$(marker "$key")
    [ -d "$dir/global_step_200" ] && { echo "[$(date)] $key already done, skip" >> "$LOG"; return 0; }
    wait_free
    pid=$(launch15 "$mode" "$seed")
    echo "[$(date)] $key resume launched pid=$pid" >> "$LOG"
    while true; do
        sleep 300
        kill -0 "$pid" 2>/dev/null && continue
        [ -d "$dir/global_step_200" ] && { echo "[$(date)] $key COMPLETED" >> "$LOG"; return 0; }
        [ "$restarts" -ge 3 ] && { echo "[$(date)] ALARM: $key died x$restarts; give up" >> "$LOG"; return 1; }
        wait_free
        sleep 100
        pid=$(launch15 "$mode" "$seed")
        restarts=$(( restarts + 1 ))
        echo "[$(date)] $key re-resumed pid=$pid (#$restarts)" >> "$LOG"
    done
}

wait_pid "$1"
echo "[$(date)] method s3 exited" >> "$LOG"
run_resume gigpo_1 gigpo 1
run_resume gigpo_2 gigpo 2
run_resume baseline_2 baseline 2
echo "[$(date)] sw-resume chain all done" >> "$LOG"
