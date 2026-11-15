#!/usr/bin/env bash
# ScienceWorld GiGPO 优先链（2026-10-08 用户拍板）
# lane A：等 SW GiGPO seed1 退出 -> SW GiGPO seed2（0-3）-> SW method seed3（0-3）
set -u
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/sw_gigpo_priority_chain.log
echo "[$(date)] sw-gigpo-priority chain started (watch gigpo1 pid=$1)" >> "$LOG"

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

wait_pid "$1"
echo "[$(date)] gigpo seed1 exited" >> "$LOG"
wait_free
P=$(launch15 gigpo 2)
echo "[$(date)] SW GiGPO seed2 launched pid=$P (GPU 0-3)" >> "$LOG"
wait_pid "$P"
echo "[$(date)] gigpo seed2 exited" >> "$LOG"
wait_free
sleep 100
P=$(launch15 bisimhead_prm 3)
echo "[$(date)] method seed3 launched pid=$P (GPU 0-3)" >> "$LOG"
