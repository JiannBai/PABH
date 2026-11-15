#!/usr/bin/env bash
# GiGPO 本体验证链（2026-10-08，pivot 暂缓版）
# lane A：等 SW GiGPO seed1 退出 -> ALFWorld plain GiGPO（EST=gigpo, 0-3）-> SW method seed3（0-3）
set -u
ALF=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/alfworld/qwen1.5b_4xh100
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/gigpo_validation_chain.log
echo "[$(date)] gigpo-validation chain started (watch gigpo1 pid=$1)" >> "$LOG"

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

wait_pid "$1"
echo "[$(date)] gigpo seed1 exited" >> "$LOG"
wait_free
cd "$ALF" && setsid nohup env EST=gigpo SEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_ladder_repro_thisnode.sh > /dev/null 2>&1 < /dev/null &
GP=$!
echo "[$(date)] ALFWorld plain GiGPO launched pid=$GP (GPU 0-3)" >> "$LOG"
wait_pid "$GP"
echo "[$(date)] ALFWorld plain GiGPO exited" >> "$LOG"
wait_free
sleep 100
cd "$SCI15" && setsid nohup env MODE=bisimhead_prm SEED=3 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null &
echo "[$(date)] method seed3 launched pid=$! (GPU 0-3)" >> "$LOG"
