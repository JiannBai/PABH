#!/usr/bin/env bash
# pivot 链：等 SW GiGPO seed1 退出 -> ALFWorld gigpo_gtext（0-3）-> method seed3（0-3）
set -u
ALF=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/alfworld/qwen1.5b_4xh100
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/pivot_chain.log
echo "[$(date)] pivot-chain watcher started (watch gigpo1 pid=$1)" >> "$LOG"

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
cd "$ALF" && setsid nohup env EST=gigpo_gtext SEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_ladder_repro_thisnode.sh > /dev/null 2>&1 < /dev/null &
PIVOT=$!
echo "[$(date)] PIVOT gigpo_gtext launched pid=$PIVOT (GPU 0-3, ALFWorld)" >> "$LOG"
wait_pid "$PIVOT"
echo "[$(date)] pivot exited" >> "$LOG"
wait_free
sleep 100
cd "$SCI15" && setsid nohup env MODE=bisimhead_prm SEED=3 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null &
echo "[$(date)] method seed3 launched pid=$! (GPU 0-3)" >> "$LOG"
