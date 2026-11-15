#!/usr/bin/env bash
# GiGPO 多环境复现编排（2026-10-08）：等 ScienceWorld GiGPO seed1 完成后——
#   GPU 0,1：WebShop GiGPO s0（2 卡官方口径）
#   GPU 2,3：Sokoban GiGPO s0（MODE=gigpo，SFT 不适用 -> raw Instruct 起点）
# 目的：回答"PABH vs GiGPO 同管线谁优"的全环境对照矩阵
set -u
WS=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/webshop/qwen1.5b_2gpu_official
SK=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sokoban/qwen1.5b
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
CKPT15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/gigpo_multi_env.log
GIGPO1_DIR=$CKPT15/gigpo_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826

echo "[$(date)] gigpo-multi-env launcher started" >> "$LOG"

# 等 GiGPO seed1 完成（global_step_200 出现）
while [ ! -d "$GIGPO1_DIR/global_step_200" ]; do sleep 300; done
echo "[$(date)] gigpo seed1 completed" >> "$LOG"

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

wait_free_gpus 0,1
cd "$WS" && setsid nohup env SEED=0 CUDA_VISIBLE_DEVICES=0,1 bash train_gigpo_official_node1613826.sh > /dev/null 2>&1 < /dev/null &
echo "[$(date)] webshop gigpo s0 launched pid=$! (GPU 0,1)" >> "$LOG"
sleep 100
wait_free_gpus 2,3
cd "$SK" && setsid nohup env MODE=gigpo SEED=0 CUDA_VISIBLE_DEVICES=2,3 bash train_sokoban_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null &
echo "[$(date)] sokoban gigpo s0 launched pid=$! (GPU 2,3)" >> "$LOG"
echo "[$(date)] launcher done" >> "$LOG"
