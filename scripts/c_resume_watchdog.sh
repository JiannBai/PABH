#!/usr/bin/env bash
# C-resume 看门狗 + 收官后接力 midarm（2026-10-01，ai-1753446）
# 背景：C 臂两 seed 于 ~s150 遭 vllm cumem sleep 随机段错误（容器平台底噪）击杀；
#       本脚本 = 容错闭环：死了→错峰原地重启（自动从 s50 倍数 ckpt 续）→ 完成→接力 midarm。
# 注意：C 臂 head 不入 ckpt，每次 resume head 重置（已在 EXPERIMENTS_LOG §29.10 登记口径）。
set -u
SCI_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
ALF_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/alfworld/qwen1.5b_4xh100
SFT_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
CKPT_SCI=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/c_resume_watchdog.log
MAX_RESTARTS=4

echo "[$(date)] C-resume watchdog+midarm-relay started (C pids: $1 $2)" >> "$LOG"

wait_free_gpus () {
    local gpus=$1 deadline=$(( $(date +%s) + 2400 ))
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

launch_c () {  # $1=seed $2=gpus
    cd "$SCI_DIR" || return 1
    nohup env MODE=bisimhead_prm SEED=$1 MODEL_PATH=$SFT_CKPT CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 &
    echo $!
}

launch_midarm () {  # $1=seed $2=gpus
    cd "$ALF_DIR" || return 1
    nohup env SEED=$1 CUDA_VISIBLE_DEVICES=$2 bash train_midarm_g_actorhidden_thisnode.sh > /dev/null 2>&1 &
    echo $!
}

watch_until_done () {  # $1=pid $2=kind(c|midarm) $3=seed $4=gpus $5=stagger_s $6=ckpt_done_dir -> echoes final pid status
    local pid=$1 kind=$2 seed=$3 gpus=$4 stagger=$5 done_dir=$6 restarts=0
    while true; do
        sleep 300
        kill -0 "$pid" 2>/dev/null && continue
        if [ -d "$done_dir/global_step_200" ]; then
            echo "[$(date)] $kind seed$seed COMPLETED (global_step_200)" >> "$LOG"
            return 0
        fi
        [ "$restarts" -ge $MAX_RESTARTS ] && { echo "[$(date)] ALARM: $kind seed$seed died x$restarts; give up" >> "$LOG"; return 1; }
        sleep "$stagger"
        wait_free_gpus "$gpus"
        if [ "$kind" = "c" ]; then pid=$(launch_c "$seed" "$gpus"); else pid=$(launch_midarm "$seed" "$gpus"); fi
        restarts=$(( restarts + 1 ))
        echo "[$(date)] $kind seed$seed restarted pid=$pid (#$restarts; auto-resume from ckpt)" >> "$LOG"
    done
}

C0_DIR=$CKPT_SCI/bisimhead_prm_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826
C1_DIR=$CKPT_SCI/bisimhead_prm_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826

# ---- 阶段 1：守 C-resume 双 seed 到收官 ----
watch_until_done "$1" c 0 0,1,2,3 0 "$C0_DIR" &
W0=$!
watch_until_done "$2" c 1 4,5,6,7 100 "$C1_DIR" &
W1=$!
wait "$W0" "$W1"

# ---- 阶段 2：接力 midarm 双 seed（无 head，resume 完全干净）----
echo "[$(date)] C-resume both done; relaying to midarm" >> "$LOG"
wait_free_gpus 0,1,2,3
P_A=$(launch_midarm 0 0,1,2,3)
echo "[$(date)] midarm seed0 launched pid=$P_A" >> "$LOG"
sleep 100
wait_free_gpus 4,5,6,7
P_B=$(launch_midarm 1 4,5,6,7)
echo "[$(date)] midarm seed1 launched pid=$P_B" >> "$LOG"

M0_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode/midarm_alfworld_1.5b_4gpu_L-12_eps0.05_g_actorhidden_seed0_thisnode
M1_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode/midarm_alfworld_1.5b_4gpu_L-12_eps0.05_g_actorhidden_seed1_thisnode
watch_until_done "$P_A" midarm 0 0,1,2,3 0 "$M0_DIR" &
watch_until_done "$P_B" midarm 1 4,5,6,7 100 "$M1_DIR" &
wait
echo "[$(date)] all done" >> "$LOG"
