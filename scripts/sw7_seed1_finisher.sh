#!/usr/bin/env bash
# SW 7B seed1 双臂收尾链（2026-10-11）
# 等 SW 7B method s1（pid $1, 0-3）完成（global_step_150）-> baseline s1 续跑（0-3）
set -u
SCI7=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen7b
SFT7_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_7b_sciworld_sft_node1613826/sft_full/global_step_483
CKPT7=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_7b_sciworld_bipace_node1613826
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/sw7_seed1_finisher.log
MAXR=3
echo "[$(date)] sw7-seed1 finisher started (watch method s1 pid=$1)" >> "$LOG"

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 180; done; }
wait_free_gpus () {
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

DIR_A=$CKPT7/bisimhead_prm_sftstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed1_node1613826
DIR_B=$CKPT7/baseline_sftstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed1_node1613826

# method s1 守护（死亡续跑，完成判据 global_step_150）
MPID=$1
restarts=0
while true; do
    sleep 300
    kill -0 "$MPID" 2>/dev/null && continue
    [ -d "$DIR_A/global_step_150" ] && { echo "[$(date)] method s1 COMPLETED" >> "$LOG"; break; }
    [ "$restarts" -ge $MAXR ] && { echo "[$(date)] ALARM: method s1 died x$restarts" >> "$LOG"; break; }
    wait_free_gpus
    cd "$SCI7" && setsid nohup env MODE=bisimhead_prm SEED=1 START=sft MODEL_PATH=$SFT7_CKPT CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sciworld_7b_node1613826.sh > /dev/null 2>&1 < /dev/null &
    MPID=$!
    restarts=$(( restarts + 1 ))
    echo "[$(date)] method s1 restarted pid=$MPID (#$restarts)" >> "$LOG"
done

# baseline s1 续跑（global_step_150 判据）
if [ ! -d "$DIR_B/global_step_150" ]; then
    wait_free_gpus
    sleep 100
    cd "$SCI7" && setsid nohup env MODE=baseline SEED=1 START=sft MODEL_PATH=$SFT7_CKPT CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sciworld_7b_node1613826.sh > /dev/null 2>&1 < /dev/null &
    BPID=$!
    echo "[$(date)] baseline s1 resume launched pid=$BPID" >> "$LOG"
    restarts=0
    while true; do
        sleep 300
        kill -0 "$BPID" 2>/dev/null && continue
        [ -d "$DIR_B/global_step_150" ] && { echo "[$(date)] baseline s1 COMPLETED" >> "$LOG"; break; }
        [ "$restarts" -ge $MAXR ] && { echo "[$(date)] ALARM: baseline s1 died x$restarts" >> "$LOG"; break; }
        wait_free_gpus
        sleep 100
        cd "$SCI7" && setsid nohup env MODE=baseline SEED=1 START=sft MODEL_PATH=$SFT7_CKPT CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sciworld_7b_node1613826.sh > /dev/null 2>&1 < /dev/null &
        BPID=$!
        restarts=$(( restarts + 1 ))
        echo "[$(date)] baseline s1 restarted pid=$BPID (#$restarts)" >> "$LOG"
    done
else
    echo "[$(date)] baseline s1 already done" >> "$LOG"
fi
echo "[$(date)] sw7-seed1 finisher done" >> "$LOG"
