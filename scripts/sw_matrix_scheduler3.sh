#!/usr/bin/env bash
# 调度器 v3（2026-10-07，论文关键路径优先）
# lane A（GPU 0-3）：等 1.5B raw-baseline 退出 -> ScienceWorld GiGPO s0 -> 1.5B raw-方法 s0
# lane B（GPU 4-7）：等 7B raw-方法 退出 -> ALFWorld GiGPO s0 -> ALFWorld GRPO s0
# 死亡->错峰重启->ckpt 续->完成->下一任务；is_done 幂等。
set -u
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
ALF=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/alfworld/qwen1.5b_4xh100
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
CKPT15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
CKPT_LAD=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_alfworld_ladder_thisnode
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/sw_matrix_scheduler3.log
MAX_RESTARTS=4
PID_A=$1   # lane A 当前任务（1.5B raw-baseline）
PID_B=$2   # lane B 当前任务（7B raw-方法）

echo "[$(date)] sw-matrix scheduler v3 started (watchA=$PID_A watchB=$PID_B)" >> "$LOG"

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 120; done; }

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

launch_task () {  # $1=task $2=gpus -> echo pid
    case "$1" in
        sw_gigpo_s0)
            cd "$SCI15" && setsid nohup env MODE=gigpo SEED=0 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        rl15_prm_raw_s0)
            cd "$SCI15" && setsid nohup env MODE=bisimhead_prm SEED=0 START=raw CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        alf_gigpo_s0)
            cd "$ALF" && setsid nohup env EST=gigpo SEED=0 CUDA_VISIBLE_DEVICES=$2 bash train_ladder_repro_thisnode.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        alf_grpo_s0)
            cd "$ALF" && setsid nohup env EST=grpo SEED=0 CUDA_VISIBLE_DEVICES=$2 bash train_ladder_repro_thisnode.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
    esac
}

marker_dir () {
    case "$1" in
        sw_gigpo_s0) echo "$CKPT15/gigpo_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826" ;;
        rl15_prm_raw_s0) echo "$CKPT15/bisimhead_prm_rawstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826" ;;
        alf_gigpo_s0) echo "$CKPT_LAD/gigpo_alfworld_1.5b_4gpu_seed0_thisnode" ;;
        alf_grpo_s0) echo "$CKPT_LAD/grpo_alfworld_1.5b_4gpu_seed0_thisnode" ;;
    esac
}

is_done () { local d; d=$(marker_dir "$1"); [ -d "$d/global_step_200" ]; }

run_one () {  # $1=task $2=gpus $3=stagger
    local task=$1 gpus=$2 stagger=$3 restarts=0 pid
    if is_done "$task"; then echo "[$(date)] $task already done, skip" >> "$LOG"; return 0; fi
    sleep "$stagger"
    wait_free_gpus "$gpus"
    pid=$(launch_task "$task" "$gpus")
    echo "[$(date)] $task launched pid=$pid on $gpus" >> "$LOG"
    while true; do
        sleep 300
        kill -0 "$pid" 2>/dev/null && continue
        if is_done "$task"; then echo "[$(date)] $task COMPLETED" >> "$LOG"; return 0; fi
        [ "$restarts" -ge $MAX_RESTARTS ] && { echo "[$(date)] ALARM: $task died x$restarts; give up" >> "$LOG"; return 1; }
        sleep "$stagger"
        wait_free_gpus "$gpus"
        pid=$(launch_task "$task" "$gpus")
        restarts=$(( restarts + 1 ))
        echo "[$(date)] $task restarted pid=$pid (#$restarts)" >> "$LOG"
    done
}

(
    wait_pid "$PID_A"
    echo "[$(date)] lane A predecessor exited" >> "$LOG"
    run_one sw_gigpo_s0 0,1,2,3 0
    run_one rl15_prm_raw_s0 0,1,2,3 100
) &
(
    wait_pid "$PID_B"
    echo "[$(date)] lane B predecessor exited" >> "$LOG"
    run_one alf_gigpo_s0 4,5,6,7 100
    run_one alf_grpo_s0 4,5,6,7 100
) &
wait
echo "[$(date)] sw-matrix scheduler v3 all done" >> "$LOG"
