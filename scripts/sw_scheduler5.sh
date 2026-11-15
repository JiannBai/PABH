#!/usr/bin/env bash
# 调度器 v5（2026-10-09 16:45）：收编全部 orphan 续跑
# lane A（0-3）：SW 7B 方法 s1（在跑 pid $1）-> GiGPO s1 续 -> GiGPO s2 续 -> baseline s2 续 -> method s3 续
# lane B（4-7）：SW 7B baseline s1（在跑 pid $2）-> 7B raw-方法续（s100->150）-> ALF GiGPO -> ALF GRPO
set -u
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
SCI7=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen7b
ALF=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/alfworld/qwen1.5b_4xh100
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
SFT7_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_7b_sciworld_sft_node1613826/sft_full/global_step_483
CKPT15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
CKPT7=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_7b_sciworld_bipace_node1613826
CKPT_LAD=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_alfworld_ladder_thisnode
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/sw_scheduler5.log
MAXR=3

echo "[$(date)] sw-scheduler v5 started (watchA=$1 watchB=$2)" >> "$LOG"

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 180; done; }
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

launch_task () {  # $1=task $2=gpus
    case "$1" in
        resume_gigpo_1)
            cd "$SCI15" && setsid nohup env MODE=gigpo SEED=1 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        resume_gigpo_2)
            cd "$SCI15" && setsid nohup env MODE=gigpo SEED=2 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        resume_base_2)
            cd "$SCI15" && setsid nohup env MODE=baseline SEED=2 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        resume_method_s3)
            cd "$SCI15" && setsid nohup env MODE=bisimhead_prm SEED=3 START=sft MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        resume_7b_raw)
            cd "$SCI7" && setsid nohup env MODE=bisimhead_prm SEED=0 START=raw CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_7b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        alf_gigpo)
            cd "$ALF" && setsid nohup env EST=gigpo SEED=0 CUDA_VISIBLE_DEVICES=$2 bash train_ladder_repro_thisnode.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        alf_grpo)
            cd "$ALF" && setsid nohup env EST=grpo SEED=0 CUDA_VISIBLE_DEVICES=$2 bash train_ladder_repro_thisnode.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
    esac
}

marker_dir () {
    case "$1" in
        resume_gigpo_1) echo "$CKPT15/gigpo_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826" ;;
        resume_gigpo_2) echo "$CKPT15/gigpo_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed2_node1613826" ;;
        resume_base_2) echo "$CKPT15/baseline_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed2_node1613826" ;;
        resume_method_s3) echo "$CKPT15/bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed3_node1613826" ;;
        resume_7b_raw) echo "$CKPT7/bisimhead_prm_rawstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed0_node1613826" ;;
        alf_gigpo) echo "$CKPT_LAD/gigpo_alfworld_1.5b_4gpu_seed0_thisnode" ;;
        alf_grpo) echo "$CKPT_LAD/grpo_alfworld_1.5b_4gpu_seed0_thisnode" ;;
    esac
}

run_task () {  # $1=task $2=gpus $3=done_step $4=stagger
    local task=$1 gpus=$2 dstep=$3 stagger=$4 restarts=0 pid dir
    dir=$(marker_dir "$task")
    [ -d "$dir/global_step_$dstep" ] && { echo "[$(date)] $task already done, skip" >> "$LOG"; return 0; }
    sleep "$stagger"
    wait_free_gpus "$gpus"
    pid=$(launch_task "$task" "$gpus")
    echo "[$(date)] $task launched pid=$pid on $gpus" >> "$LOG"
    while true; do
        sleep 300
        kill -0 "$pid" 2>/dev/null && continue
        [ -d "$dir/global_step_$dstep" ] && { echo "[$(date)] $task COMPLETED" >> "$LOG"; return 0; }
        [ "$restarts" -ge $MAXR ] && { echo "[$(date)] ALARM: $task died x$restarts; give up" >> "$LOG"; return 1; }
        sleep "$stagger"
        wait_free_gpus "$gpus"
        pid=$(launch_task "$task" "$gpus")
        restarts=$(( restarts + 1 ))
        echo "[$(date)] $task restarted pid=$pid (#$restarts)" >> "$LOG"
    done
}

(
    wait_pid "$1"
    echo "[$(date)] lane A: SW7B method s1 exited" >> "$LOG"
    run_task resume_gigpo_1 0,1,2,3 200 0
    run_task resume_gigpo_2 0,1,2,3 200 100
    run_task resume_base_2 0,1,2,3 200 100
    run_task resume_method_s3 0,1,2,3 200 100
) &
(
    wait_pid "$2"
    echo "[$(date)] lane B: SW7B baseline s1 exited" >> "$LOG"
    run_task resume_7b_raw 4,5,6,7 150 100
    run_task alf_gigpo 4,5,6,7 200 100
    run_task alf_grpo 4,5,6,7 200 100
) &
wait
echo "[$(date)] sw-scheduler v5 all done" >> "$LOG"
