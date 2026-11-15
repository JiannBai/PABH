#!/usr/bin/env bash
# ScienceWorld 全量矩阵调度器 v2（2026-10-04，10/4 同名覆盖事故后重建）
# 修复：①任务命名带 START 标签（raw/sft 起点彻底分目录）；②launch 用 setsid+全 fd 重定向（修日志延迟 bug）
# 泳道 A（GPU 0-3）：SFT-baseline s0【从 s100 续】→ SFT-方法 s0 → 1.5B raw-baseline s0
# 泳道 B（GPU 4-7）：raw-baseline s0【全新】→ raw-方法 s0 → 1.5B raw-方法 s0
set -u
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
SCI7=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen7b
CKPT7=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_7b_sciworld_bipace_node1613826
CKPT15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
SFT7_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_7b_sciworld_sft_node1613826/sft_full/global_step_483
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/sw_matrix_scheduler2.log
MAX_RESTARTS=4

echo "[$(date)] sw-matrix scheduler v2 started" >> "$LOG"

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

launch_task () {  # $1=task_id $2=gpus -> echo pid
    case "$1" in
        rl7_base_sft_s0)
            cd "$SCI7" && setsid nohup env MODE=baseline SEED=0 START=sft MODEL_PATH=$SFT7_CKPT CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_7b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        rl7_prm_sft_s0)
            cd "$SCI7" && setsid nohup env MODE=bisimhead_prm SEED=0 START=sft MODEL_PATH=$SFT7_CKPT CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_7b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        rl7_base_raw_s0)
            cd "$SCI7" && setsid nohup env MODE=baseline SEED=0 START=raw CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_7b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        rl7_prm_raw_s0)
            cd "$SCI7" && setsid nohup env MODE=bisimhead_prm SEED=0 START=raw CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_7b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        rl15_base_raw_s0)
            cd "$SCI15" && setsid nohup env MODE=baseline SEED=0 START=raw CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        rl15_prm_raw_s0)
            cd "$SCI15" && setsid nohup env MODE=bisimhead_prm SEED=0 START=raw CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
    esac
}

marker_dir () {  # $1=task_id
    case "$1" in
        rl7_base_sft_s0) echo "$CKPT7/baseline_sftstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed0_node1613826" ;;
        rl7_prm_sft_s0)  echo "$CKPT7/bisimhead_prm_sftstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed0_node1613826" ;;
        rl7_base_raw_s0) echo "$CKPT7/baseline_rawstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed0_node1613826" ;;
        rl7_prm_raw_s0)  echo "$CKPT7/bisimhead_prm_rawstart_sciworld_7b_4gpu_L-8_eps0.10_qstyle_actiontag_seed0_node1613826" ;;
        rl15_base_raw_s0) echo "$CKPT15/baseline_rawstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826" ;;
        rl15_prm_raw_s0)  echo "$CKPT15/bisimhead_prm_rawstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826" ;;
    esac
}

is_done () { local d; d=$(marker_dir "$1"); [ -d "$d/global_step_150" ] || [ -d "$d/global_step_200" ]; }

run_queue () {  # $1=gpus $2=stagger $3..=task_ids
    local gpus=$1 stagger=$2; shift 2
    for task in "$@"; do
        if is_done "$task"; then echo "[$(date)] $task already done, skip" >> "$LOG"; continue; fi
        local restarts=0 pid
        sleep "$stagger"
        wait_free_gpus "$gpus"
        pid=$(launch_task "$task" "$gpus")
        echo "[$(date)] $task launched pid=$pid on $gpus" >> "$LOG"
        while true; do
            sleep 300
            kill -0 "$pid" 2>/dev/null && continue
            if is_done "$task"; then echo "[$(date)] $task COMPLETED" >> "$LOG"; break; fi
            [ "$restarts" -ge $MAX_RESTARTS ] && { echo "[$(date)] ALARM: $task died x$restarts; give up" >> "$LOG"; break; }
            sleep "$stagger"
            wait_free_gpus "$gpus"
            pid=$(launch_task "$task" "$gpus")
            restarts=$(( restarts + 1 ))
            echo "[$(date)] $task restarted pid=$pid (#$restarts, auto-resume)" >> "$LOG"
        done
    done
}

run_queue 0,1,2,3 0 rl7_base_sft_s0 rl7_prm_sft_s0 rl15_base_raw_s0 &
run_queue 4,5,6,7 100 rl7_base_raw_s0 rl7_prm_raw_s0 rl15_prm_raw_s0 &
wait
echo "[$(date)] sw-matrix scheduler v2 all done" >> "$LOG"
