#!/usr/bin/env bash
# 调度器 v6（2026-10-12 08:40）：g_text 优先——方法臂全环境带 g_text 重跑计划
# 背景：g_text（三层栈第 1 层）为方法专有；SW/Sokoban 此前无机制，已补实现（gate=bisim_head）。
# 旧 SW 方法 run（无 g_text）保留为 gen1 消融；本链跑 gen2（TAG=gtext）。
# lane A（0-3）：SW 1.5B 方法 gtext s0 → s1 → s2
# lane B（4-7）：ALF GRPO 续（s150→200）→ ALFWorld midarm-gtext s0 → s1（真归因臂）
set -u
SCI15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/sciworld/qwen1.5b
ALF=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/examples/bipace/alfworld/qwen1.5b_4xh100
SFT15_CKPT=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
CKPT15=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
CKPT_LAD=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_alfworld_ladder_thisnode
CKPT_MID=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/sw_scheduler6.log
MAXR=3
echo "[$(date)] sw-scheduler v6 (gtext priority) started" >> "$LOG"

wait_free_gpus () {
    local gpus=$1 deadline=$(( $(date +%s) + 7200 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local busy=0
        IFS=',' read -ra arr <<< "$gpus"
        for g in "${arr[@]}"; do
            local mem
            mem=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
            [ "$mem" -gt 2000 ] && busy=1
        done
        [ "$busy" = "0" ] && return 0
        sleep 120
    done
    return 1
}

launch_task () {  # $1=task $2=gpus
    case "$1" in
        sw_m_gtext_s0|sw_m_gtext_s1|sw_m_gtext_s2)
            local seed=${1##*_s}
            cd "$SCI15" && setsid nohup env MODE=bisimhead_prm SEED=$seed START=sft TAG=gtext MODEL_PATH=$SFT15_CKPT CUDA_VISIBLE_DEVICES=$2 bash train_sciworld_1.5b_node1613826.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        alf_grpo_resume)
            cd "$ALF" && setsid nohup env EST=grpo SEED=0 CUDA_VISIBLE_DEVICES=$2 bash train_ladder_repro_thisnode.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
        mid_gtext_s0|mid_gtext_s1)
            local seed=${1##*_s}
            cd "$ALF" && setsid nohup env SEED=$seed GTEXT=1 CUDA_VISIBLE_DEVICES=$2 bash train_midarm_g_actorhidden_thisnode.sh > /dev/null 2>&1 < /dev/null & echo $! ;;
    esac
}

marker_dir () {
    case "$1" in
        sw_m_gtext_s0) echo "$CKPT15/bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_gtext_node1613826" ;;
        sw_m_gtext_s1) echo "$CKPT15/bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_gtext_node1613826" ;;
        sw_m_gtext_s2) echo "$CKPT15/bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed2_gtext_node1613826" ;;
        alf_grpo_resume) echo "$CKPT_LAD/grpo_alfworld_1.5b_4gpu_seed0_thisnode" ;;
        mid_gtext_s0) echo "$CKPT_MID/midarm_g_alfworld_1.5b_4gpu_L-12_eps0.05_gtext_seed0_thisnode" ;;
        mid_gtext_s1) echo "$CKPT_MID/midarm_g_alfworld_1.5b_4gpu_L-12_eps0.05_gtext_seed1_thisnode" ;;
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
    run_task sw_m_gtext_s0 0,1,2,3 200 0
    run_task sw_m_gtext_s1 0,1,2,3 200 100
    run_task sw_m_gtext_s2 0,1,2,3 200 100
    echo "[$(date)] lane A gtext trio done" >> "$LOG"
) &
(
    run_task alf_grpo_resume 4,5,6,7 200 0
    run_task mid_gtext_s0 4,5,6,7 200 100
    run_task mid_gtext_s1 4,5,6,7 200 100
    echo "[$(date)] lane B grpo+midarm-gtext done" >> "$LOG"
) &
wait
echo "[$(date)] sw-scheduler v6 all done" >> "$LOG"
