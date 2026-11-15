#!/usr/bin/env bash
# R7 补 seeds 看门狗（ai-1774370，2026-10-05）——落共享存储，防容器死亡丢失
# 监控两条 1.5B SFT 起点 run：
#   laneA: baseline seed1      GPU 0-3
#   laneB: bisimhead_prm seed2 GPU 4-7
# 行为：死亡 -> 错峰 100s 原地重启（ckpt 自动续，save_freq=50）-> 完成(200步)退出
# 口径注记：bisimhead_prm 的 head 不入 ckpt，resume 后 head 重置为正交初始化（已登记 §29.10）

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/sciworld/qwen1.5b/train_sciworld_1.5b_node1613826.sh
SFT_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
PROJ_DIR=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
LOG=$BASE_DIR/verl_log/sw_r7_watchdog.log
TOTAL_STEPS=200

EXP_A=baseline_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826
EXP_B=bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed2_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

is_done() { [ -d "$PROJ_DIR/$1/global_step_$TOTAL_STEPS" ]; }

is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }

launch() { # $1=exp_name $2=mode $3=seed $4=gpus
    log "launching $1 (mode=$2 seed=$3 gpus=$4)"
    cd "$BASE_DIR" || exit 1
    MODE=$2 SEED=$3 START=sft MODEL_PATH=$SFT_CKPT CUDA_VISIBLE_DEVICES=$4 \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    log "launched $1 pid=$!"
}

log "sw_r7 watchdog started (total_steps=$TOTAL_STEPS)"
doneA=0; doneB=0
while true; do
    # 完成判定（幂等，靠 global_step_200 目录）
    [ $doneA -eq 0 ] && is_done "$EXP_A" && { log "$EXP_A DONE"; doneA=1; }
    [ $doneB -eq 0 ] && is_done "$EXP_B" && { log "$EXP_B DONE"; doneB=1; }
    [ $doneA -eq 1 ] && [ $doneB -eq 1 ] && { log "both runs DONE, watchdog exit"; exit 0; }

    restarted=0
    if [ $doneA -eq 0 ] && ! is_alive "$EXP_A"; then
        log "$EXP_A not alive, restarting"
        launch "$EXP_A" baseline 1 "0,1,2,3"
        restarted=1
    fi
    if [ $doneB -eq 0 ] && ! is_alive "$EXP_B"; then
        [ $restarted -eq 1 ] && { log "stagger 100s before laneB restart (ray cluster collision avoidance)"; sleep 100; }
        log "$EXP_B not alive, restarting"
        launch "$EXP_B" bisimhead_prm 2 "4,5,6,7"
    fi
    sleep 60
done
