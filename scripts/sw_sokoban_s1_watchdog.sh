#!/usr/bin/env bash
# Sokoban seed1 双臂看门狗（ai-1774370，2026-10-08）——落共享存储
# 机器控制设计：Sokoban s0/s1 双臂全部在 ai-1774370 同机，方法-vs-baseline 对照零机器混杂
#   laneA: baseline seed1        GPU 0-3
#   laneB: bisimhead_prm seed1   GPU 4-7（box_prm 锚）
# 行为：死亡 -> 错峰 100s 原地重启（ckpt 自动续，save_freq=50）-> 完成(200步)退出
# 退出时自动拉起 gpu_idle_monitor（防 GPU 空闲被回收）
# 口径注记：bisimhead_prm 的 head 不入 ckpt，resume 后 head 重置为正交初始化（已登记 §29.10）

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/sokoban/qwen1.5b/train_sokoban_1.5b_node1613826.sh
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct
PROJ_DIR=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sokoban_bipace_node1613826
LOG=$BASE_DIR/verl_log/sw_sokoban_s1_watchdog.log
MONITOR_DIR=/mnt/tidalfs-hssh01/dataset/bxd
TOTAL_STEPS=200

EXP_A=baseline_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826
EXP_B=bisimhead_prm_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

is_done() { [ -d "$PROJ_DIR/$1/global_step_$TOTAL_STEPS" ]; }

is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }

launch() { # $1=exp_name $2=mode $3=seed $4=gpus
    log "launching $1 (mode=$2 seed=$3 gpus=$4)"
    cd "$BASE_DIR" || exit 1
    MODE=$2 SEED=$3 MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=$4 \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    log "launched $1 pid=$!"
}

log "sw_sokoban seed1 watchdog started (total_steps=$TOTAL_STEPS)"
doneA=0; doneB=0
while true; do
    [ $doneA -eq 0 ] && is_done "$EXP_A" && { log "$EXP_A DONE"; doneA=1; }
    [ $doneB -eq 0 ] && is_done "$EXP_B" && { log "$EXP_B DONE"; doneB=1; }
    if [ $doneA -eq 1 ] && [ $doneB -eq 1 ]; then
        log "both runs DONE, starting gpu_idle_monitor, watchdog exit"
        (cd "$MONITOR_DIR" && setsid nohup python3 gpu_idle_monitor.py >> gpu_idle_monitor.out 2>&1 &)
        exit 0
    fi

    restarted=0
    if [ $doneA -eq 0 ] && ! is_alive "$EXP_A"; then
        log "$EXP_A not alive, restarting"
        launch "$EXP_A" baseline 1 "0,1,2,3"
        restarted=1
    fi
    if [ $doneB -eq 0 ] && ! is_alive "$EXP_B"; then
        [ $restarted -eq 1 ] && { log "stagger 100s before laneB restart (ray cluster collision avoidance)"; sleep 100; }
        log "$EXP_B not alive, restarting"
        launch "$EXP_B" bisimhead_prm 1 "4,5,6,7"
    fi
    sleep 60
done
