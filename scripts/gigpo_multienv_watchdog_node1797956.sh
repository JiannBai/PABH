#!/usr/bin/env bash
# GiGPO 本体验证三 run 看门狗（ai-1797956，2026-10-09）——落共享存储
# 接手已死的 gigpo_multi_env_launcher（其死等永不出现的 global_step_200）
#   laneA: Sokoban GiGPO s0    GPU 0-3（200 步）
#   laneB: WebShop GiGPO s0    GPU 4,5（官方口径 2 卡，200 步）
#   laneC: WebShop GiGPO s2    GPU 6,7（同上）
# 行为：死亡 -> 错峰 100s 原地重启（ckpt 自动续；GiGPO 无 head，resume 干净）
#       -> 完成（global_step_200 目录判据，教训 §29.28：不看进程退出）-> 退出
# 退出时若 gpu_idle_monitor 不在则自动拉起（防 GPU 空闲被回收）

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct
LOG=$BASE_DIR/verl_log/gigpo_multienv_watchdog_node1797956.log
MONITOR_DIR=/mnt/tidalfs-hssh01/dataset/bxd
TOTAL_STEPS=200

SOKO_SCRIPT=$BASE_DIR/examples/bipace/sokoban/qwen1.5b/train_sokoban_1.5b_node1613826.sh
WEB_SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen1.5b_2gpu_official/train_gigpo_official_node1613826.sh

EXP_A=gigpo_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826
CKPT_A=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sokoban_bipace_node1613826/$EXP_A

EXP_B=gigpo_official_webshop_1.5b_2gpu_L-12_seed0_node1613826
CKPT_B=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$EXP_B

EXP_C=gigpo_official_webshop_1.5b_2gpu_L-12_seed2_node1613826
CKPT_C=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$EXP_C

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

is_done() { [ -d "$1/global_step_$TOTAL_STEPS" ]; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }

log "gigpo multienv watchdog started (ai-1797956, total_steps=$TOTAL_STEPS)"
doneA=0; doneB=0; doneC=0
while true; do
    [ $doneA -eq 0 ] && is_done "$CKPT_A" && { log "$EXP_A DONE"; doneA=1; }
    [ $doneB -eq 0 ] && is_done "$CKPT_B" && { log "$EXP_B DONE"; doneB=1; }
    [ $doneC -eq 0 ] && is_done "$CKPT_C" && { log "$EXP_C DONE"; doneC=1; }
    if [ $doneA -eq 1 ] && [ $doneB -eq 1 ] && [ $doneC -eq 1 ]; then
        log "all three runs DONE"
        if ! pgrep -f "gpu_idle_monitor.py" > /dev/null 2>&1; then
            log "gpu_idle_monitor not running, starting it"
            (cd "$MONITOR_DIR" && setsid nohup python3 gpu_idle_monitor.py >> gpu_idle_monitor.out 2>&1 &)
        else
            log "gpu_idle_monitor already running, skip"
        fi
        exit 0
    fi

    restarted=0
    if [ $doneA -eq 0 ] && ! is_alive "$EXP_A"; then
        log "$EXP_A not alive, restarting"
        cd "$BASE_DIR" || exit 1
        MODE=gigpo SEED=0 MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=0,1,2,3 \
            setsid nohup bash "$SOKO_SCRIPT" >> /dev/null 2>&1 &
        restarted=1
    fi
    if [ $doneB -eq 0 ] && ! is_alive "$EXP_B"; then
        [ $restarted -eq 1 ] && { log "stagger 100s before laneB restart"; sleep 100; }
        log "$EXP_B not alive, restarting"
        cd "$BASE_DIR" || exit 1
        SEED=0 CUDA_VISIBLE_DEVICES=4,5 \
            setsid nohup bash "$WEB_SCRIPT" >> /dev/null 2>&1 &
        restarted=1
    fi
    if [ $doneC -eq 0 ] && ! is_alive "$EXP_C"; then
        [ $restarted -eq 1 ] && { log "stagger 100s before laneC restart"; sleep 100; }
        log "$EXP_C not alive, restarting"
        cd "$BASE_DIR" || exit 1
        SEED=2 CUDA_VISIBLE_DEVICES=6,7 \
            setsid nohup bash "$WEB_SCRIPT" >> /dev/null 2>&1 &
    fi
    sleep 60
done
