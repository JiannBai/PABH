#!/usr/bin/env bash
# 方法（bisimhead）WebShop 新 seed 接力器（ai-1797956，2026-10-10）
# 动机：WebShop seed 方差大（GiGPO s2=0.750 vs s0=0.699，方法 4 seeds 极差 2.3pp）
#       -> 给方法加 seed（已用 {0,2,5,10}，新增 1/3），报全量均值（红线：不挑 seed）
# 触发：Sokoban GiGPO s0 收官（global_step_200）释放 GPU 0-3
#   -> seed1 上 GPU 0,1；错峰 100s 后 seed3 上 GPU 2,3（两个 2 卡 run）
# 判活宽限：新启动 3 分钟内不判死（relay 教训：python fork 前有 ~60s 空窗）
# 完成 = global_step_200 目录判据；死亡 -> 错峰 100s 重启（head 不入 ckpt，resume 重置已登记）

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen1.5b_2gpu_official/train_bisimhead_official_node1613826.sh
LOG=$BASE_DIR/verl_log/method_seeds_relay_node1797956.log
TOTAL_STEPS=200
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct

SOKO_EXP=gigpo_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826
SOKO_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sokoban_bipace_node1613826/$SOKO_EXP

EXP1=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.06_qstyle_g_prmanchor_seed1_node1613826
CKPT1=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$EXP1
EXP3=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.06_qstyle_g_prmanchor_seed3_node1613826
CKPT3=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$EXP3

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }

log "method-seeds relay started: waiting for Sokoban GiGPO s0 to finish"
# 阶段1：等 Sokoban s0 完成（最多 8h）
for i in $(seq 1 480); do
    [ -d "$SOKO_CKPT/global_step_$TOTAL_STEPS" ] && break
    sleep 60
done
log "Sokoban done (or timeout), launching method seed1 on GPU 0,1"
cd "$BASE_DIR" || exit 1
SEED=1 CUDA_VISIBLE_DEVICES=0,1 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
log "seed1 launched pid=$!"
sleep 100
log "launching method seed3 on GPU 2,3"
SEED=3 CUDA_VISIBLE_DEVICES=2,3 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
log "seed3 launched pid=$!"
t_launch1=$(date +%s); t_launch3=$(date +%s)

# 阶段2：双 run 看门狗（3 分钟宽限 + 错峰重启）
while true; do
    now=$(date +%s)
    if [ $((now - t_launch1)) -gt 180 ] && ! is_alive "$EXP1"; then
        log "seed1 not alive (launched $((now-t_launch1))s ago), restarting in 100s"
        sleep 100; cd "$BASE_DIR" || exit 1
        SEED=1 CUDA_VISIBLE_DEVICES=0,1 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
        t_launch1=$(date +%s); log "seed1 relaunched pid=$!"
    fi
    if [ $((now - t_launch3)) -gt 180 ] && ! is_alive "$EXP3"; then
        log "seed3 not alive (launched $((now-t_launch3))s ago), restarting in 100s"
        sleep 100; cd "$BASE_DIR" || exit 1
        SEED=3 CUDA_VISIBLE_DEVICES=2,3 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
        t_launch3=$(date +%s); log "seed3 relaunched pid=$!"
    fi
    if [ -d "$CKPT1/global_step_$TOTAL_STEPS" ] && [ -d "$CKPT3/global_step_$TOTAL_STEPS" ]; then
        log "both method seeds DONE, relay exit"
        exit 0
    fi
    sleep 60
done
