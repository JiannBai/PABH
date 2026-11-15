#!/usr/bash
# Sokoban 1.5B 三臂补种子守候接力器（ai-1797956，2026-10-12）
# 目标：Sokoban 三臂凑 3v3v3（GiGPO s1/s2 + baseline s2 + 方法 s2，4 个 run 串行）
# 触发条件（防撞）：GPU 0-3 连续 15 分钟空闲（nvidia-smi 显存<500MB 且 util<5%）
#   —— 他机 run 在跑时条件不成立，绝不抢卡；已在 CROSS_MACHINE_STATUS 登记
# 每个 run：死亡 -> 错峰 100s 重启（宽限 3min）；完成 = global_step_200 目录
# 全部完成 -> 拉起 gpu_idle_monitor -> 退出

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/sokoban/qwen1.5b/train_sokoban_1.5b_node1613826.sh
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct
LOG=$BASE_DIR/verl_log/sokoban_seeds_relay_node1797956.log
TOTAL_STEPS=200
PROJ=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sokoban_bipace_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$PROJ/$1/global_step_$TOTAL_STEPS" ]; }

gpus_0123_free() {
    local out
    out=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1<=3 && $2>500 {f=1} END {print (f?"busy":"free")}')
    [ "$out" = "free" ]
}

launch() { # $1=mode $2=seed
    cd "$BASE_DIR" || exit 1
    MODE=$1 SEED=$2 MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=0,1,2,3 \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
}

# 队列：baseline s2 -> gigpo s1 -> gigpo s2 -> bisimhead_prm s2
QUEUE="baseline:2 gigpo:1 gigpo:2 bisimhead_prm:2"

log "sokoban seeds relay started, waiting for GPU 0-3 to be free (15min confirm)"
stable=0
while [ $stable -lt 900 ]; do
    if gpus_0123_free; then
        stable=$((stable + 30))
    else
        stable=0
    fi
    sleep 30
done
log "GPU 0-3 free for 15min, starting queue"

for item in $QUEUE; do
    MODE_Q=${item%%:*}; SEED_Q=${item##*:}
    EXP="${MODE_Q}_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${SEED_Q}_node1613826"
    if is_done "$EXP"; then log "$EXP already done, skip"; continue; fi
    # 上卡前再确认卡还空（他机可能刚上）
    ok=0
    for i in 1 2 3; do
        gpus_0123_free && { ok=1; break; } || { log "GPUs busy again before $EXP, wait 300s"; sleep 300; }
    done
    [ $ok -eq 1 ] || { log "GPUs still busy, defer $EXP (queue paused; rerun relay if needed)"; exit 1; }
    launch $MODE_Q $SEED_Q; t=$(date +%s)
    log "$EXP launched pid=$!"
    while true; do
        now=$(date +%s)
        is_done "$EXP" && { log "$EXP DONE"; break; }
        if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
            log "$EXP not alive, restarting in 100s"
            sleep 100; launch $MODE_Q $SEED_Q; t=$(date +%s); log "$EXP relaunched pid=$!"
        fi
        sleep 60
    done
    sleep 100   # run 间错峰
done
log "all sokoban seed runs DONE, starting gpu_idle_monitor, relay exit"
cd /mnt/tidalfs-hssh01/dataset/bxd && setsid nohup python3 gpu_idle_monitor.py >> gpu_idle_monitor.out 2>&1 &
exit 0
