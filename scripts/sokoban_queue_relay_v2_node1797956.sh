#!/usr/bin/env bash
# Sokoban 队列接力器 v2（ai-1797956，2026-10-15）
# 触发：SW gtext s3 完成（global_step_200）释放 GPU 0-3
# 队列：Sokoban 方法 gtext s0/s1/s2 -> GiGPO s1/s2 -> baseline s2（串行）
# 注意：他机正在重建环境，GPU 检测保持 3 分钟确认（防他机回来抢卡时误判）
# 每 run：死亡错峰 100s 重启（宽限 3min）；完成 = global_step_200 目录

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/sokoban/qwen1.5b/train_sokoban_1.5b_node1613826.sh
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct
LOG=$BASE_DIR/verl_log/sokoban_queue_relay_v2_node1797956.log
TOTAL_STEPS=200
PROJ=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sokoban_bipace_node1613826
SW_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826/bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed3_gtext_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$2/global_step_$TOTAL_STEPS" ]; }

gpus_0123_free() {
    local out
    out=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1<=3 && $2>500 {f=1} END {print (f?"busy":"free")}')
    [ "$out" = "free" ]
}

log "sokoban queue relay v2 started, waiting for SW gtext s3 to finish"
for i in $(seq 1 1800); do
    is_done "$SW_CKPT" 2>/dev/null && break
    sleep 60
done
log "s3 done (or 30h timeout), waiting GPU free (3min confirm)"
stable=0
while [ $stable -lt 180 ]; do
    gpus_0123_free && stable=$((stable + 30)) || stable=0
    sleep 30
done
log "GPU 0-3 free, starting sokoban queue"

launch() { # $1=mode $2=seed $3=tag
    cd "$BASE_DIR" || return 1
    env MODE=$1 SEED=$2 ${3:+TAG=$3} MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=0,1,2,3 \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
}

QUEUE="bisimhead_prm:0:gtext bisimhead_prm:1:gtext bisimhead_prm:2:gtext gigpo:1: gigpo:2: baseline:2:"
for item in $QUEUE; do
    IFS=: read MODE_Q SEED_Q TAG_Q <<< "$item"
    EXP="${MODE_Q}_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${SEED_Q}${TAG_Q:+_${TAG_Q}}_node1613826"
    if is_done "$EXP" "$PROJ"; then log "$EXP already done, skip"; continue; fi
    launch $MODE_Q $SEED_Q $TAG_Q; t=$(date +%s)
    log "$EXP launched pid=$!"
    while true; do
        now=$(date +%s)
        is_done "$EXP" "$PROJ" && { log "$EXP DONE"; break; }
        if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
            log "$EXP not alive, restarting in 100s"
            sleep 100; launch $MODE_Q $SEED_Q $TAG_Q; t=$(date +%s); log "$EXP relaunched pid=$!"
        fi
        sleep 60
    done
    sleep 100
done
log "sokoban queue DONE, starting gpu_idle_monitor, relay exit"
cd /mnt/tidalfs-hssh01/dataset/bxd && setsid nohup python3 gpu_idle_monitor.py >> gpu_idle_monitor.out 2>&1 &
exit 0
