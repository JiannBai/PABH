#!/usr/bin/env bash
# 修复版方法臂重跑守候接力器（ai-1797956，2026-10-12）
# 背景：SW/Sokoban 的方法臂（bisimhead_prm）此前缺 g_text 注入（第 1 层缺席，
#       协作者 10/12 09:57 修复 env_manager/sciworld_env_manager/sokoban envs）。
#       旧方法臂数据作废，需以 TAG=gtext 新口径重跑（experiment_name 带 _gtext）。
# 触发：GPU 0-3 连续 15 分钟空闲（他机 run 在跑则不触发，绝不抢卡）
# 队列（全部 4 卡 run，串行）：
#   1-4. SW 1.5B 方法 gtext 版 s0/s1/s2/s3（SFT 起点，~25h each）
#   5-7. Sokoban 1.5B 方法 gtext 版 s0/s1/s2（~16h each）
#   8-10. Sokoban GiGPO s1/s2 + baseline s2（补种子，不受 g_text 影响）
# 每 run：死亡错峰 100s 重启（宽限 3min）；完成 = global_step_200 目录判据
# 完成后拉起 gpu_idle_monitor

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SW_SCRIPT=$BASE_DIR/examples/bipace/sciworld/qwen1.5b/train_sciworld_1.5b_node1613826.sh
SOKO_SCRIPT=$BASE_DIR/examples/bipace/sokoban/qwen1.5b/train_sokoban_1.5b_node1613826.sh
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct
SFT_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492
LOG=$BASE_DIR/verl_log/gtext_rerun_relay_node1797956.log
TOTAL_STEPS=200
SW_CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826
SOKO_CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sokoban_bipace_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$2/global_step_$TOTAL_STEPS" ]; }

gpus_0123_free() {
    local out
    out=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1<=3 && $2>500 {f=1} END {print (f?"busy":"free")}')
    [ "$out" = "free" ]
}

log "gtext-rerun relay started: SW method s0-s3 + Sokoban method s0-s2 (+ Sokoban seeds)"

# ---- 清障：stress 占卡 vs relay 空闲检测死锁修复（2026-10-15）----
# idle monitor 在 GPU 空闲时上 stress（防回收），relay 的空闲检测（显存<500MB）
# 永远等不到 → 死锁。修复：确认他机无活跃 run（verl_log 10 分钟内无他机日志更新）
# 后杀 stress，等显存释放再进入原守候循环。队列跑完重新拉起 monitor。
foreign_quiet() {
    local n
    n=$(find $BASE_DIR/verl_log -maxdepth 1 -name "*.log" -mmin -10 2>/dev/null | \
        grep -vE "gtext_rerun_relay|webshop_grpo_relay|grpo_1.5b_webshop" | wc -l)
    [ "$n" -eq 0 ]
}
if foreign_quiet; then
    log "no foreign activity, clearing stress to unblock idle detection"
    pkill -f "gpu_idle_monitor.py" 2>/dev/null
    pkill -f "gpu_stress.py" 2>/dev/null
    sleep 90   # 显存释放
else
    log "foreign runs active, normal wait mode"
fi

log "waiting for GPU 0-3 free (3min confirm, peer down)"
stable=0
while [ $stable -lt 180 ]; do
    if gpus_0123_free; then stable=$((stable + 30)); else stable=0; fi
    sleep 30
done
log "GPU 0-3 free, starting rerun queue"

run_one() { # $1=script $2=mode $3=seed $4=exp_name $5=ckpt_root $6=model $7=extra_env
    local SCRIPT=$1 MODE=$2 SEED=$3 EXP=$4 CKPT_ROOT=$5 MODEL=$6 EXTRA=$7
    if is_done "$EXP" "$CKPT_ROOT"; then log "$EXP already done, skip"; return 0; fi
    local ok=0
    for i in 1 2 3; do
        gpus_0123_free && { ok=1; break; } || { log "GPUs busy before $EXP, wait 300s"; sleep 300; }
    done
    [ $ok -eq 1 ] || { log "GPUs still busy, defer $EXP"; return 1; }
    cd "$BASE_DIR" || return 1
    env MODE=$MODE SEED=$SEED TAG=gtext MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=0,1,2,3 $EXTRA \
        setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    local t=$(date +%s); log "$EXP launched pid=$!"
    while true; do
        local now=$(date +%s)
        is_done "$EXP" "$CKPT_ROOT" && { log "$EXP DONE"; return 0; }
        if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
            log "$EXP not alive, restarting in 100s"
            sleep 100
            cd "$BASE_DIR" || return 1
            env MODE=$MODE SEED=$SEED TAG=gtext MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=0,1,2,3 $EXTRA \
                setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
            t=$(date +%s); log "$EXP relaunched pid=$!"
        fi
        sleep 60
    done
}

for s in 0 1 2 3; do
    run_one "$SW_SCRIPT" bisimhead_prm $s \
        "bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${s}_gtext_node1613826" \
        "$SW_CKPT_ROOT" "$SFT_CKPT" "" || exit 1
    sleep 100
done
for s in 0 1 2; do
    run_one "$SOKO_SCRIPT" bisimhead_prm $s \
        "bisimhead_prm_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${s}_gtext_node1613826" \
        "$SOKO_CKPT_ROOT" "$MODEL" "" || exit 1
    sleep 100
done
# Sokoban 补种子（非方法臂，无 TAG）
for item in "gigpo:1" "gigpo:2" "baseline:2"; do
    MODE_Q=${item%%:*}; SEED_Q=${item##*:}
    run_one "$SOKO_SCRIPT" $MODE_Q $SEED_Q \
        "${MODE_Q}_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${SEED_Q}_node1613826" \
        "$SOKO_CKPT_ROOT" "$MODEL" "" || exit 1
    sleep 100
done
log "all rerun queue DONE, starting gpu_idle_monitor, relay exit"
cd /mnt/tidalfs-hssh01/dataset/bxd && setsid nohup python3 gpu_idle_monitor.py >> gpu_idle_monitor.out 2>&1 &
exit 0
