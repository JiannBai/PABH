#!/usr/bin/env bash
# 总接力器 v3（ai-1797956，2026-10-15）：GPU 0-3 队列
# 触发：SW gtext s3 完成（global_step_200）释放 GPU 0-3
# 队列（全部 4 卡，串行）：
#   1. ALFWorld OOD 方法 s0（~28h，CVPR 泛化证据，优先级最高）
#   2. ALFWorld OOD baseline s0（~28h，对照臂）
#   3-5. Sokoban 方法 gtext s0/s1/s2（~16h each）
#   6-8. Sokoban GiGPO s1/s2 + baseline s2（补种子）
# 每 run：死亡错峰 100s 重启（宽限 3min）；完成 = global_step_200 目录判据
# 注意：ALFWorld 用 bipace env（py3.12），Sokoban 用 bipace-webshop（py3.10）——脚本自激活

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
ALF_DIR=$BASE_DIR/examples/bipace/alfworld/qwen1.5b_4xh100
SOKO_SCRIPT=$BASE_DIR/examples/bipace/sokoban/qwen1.5b/train_sokoban_1.5b_node1613826.sh
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct
LOG=$BASE_DIR/verl_log/master_queue_relay_v3_node1797956.log
TOTAL_STEPS=200
ALF_PROJ=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_alfworld_bipace_thisnode
SOKO_PROJ=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sokoban_bipace_node1613826
SW_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_sciworld_bipace_node1613826/bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed3_gtext_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$2/global_step_$TOTAL_STEPS" ]; }

gpus_free() {
    local out
    out=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1<=3 && $2>500 {f=1} END {print (f?"busy":"free")}')
    [ "$out" = "free" ]
}

log "master queue relay v3 started, waiting for SW gtext s3"
for i in $(seq 1 1800); do
    is_done "$SW_CKPT" 2>/dev/null && break
    sleep 60
done
log "s3 done (or timeout), waiting GPU free (3min)"
stable=0
while [ $stable -lt 180 ]; do
    gpus_free && stable=$((stable + 30)) || stable=0
    sleep 30
done
log "GPU 0-3 free, starting queue"

run_one() { # $1=script $2=exp_name $3=proj_root $4=mode $5=seed $6=tag $7=env_assign
    local SCRIPT=$1 EXP=$2 PROJ=$3 MODE=$4 SEED=$5 TAG=$6 ENV_A=$7
    if is_done "$EXP" "$PROJ"; then log "$EXP already done, skip"; return 0; fi
    cd "$BASE_DIR" || return 1
    if [ -n "$ENV_A" ]; then
        env $ENV_A CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    else
        env MODE=$MODE SEED=$SEED ${TAG:+TAG=$TAG} MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=0,1,2,3 \
            setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
    fi
    local t=$(date +%s); log "$EXP launched pid=$!"
    while true; do
        local now=$(date +%s)
        is_done "$EXP" "$PROJ" && { log "$EXP DONE"; return 0; }
        if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
            log "$EXP not alive, restarting in 100s"
            sleep 100
            cd "$BASE_DIR" || return 1
            if [ -n "$ENV_A" ]; then
                env $ENV_A CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
            else
                env MODE=$MODE SEED=$SEED ${TAG:+TAG=$TAG} MODEL_PATH=$MODEL CUDA_VISIBLE_DEVICES=0,1,2,3 \
                    setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
            fi
            t=$(date +%s); log "$EXP relaunched pid=$!"
        fi
        sleep 60
    done
}

# 1. ALF OOD 方法 s0（SEED=0）
run_one "$ALF_DIR/train_bisimhead_g_prmanchor_ood_thisnode.sh" \
    "bisimhead_alfworld_ood_1.5b_4gpu_L-12_eps0.05_g_prmanchor_lr1e-4_tau0.005_ups3_seed0_thisnode" \
    "$ALF_PROJ" "" 0 "" "SEED=0" || exit 1
sleep 100
# 2. ALF OOD baseline s0（SEED=0）
run_one "$ALF_DIR/train_official_bipaceq_ood_thisnode.sh" \
    "official_bipaceq_alfworld_ood_1.5b_4gpu_L-12_eps0.05_firstn_seed0_thisnode" \
    "$ALF_PROJ" "" 0 "" "SEED=0" || exit 1
sleep 100
# 3-5. Sokoban 方法 gtext s0/s1/s2
for s in 0 1 2; do
    run_one "$SOKO_SCRIPT" \
        "bisimhead_prm_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${s}_gtext_node1613826" \
        "$SOKO_PROJ" bisimhead_prm $s gtext "" || exit 1
    sleep 100
done
# 6-8. Sokoban GiGPO s1/s2 + baseline s2（非方法臂，无 TAG）
for item in "gigpo:1" "gigpo:2" "baseline:2"; do
    MODE_Q=${item%%:*}; SEED_Q=${item##*:}
    run_one "$SOKO_SCRIPT" \
        "${MODE_Q}_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${SEED_Q}_node1613826" \
        "$SOKO_PROJ" $MODE_Q $SEED_Q "" "" || exit 1
    sleep 100
done
log "all queue DONE, starting gpu_idle_monitor, relay exit"
cd /mnt/tidalfs-hssh01/dataset/bxd && setsid nohup python3 gpu_idle_monitor.py >> gpu_idle_monitor.out 2>&1 &
exit 0
