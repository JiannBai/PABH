#!/usr/bin/env bash
# WebShop GRPO 阶梯接力器（ai-1797956，2026-10-12）
# 触发：方法 seed4 收官（global_step_200）释放 GPU 6,7
#   -> WebShop GRPO s0（episode 级地板，官方口径 2 卡）-> 完 -> s1 -> 完 -> s2
# 判活宽限 3 分钟（fork 空窗教训）；完成 = global_step_200 目录判据；死亡错峰 100s 重启

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen1.5b_2gpu_official/train_grpo_official_node1613826.sh
LOG=$BASE_DIR/verl_log/webshop_grpo_relay_node1797956.log
TOTAL_STEPS=200

GATE=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.06_qstyle_g_prmanchor_seed4_node1613826
GATE_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$GATE

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$1/global_step_$TOTAL_STEPS" ]; }

log "webshop GRPO relay started, waiting for method seed4 to finish"
for i in $(seq 1 1440); do
    is_done "$GATE" && break
    sleep 60
done
log "gate open (seed4 done or 24h timeout), launching GRPO seed0 on GPU 6,7"

launch() {
    cd "$BASE_DIR" || exit 1
    SEED=$1 CUDA_VISIBLE_DEVICES=6,7 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
}

SEQ=(0 1 2)
for s in "${SEQ[@]}"; do
    EXP=grpo_official_webshop_1.5b_2gpu_seed${s}_node1613826
    if is_done "$EXP"; then log "seed$s already done, skip"; continue; fi
    launch $s; t=$(date +%s)
    log "seed$s launched pid=$!"
    while true; do
        now=$(date +%s)
        is_done "$EXP" && { log "seed$s DONE"; break; }
        if [ $((now - t)) -gt 180 ] && ! is_alive "$EXP"; then
            log "seed$s not alive, restarting in 100s"
            sleep 100; launch $s; t=$(date +%s); log "seed$s relaunched pid=$!"
        fi
        sleep 60
    done
    # seed 间错峰
    sleep 100
done
log "all GRPO seeds DONE, relay exit"
