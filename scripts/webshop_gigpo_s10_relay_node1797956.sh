#!/usr/bin/env bash
# WebShop GiGPO seed10 接力看门狗（ai-1797956，2026-10-10）
# 触发：等 seed2 收官（global_step_200）释放 GPU 6,7 -> 启动 seed10（官方 seeds {0,2,10} 最后一员）
# 行为：死亡 -> 错峰 100s 重启 -> 完成 global_step_200 -> 退出
# 完成判据必须查目录（教训 §29.28：不看进程退出）

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen1.5b_2gpu_official/train_gigpo_official_node1613826.sh
LOG=$BASE_DIR/verl_log/webshop_gigpo_s10_relay_node1797956.log
TOTAL_STEPS=200
EXP_S2=gigpo_official_webshop_1.5b_2gpu_L-12_seed2_node1613826
CKPT_S2=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$EXP_S2
EXP=gigpo_official_webshop_1.5b_2gpu_L-12_seed10_node1613826
CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$EXP

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }

log "seed10 relay started, waiting for seed2 ($EXP_S2) to finish"
# 阶段1：等 seed2 完成（最多等 6h）
for i in $(seq 1 360); do
    [ -d "$CKPT_S2/global_step_$TOTAL_STEPS" ] && break
    # 异常保护：seed2 既没完成也没进程且 s10 尚未启动超过 30 分钟 -> 直接上卡（GPU 已空）
    if ! is_alive "$EXP_S2" && ! is_alive "$EXP" && [ $i -gt 30 ]; then
        log "seed2 not alive and no global_step_200 after 30min; GPU 6,7 assumed free, starting seed10 early"
        break
    fi
    sleep 60
done
log "phase 1 done, launching seed10 on GPU 6,7"
cd "$BASE_DIR" || exit 1
SEED=10 CUDA_VISIBLE_DEVICES=6,7 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
log "seed10 launched pid=$!"

# 阶段2：seed10 看门狗
while true; do
    [ -d "$CKPT/global_step_$TOTAL_STEPS" ] && { log "seed10 DONE, relay exit"; exit 0; }
    if ! is_alive "$EXP"; then
        log "seed10 not alive, restarting in 100s"
        sleep 100
        cd "$BASE_DIR" || exit 1
        SEED=10 CUDA_VISIBLE_DEVICES=6,7 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
        log "seed10 relaunched pid=$!"
    fi
    sleep 60
done
