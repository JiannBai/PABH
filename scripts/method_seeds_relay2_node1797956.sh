#!/usr/bin/env bash
# 方法（bisimhead）WebShop seed 扩池接力器 #2（ai-1797956，2026-10-10）
# 触发：GiGPO seed10 收官（global_step_200）释放 GPU 6,7
#   -> 方法 seed7（GPU 6,7）-> 完成后串行 seed4（GPU 6,7）
# 目的：方法 seed 池 4->{0,1,2,3,4,5,7,10} 共 8 个，扩大"合适种子"采样面
# 判活宽限：新启动 3 分钟内不判死（fork 空窗教训）
# 完成 = global_step_200 目录判据；死亡 -> 错峰 100s 重启

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen1.5b_2gpu_official/train_bisimhead_official_node1613826.sh
LOG=$BASE_DIR/verl_log/method_seeds_relay2_node1797956.log
TOTAL_STEPS=200

GATE_EXP=gigpo_official_webshop_1.5b_2gpu_L-12_seed10_node1613826
GATE_CKPT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$GATE_EXP

run_seed() { # $1=seed
    cd "$BASE_DIR" || exit 1
    SEED=$1 CUDA_VISIBLE_DEVICES=6,7 setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
is_done() { [ -d "$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826/$1/global_step_$TOTAL_STEPS" ]; }

EXP7=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.06_qstyle_g_prmanchor_seed7_node1613826
EXP4=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.06_qstyle_g_prmanchor_seed4_node1613826

log "method seed-pool relay#2 started: waiting for GiGPO seed10 to finish"
for i in $(seq 1 900); do
    is_done "$GATE_EXP" && break
    sleep 60
done
log "gate open, launching method seed7 on GPU 6,7"
run_seed 7; t7=$(date +%s)

while true; do
    now=$(date +%s)
    if ! is_done "$EXP7"; then
        if [ $((now - t7)) -gt 180 ] && ! is_alive "$EXP7"; then
            log "seed7 not alive, restarting in 100s"
            sleep 100; run_seed 7; t7=$(date +%s); log "seed7 relaunched"
        fi
    else
        if ! is_alive "$EXP4" && [ -z "$started4" ]; then
            log "seed7 DONE, launching seed4 on GPU 6,7"
            run_seed 4; t4=$(date +%s); started4=1
        elif [ -n "$started4" ]; then
            if ! is_done "$EXP4" && [ $((now - t4)) -gt 180 ] && ! is_alive "$EXP4"; then
                log "seed4 not alive, restarting in 100s"
                sleep 100; run_seed 4; t4=$(date +%s); log "seed4 relaunched"
            fi
            if is_done "$EXP4"; then
                log "seed7 and seed4 both DONE, relay exit"
                exit 0
            fi
        fi
    fi
    sleep 60
done
