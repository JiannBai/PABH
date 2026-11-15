#!/usr/bin/env bash
# WebShop 1.5B PABH ε 寻优接力 v2（本机 ai-1851164）
# 负责确保 eps0.04 / eps0.08 在跑，并在其中任一完赛后于空出的 2 卡上启动 eps0.10。
# 0.06 已有 4 seeds，不再补。
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
SCRIPT=$BASE_DIR/examples/bipace/webshop/qwen1.5b_2gpu_official/train_webshop15b_pabh_eps_node1613826.sh
LOG=$BASE_DIR/verl_log/webshop15_pabh_sweep_relay_ai1851164.log
CKPT_ROOT=$BASE_DIR/verl_checkpoints/qwen2.5_1.5b_webshop_bipace_official_node1613826
MODEL=/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct

E004_EXP=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.04_qstyle_g_prmanchor_seed0_node1613826
E008_EXP=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.08_qstyle_g_prmanchor_seed0_node1613826
E010_EXP=bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.10_qstyle_g_prmanchor_seed0_node1613826

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }
done1() { [ -d "$CKPT_ROOT/$1/global_step_200" ]; }

mkdir -p "$BASE_DIR/verl_log" /root/ws15_pabh_eps004_s0 /root/ws15_pabh_eps008_s0 /root/ws15_pabh_eps010_s0
log "1.5B PABH sweep relay v2 started"

launch() {  # $1=EXP $2=EPS $3=HOME_DIR $4=GPU_PAIR
  local exp=$1 eps=$2 home_dir=$3 gpus=$4
  if done1 "$exp"; then log "$exp already done"; return 0; fi
  if alive "$exp"; then log "$exp already running"; return 0; fi
  log "launch $exp (eps=$eps, gpus=$gpus)"
  cd "$BASE_DIR" || exit 1
  env HOME="$home_dir" CUDA_VISIBLE_DEVICES="$gpus" MODEL_PATH="$MODEL" EPS="$eps" SEED=0 \
      setsid nohup bash "$SCRIPT" >> /dev/null 2>&1 &
  log "$exp launched pid=$!"
}

# 确保 eps0.04 / eps0.08 两个首轮实验在跑（幂等：已在跑则跳过）。
launch "$E004_EXP" 0.04 /root/ws15_pabh_eps004_s0 0,1
launch "$E008_EXP" 0.08 /root/ws15_pabh_eps008_s0 2,3

e010_started=0
dead004=0
dead008=0
while true; do
  if [ "$e010_started" -eq 1 ] && done1 "$E010_EXP"; then
    log "1.5B PABH eps sweep finished (0.04/0.08/0.10 all reached step 200)"
    break
  fi

  # 只有明确看到 checkpoint 完赛且进程退出，才在空出的 2 卡上启动 eps0.10。
  # 之前的 v1 因用“进程是否存活”判空，会在 prepare_offline 阶段误判为空卡导致提前启动。
  if [ "$e010_started" -eq 0 ]; then
    if done1 "$E004_EXP" && ! alive "$E004_EXP"; then
      launch "$E010_EXP" 0.10 /root/ws15_pabh_eps010_s0 0,1
      e010_started=1
    elif done1 "$E008_EXP" && ! alive "$E008_EXP"; then
      launch "$E010_EXP" 0.10 /root/ws15_pabh_eps010_s0 2,3
      e010_started=1
    fi
  fi

  # 首轮实验死亡但未完赛时，在 eps0.10 尚未占卡的前提下重拉一次（各自原 GPU 对）。
  if ! alive "$E004_EXP" && ! done1 "$E004_EXP"; then
    dead004=$((dead004 + 1))
    if [ "$dead004" -gt 8 ]; then
      log "$E004_EXP dead $dead004 checks, give up"
    elif [ "$e010_started" -eq 0 ]; then
      log "$E004_EXP dead (check #$dead004), relaunch"
      launch "$E004_EXP" 0.04 /root/ws15_pabh_eps004_s0 0,1
    fi
  else
    dead004=0
  fi

  if ! alive "$E008_EXP" && ! done1 "$E008_EXP"; then
    dead008=$((dead008 + 1))
    if [ "$dead008" -gt 8 ]; then
      log "$E008_EXP dead $dead008 checks, give up"
    elif [ "$e010_started" -eq 0 ]; then
      log "$E008_EXP dead (check #$dead008), relaunch"
      launch "$E008_EXP" 0.08 /root/ws15_pabh_eps008_s0 2,3
    fi
  else
    dead008=0
  fi

  sleep 180
done

bash "$BASE_DIR/scripts/idle_monitor_restart.sh" >> "$LOG" 2>&1
