#!/usr/bin/env bash
# BiPACE-Q + 进度感知 bisim_head（g_set 进 anchor + prm 锚（|r_i-r_j|））· TextCraft 7B · 本机（pod1491319）
#
# 目的：把双机 12 次 TextCraft 1.5B 跑（EXPERIMENTS_LOG_THISNODE.md §十~十二）里识别出的
# "目前最优 head 超参组合"（lr=1e-4 慢EMA τ=0.005 + resample_pairs + min_dt=3 + updates_per_sample=3）
# 搬到 7B 上验证：官方数据 7B TextCraft run 级 std 只有 ±2.4pp（vs 1.5B 的 ±7-8pp），
# 3-8pp 的处理效应在 7B 上有更大机会被检测出来。
#
# 同时把 val_data_size 从官方 64 扩到 320（TextCraft val 池只有 79 个唯一 goal，
# 320 会循环 ~4 圈，每圈独立 temperature=0.4 采样，二项标准差从 ~0.06 降到 ~0.027，
# 参考 EXPERIMENTS_LOG_THISNODE.md §11.3-a 的方法论建议）。
#
# 与对照脚本（..._g_prmanchor_val320_pod1491319.sh）唯一变量 = MICo 锚（d_g vs prm）。
# 7B 官方预算是 50 步（收敛快，见 README/EXPERIMENTS_LOG_THISNODE §12.5）。

ENGINE=${1:-vllm}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 见 EXPERIMENTS_LOG.md：这些 env_vars 提前 export，绕开 Ray runtime_env_agent
# 在网络文件系统上冷启动导入超过 30s 硬编码超时的问题。
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/bipace_7b_textcraft_bisimhead_g_prmanchor_val320_pod1491319_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] BiPACE 7B TextCraft BISIM-HEAD (g_in_anchor + prm anchor, val=320) POD1491319 (4gpu TP=2, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) started"

set -x

num_cpus_per_env_worker=0.5
train_data_size=16
val_data_size=320        # 官方默认64 → 扩到320（TextCraft val池79个goal，循环~4圈）
group_size=12
mode="mean_std_norm"
max_steps=70
total_train_steps=50     # 官方 7B TextCraft 预算（收敛快，见 README）
bisim_eps=0.10
bisim_embedder=bisim_head
bisim_layer=-8            # 7B=28层，对齐官方 alfworld 7B BiGPO 的取法
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=action_tag
cacb_estimator=q_style

# ---- bisim head 超参：双机12次跑里目前唯一跑赢baseline的组合 ----
bisim_head_dim=256
bisim_head_gamma=0.95            # 与 PPO algorithm.gamma 一致
bisim_head_lr=1e-4               # 慢 EMA 组合
bisim_head_tau=0.005             # target head 慢速更新（准冻结）
bisim_head_n_pairs=256
bisim_head_train_steps=10        # 每 PPO step 采样轮数
bisim_head_updates_per_sample=3  # 每轮同批 pairs 做 3 次梯度更新 → 30 次/ppo step
bisim_head_resample_pairs=True   # 每轮独立重采样，覆盖率 4.7%→~38%
bisim_head_min_dt=3               # 同轨迹pair最小步距，避免退化目标

project_name="qwen2.5_7b_textcraft_bipace_pod1491319"
experiment_name="bisimhead_textcraft_7b_4gpu_L-8_eps0.10_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_val320_pod1491319"
CKPT_DIR=${BASE_DIR}/verl_checkpoints/${project_name}/${experiment_name}

python3 -m examples.data_preprocess.prepare_textcraft \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$LOG_DIR/hydra_$TS \
    algorithm.adv_estimator=gigpo \
    algorithm.gigpo.bisim_grouping=True \
    algorithm.gigpo.bisim_eps=$bisim_eps \
    algorithm.gigpo.bisim_embedder=$bisim_embedder \
    algorithm.gigpo.bisim_layer=$bisim_layer \
    algorithm.gigpo.cacb_enabled=$cacb_enabled \
    algorithm.gigpo.cacb_n_action_tokens=$cacb_n_action_tokens \
    algorithm.gigpo.cacb_action_key_mode=$cacb_action_key_mode \
    algorithm.gigpo.cacb_estimator=$cacb_estimator \
    +algorithm.gigpo.bisim_head_dim=$bisim_head_dim \
    +algorithm.gigpo.bisim_gamma=$bisim_head_gamma \
    +algorithm.gigpo.bisim_head_lr=$bisim_head_lr \
    +algorithm.gigpo.bisim_head_tau=$bisim_head_tau \
    +algorithm.gigpo.bisim_n_pairs=$bisim_head_n_pairs \
    +algorithm.gigpo.bisim_train_steps=$bisim_head_train_steps \
    algorithm.gigpo.bisim_updates_per_sample=$bisim_head_updates_per_sample \
    algorithm.gigpo.bisim_resample_pairs=$bisim_head_resample_pairs \
    algorithm.gigpo.bisim_min_dt=$bisim_head_min_dt \
    algorithm.gigpo.bisim_anchor_mode=prm \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    data.train_files=$HOME/data/verl-agent/textcraft/train.parquet \
    data.val_files=$HOME/data/verl-agent/textcraft/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=3500 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=textcraft \
    env.seed=0 \
    env.max_steps=$max_steps \
    env.history_length=4 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.textcraft.reward_scale=10.0 \
    env.textcraft.max_distractor=10 \
    env.textcraft.show_only_relevant_recipes=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_train_steps \
    trainer.total_training_steps=$total_train_steps \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${CKPT_DIR}/rollouts" \
    trainer.validation_data_dir="${CKPT_DIR}/val_rollouts" \
    trainer.val_before_train=True $@
