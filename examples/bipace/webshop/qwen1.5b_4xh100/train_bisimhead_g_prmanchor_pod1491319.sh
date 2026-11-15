#!/usr/bin/env bash
# BiPACE-Q + 进度感知 bisim_head（g_text进anchor + PRM锚）· WebShop 1.5B · 本机（pod1491319）
# 与 baseline 脚本（train_bipace_baseline_pod1491319.sh）唯一算法差：
#   bisim_embedder=bisim_head + PRM锚 + g_text 进 anchor（WebshopProgressTracker: search/view/buy事件）
# head 超参 = ALFWorld 1.5B 验证过的最优组合（lr1e-4, τ0.005, γ0.95, ups3, resample, min_dt3）
# 预判（EXPERIMENTS_LOG_THISNODE §19.4）：WebShop 的 g_set 与 action_tag key 高度相关 →
# d_g 锚可能失效，PRM 锚预期更优（与 ALFWorld 结论一致）→ 本机只跑 PRM 锚，不跑 d_g。
# 对照目标：官方 BiPACE-Q 1.5B WebShop 71.9±2.1（200步）；本机基线（同脚本同seed）

ENGINE=${1:-vllm}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace-webshop
export JAVA_HOME=${JAVA_HOME:-/root/miniconda3/envs/bipace-webshop}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/bipace_1.5b_webshop_bisimhead_g_prmanchor_pod1491319_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] BiPACE 1.5B WebShop BISIM-HEAD (g_in_anchor + PRM anchor) POD1491319 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) started"

set -x

num_cpus_per_env_worker=0.5
train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"
bisim_eps=0.05                  # 官方 1.5B WebShop 设定（与基线一致）
bisim_embedder=bisim_head       # ← 唯一算法差：可训 head
bisim_layer=-12
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=action_tag
cacb_estimator=q_style
total_train_steps=200           # 官方 1.5B WebShop 预算（与基线一致）

# ---- bisim head 超参：ALFWorld 1.5B 3-seed 0.951 的同款组合 ----
bisim_head_dim=256
bisim_head_gamma=0.95           # 与 PPO gamma 一致
bisim_head_lr=1e-4              # 慢 EMA 组合
bisim_head_tau=0.005            # target head 滞后 ~200 次更新（准冻结）
bisim_head_n_pairs=256
bisim_head_train_steps=10
bisim_head_updates_per_sample=3 # 10轮×3 = 30次梯度更新/ppo step
bisim_head_resample_pairs=True
bisim_head_min_dt=3

project_name="qwen2.5_1.5b_webshop_bipace_pod1491319"
experiment_name="bisimhead_webshop_1.5b_4gpu_L-12_eps0.05_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_pod1491319"
CKPT_DIR=${BASE_DIR}/verl_checkpoints/${project_name}/${experiment_name}

python3 -m examples.data_preprocess.prepare_offline \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2))

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
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_train_steps \
    trainer.total_training_steps=$total_train_steps \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${CKPT_DIR}/rollouts" \
    trainer.validation_data_dir="${CKPT_DIR}/val_rollouts" \
    trainer.val_before_train=True $@
