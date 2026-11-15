#!/usr/bin/env bash
# BiPACE-Q + 进度感知 bisim_head（g_set 进 anchor + d_g 结构锚）· ALFWorld 1.5B · 本机（thisnode）
# 对照：官方 BiPACE-Q（train_bigpo_cacb.sh, val 0.935±1.2 官方 / 我们复现 0.898-0.906）
# 单变量差（vs 官方 BiPACE-Q）：bisim_embedder=bisim_head + d_g 锚（g_text 已进 anchor）
# head 超参沿用 TextCraft exp3（lr=1e-3, γ=0.95, τ=0.1）——TextCraft 上唯一中期领先的组合

ENGINE=${1:-vllm}

# conda env（nohup/裸 bash 启动时不继承激活状态，必须显式激活）
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
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
LOG_FILE="$LOG_DIR/bipace_1.5b_alfworld_bisimhead_g_dg_thisnode_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] BiPACE 1.5B ALFWorld BISIM-HEAD (g_in_anchor + d_g anchor) THISNODE (4gpu, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) started"

set -x

export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
num_cpus_per_env_worker=0.5     # 本机适配（0.1 会 Ray fork 卡死）
train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"
bisim_eps=0.05                  # 官方 1.5B ALFWorld 设定
bisim_embedder=bisim_head       # ← 唯一算法差：可训 head（含 d_g 锚）
bisim_layer=-12                 # 官方 1.5B ALFWorld 设定
cacb_enabled=True
cacb_n_action_tokens=8
# CACB 口径 = 官方 1.5B ALFWorld BiPACE-Q 口径（first_n + counterfactual），
# 保证与官方 93.5±1.2 构成严格单变量对照（唯一差 = bisim_head + d_g 锚 + g_text）
cacb_action_key_mode=first_n
cacb_estimator=counterfactual
total_train_steps=200           # 官方 ALFWorld 1.5B 预算

# ---- bisim head 超参（用户定稿 2026-09-07）----
bisim_head_dim=256
bisim_head_gamma=0.95           # 与 PPO gamma 一致
bisim_head_lr=1e-4              # 慢 EMA 组合（他机 TextCraft 0.719 配置）
bisim_head_tau=0.005            # target head 滞后 ~200 次更新（准冻结，bootstrapping 目标稳定）
bisim_head_n_pairs=256
bisim_head_train_steps=10       # 每 PPO step 采样轮数
bisim_head_updates_per_sample=3 # 每轮采样后对同批 pairs 做 3 次梯度更新（mini-epoch）
                                # → head 总更新 = 10 轮 × 3 = 30 次/ppo step
bisim_head_resample_pairs=True  # 每轮独立重采样（覆盖 4.7% → ~38%）
bisim_head_min_dt=3             # 同轨迹 pair 最小步距（跨轨迹不受限）

project_name="qwen2.5_1.5b_alfworld_bipace_thisnode"
experiment_name="bisimhead_alfworld_1.5b_4gpu_L-12_eps0.05_g_dg_lr1e-4_tau0.005_ups3_thisnode"
CKPT_DIR=${BASE_DIR}/verl_checkpoints/${project_name}/${experiment_name}

python3 -m examples.data_preprocess.prepare_offline \
    --mode 'text' \
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
    algorithm.gigpo.bisim_anchor_mode=d_g \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
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
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
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
