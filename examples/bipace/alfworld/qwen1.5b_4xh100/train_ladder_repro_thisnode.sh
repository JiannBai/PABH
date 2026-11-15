#!/usr/bin/env bash
# GiGPO / GRPO 官方口径复现（估计器阶梯对照 + 环境差校准）· ALFWorld 1.5B · 本机（ai-1753446）
# =========================================================================
# 双重目的（WWW'27 审稿防御，EXPERIMENTS_LOG §29.20）：
#   ① 同管线估计器阶梯：GRPO(episode级) -> GiGPO(精确锚) -> BiPACE-Q(近似锚) -> PABH(训练度量)
#   ② 环境差校准器：GiGPO 官方 86.7±1.7，若本机复现同样低 ~3pp，则
#     "复现低于官方 = 硬件/库系统差"（而非选择性压低 baseline）实锤
# 用法：EST=gigpo|grpo SEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_ladder_repro_thisnode.sh

ENGINE=${1:-vllm}
EST=${EST:-gigpo}                # gigpo | grpo
SEED=${SEED:-0}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace
unset TRITON_PTXAS_PATH

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}
export VLLM_ALLREDUCE_USE_SYM_MEM=${VLLM_ALLREDUCE_USE_SYM_MEM:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$BASE_DIR" || exit 1
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ladder_${EST}_1.5b_alfworld_seed${SEED}_thisnode_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] LADDER ${EST} 1.5B ALFWorld SEED${SEED} THISNODE (4gpu, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) started"

set -x

export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
num_cpus_per_env_worker=0.5
train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"

if [ "$EST" = "gigpo" ]; then
    EST_ARGS="algorithm.adv_estimator=gigpo \
        algorithm.gigpo.bisim_grouping=False \
        algorithm.gigpo.enable_similarity=False \
        algorithm.gigpo.step_advantage_w=1.0 \
        algorithm.gigpo.mode=$mode"
elif [ "$EST" = "gigpo_gtext" ]; then
    # 【pivot 臂】GiGPO + g_text：精确匹配 key = (g_text + obs)
    # bisim_embedder=bisim_head 仅为打开 env_manager 的 g_text gate（grouping=False 时 head 不构造，ray_trainer.py:1565）
    # 语义：状态重定义作为精确 key，构造性解决 GiGPO 进度混淆（ALFWorld 上 GiGPO 唯一死因）
    EST_ARGS="algorithm.adv_estimator=gigpo \
        algorithm.gigpo.bisim_grouping=False \
        algorithm.gigpo.enable_similarity=False \
        algorithm.gigpo.bisim_embedder=bisim_head \
        algorithm.gigpo.step_advantage_w=1.0 \
        algorithm.gigpo.mode=$mode"
else  # grpo：episode 级估计（无步级信用）
    EST_ARGS="algorithm.adv_estimator=grpo"
fi

project_name="qwen2.5_1.5b_alfworld_ladder_thisnode"
experiment_name="${EST}_alfworld_1.5b_4gpu_seed${SEED}_thisnode"
CKPT_DIR=${BASE_DIR}/verl_checkpoints/${project_name}/${experiment_name}

python3 -m examples.data_preprocess.prepare_offline \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$LOG_DIR/hydra_$TS \
    ${EST_ARGS} \
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
    env.seed=$SEED \
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
    trainer.keep_best_and_latest=True \
    trainer.test_freq=10 \
    trainer.total_epochs=200 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${CKPT_DIR}/rollouts" \
    trainer.validation_data_dir="${CKPT_DIR}/val_rollouts" \
    trainer.val_before_train=True $@
