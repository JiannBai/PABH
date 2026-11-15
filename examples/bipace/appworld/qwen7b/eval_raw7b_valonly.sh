#!/usr/bin/env bash
# AppWorld · raw Qwen2.5-7B-Instruct · val-only 诊断
# 目的：回答"7B 裸模型是否还需要 SFT 冷启"——不训练，只跑 val_before_train，
# 输出 step0 的 val/success_rate。
# 用法：CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval_raw7b_valonly.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace
unset TRITON_PTXAS_PATH

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0
export APPWORLD_PORT_FILE=${APPWORLD_PORT_FILE:-/tmp/codewiz/appworld_ports.ports}

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$BASE_DIR" || exit 1
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct}

LOG_DIR=$BASE_DIR/verl_log
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/appworld_raw7b_valonly_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] AppWorld raw-7B val-only started"

set -x

python3 -m examples.data_preprocess.prepare_offline \
    --mode text --train_data_size 16 --val_data_size 32

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$LOG_DIR/hydra_$TS \
    algorithm.adv_estimator=gigpo \
    algorithm.gigpo.bisim_grouping=True \
    algorithm.gigpo.bisim_eps=0.10 \
    algorithm.gigpo.bisim_embedder=actor_hidden \
    algorithm.gigpo.bisim_layer=-12 \
    algorithm.gigpo.cacb_enabled=True \
    algorithm.gigpo.cacb_n_action_tokens=8 \
    algorithm.gigpo.cacb_action_key_mode=action_tag \
    algorithm.gigpo.cacb_estimator=q_style \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_std_norm \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
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
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=appworld \
    env.seed=0 \
    env.max_steps=40 \
    env.history_length=2 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.5 \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=qwen2.5_7b_appworld_raw_valonly \
    trainer.experiment_name=raw7b_valonly \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.keep_best_and_latest=False \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.val_only=True \
    trainer.default_local_dir=$BASE_DIR/verl_checkpoints/qwen2.5_7b_appworld_raw_valonly \
    trainer.rollout_data_dir=$BASE_DIR/verl_checkpoints/qwen2.5_7b_appworld_raw_valonly/rollouts \
    trainer.validation_data_dir=$BASE_DIR/verl_checkpoints/qwen2.5_7b_appworld_raw_valonly/val_rollouts \
    trainer.val_before_train=True $@
