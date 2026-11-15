#!/usr/bin/env bash
# BiPACE Out-of-Distribution (OOD) evaluation, 7B.
# Loads a trained checkpoint and runs val_only on eval_out_of_distribution.
# Usage: bash train_bipace_eval_local.sh [CKPT_STEP]
# Same mechanism as the official setup: config.env.alfworld.eval_dataset=eval_out_of_distribution + trainer.val_only=True + resume_path.

CKPT_STEP=${1:-150}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct}

project_name="qwen2.5_7b_alfworld_bipace"
experiment_name="bipace_7b_4gpu_L-8_eps0.10_qstyle_actiontag"
CKPT_DIR=${BASE_DIR}/verl_checkpoints/${project_name}/${experiment_name}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/bipace_7b_eval_ood_step${CKPT_STEP}_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] BiPACE 7B OOD eval (checkpoint global_step_${CKPT_STEP}) started, logging to $LOG_FILE"

set -x

train_data_size=16
val_data_size=128
group_size=8

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$LOG_DIR/hydra_eval_$TS \
    algorithm.adv_estimator=gigpo \
    algorithm.gigpo.bisim_grouping=True \
    algorithm.gigpo.bisim_eps=0.10 \
    algorithm.gigpo.bisim_embedder=actor_hidden \
    algorithm.gigpo.bisim_layer=-8 \
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
    actor_rollout_ref.rollout.name=vllm \
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
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.5 \
    env.alfworld.eval_dataset=eval_out_of_distribution \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name="${experiment_name}_eval_ood_step${CKPT_STEP}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${CKPT_DIR}/global_step_${CKPT_STEP}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.test_freq=1 $@