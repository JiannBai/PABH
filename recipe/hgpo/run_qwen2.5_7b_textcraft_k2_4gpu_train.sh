#!/usr/bin/env bash
# HGPO k=2 on TextCraft with Qwen2.5-7B-Instruct, 4 GPUs.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

ENGINE=${1:-vllm}
SEED=${SEED:-1}
ulimit -u 65536

LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_textcraft_hgpo_k2_7b_4gpu_seed${SEED}_${TS}.log"
ln -sf "$LOG_FILE" "$LOG_DIR/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] TextCraft HGPO k=2 7B 4gpu seed=${SEED} started, logging to $LOG_FILE"

set -x

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}

export SWANLAB_MODE=${SWANLAB_MODE:-local}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$PWD/swanlog_textcraft_hgpo_k2_7b_4gpu_seed${SEED}}
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=64
group_size=12
history_length=2
max_steps=70
total_train_steps=50
mode="mean_std_norm"
weight_type="length"
length_weight_alpha=1.0
base_group=False

python3 -m examples.data_preprocess.prepare_textcraft \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2))

python3 -m recipe.hgpo.main_hgpo \
    hydra.run.dir=$LOG_DIR/hydra_textcraft_hgpo_k${history_length}_7b_seed${SEED}_$TS \
    algorithm.adv_estimator=hgpo \
    algorithm.hgpo.weight_type=$weight_type \
    algorithm.hgpo.mode=$mode \
    algorithm.hgpo.length_weight_alpha=$length_weight_alpha \
    algorithm.hgpo.base_group=$base_group \
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
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
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
    env.seed=$SEED \
    env.max_steps=$max_steps \
    env.history_length=$history_length \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.textcraft.reward_scale=10.0 \
    env.textcraft.max_distractor=10 \
    env.textcraft.show_only_relevant_recipes=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_textcraft_hgpo' \
    trainer.experiment_name="hgpo_k${history_length}_qwen2.5_7b_4gpu_textcraft_${total_train_steps}steps_seed${SEED}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_train_steps \
    trainer.total_training_steps=$total_train_steps \
    trainer.val_before_train=True $@
