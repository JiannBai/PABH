#!/usr/bin/env bash
# BiGPO + CACB Q-style: Qwen2.5-1.5B-Instruct on WebShop, 2 GPUs (cards 2,3 by default).
# Q-style: A_i = Q_hat(s,a_i) - V_hat(s), paired with action_tag key.

ENGINE=${1:-vllm}
ulimit -u 65536

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_webshop_bigpo_cacb_qstyle_1.5b_2gpu_${TS}.log"
ln -sf "$LOG_FILE" "$LOG_DIR/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] WebShop BiGPO+CACB(q-style) 1.5B (2gpu, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) run started, logging to $LOG_FILE"

set -x

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$PWD/swanlog_webshop_bigpo_cacb_qstyle_1.5b_2gpu}
export WANDB_MODE=offline
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

export BIGPO_LOG_DIR=${BIGPO_LOG_DIR:-$PWD/train_logs/webshop_bigpo_cacb_qstyle_1.5b_2gpu_diag}
export CACB_VERBOSE=${CACB_VERBOSE:-0}

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"
bisim_eps=0.05
bisim_embedder=actor_hidden
bisim_layer=-12
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=action_tag
cacb_estimator=q_style

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
    +algorithm.gigpo.cacb_enabled=$cacb_enabled \
    +algorithm.gigpo.cacb_n_action_tokens=$cacb_n_action_tokens \
    +algorithm.gigpo.cacb_action_key_mode=$cacb_action_key_mode \
    +algorithm.gigpo.cacb_estimator=$cacb_estimator \
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
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_webshop_bigpo' \
    trainer.experiment_name='bigpo_cacb_qwen2.5_1.5b_2gpu_webshop_actorhidden_L-12_eps0.05_qstyle_actiontag' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True $@
