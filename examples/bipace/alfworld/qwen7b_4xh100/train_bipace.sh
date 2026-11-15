#!/usr/bin/env bash
# BiGPO + CACB Q-style: Direction A from the design discussion.
#
# Replaces the original CACB baseline
#     A_i = R_i - mean(R of different-action peers)
# with the textbook action-conditioned estimator
#     A_i = mean(R of same-action peers in cluster) - mean(R of cluster)
#
# Why: ALFWorld observations are extremely templated (~66 unique strings
# across all training, top 10 cover ~50%), so within-cluster state is
# nearly fixed by construction. The remaining credit-assignment signal
# is purely about ACTION choice. The legacy "diff-only mean" baseline
# is dominated by tiny diff-peer counts (often |diff|=1) → high seed
# variance. Q-style instead pools same-action peers to get a tighter
# Q(s,a) estimate and uses the cluster mean as V(s) — a proper bandit
# advantage when state is fixed.
#
# Pairs with action_tag mode: with first_n=8 the action-key partition
# is dominated by <think> prefix, so q_style would degenerate. action_tag
# extracts the executed action body, giving the estimator a real action
# partition to work on.

ENGINE=${1:-vllm}

LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_bigpo_cacb_qstyle_${TS}.log"
ln -sf "$LOG_FILE" "$LOG_DIR/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] BiGPO+CACB(q-style) run started, logging to $LOG_FILE"

set -x

export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$PWD/swanlog_bigpo_cacb_qstyle}
export WANDB_MODE=offline
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

export BIGPO_LOG_DIR=${BIGPO_LOG_DIR:-$PWD/train_logs/bigpo_cacb_qstyle_diag}

# CACB diagnostic prints — count of multi-action clusters etc.
# Cheap (<1ms/step) but spammy in distributed training; enable on demand.
export CACB_VERBOSE=${CACB_VERBOSE:-0}

num_cpus_per_env_worker=0.5

train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"
bisim_eps=0.10
bisim_embedder=actor_hidden
bisim_layer=-8
cacb_enabled=True
cacb_n_action_tokens=8                 # only used as fallback path
cacb_action_key_mode=action_tag        # reference action partition
cacb_estimator=q_style                 # <-- the new piece

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
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_alfworld_bigpo' \
    trainer.experiment_name='bigpo_cacb_qwen2.5_7b_4xh100_actorhidden_L-8_eps0.10_qstyle_actiontag' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
