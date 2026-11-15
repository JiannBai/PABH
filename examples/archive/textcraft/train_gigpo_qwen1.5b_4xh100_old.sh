#!/usr/bin/env bash
# GiGPO baseline: Qwen2.5-1.5B-Instruct on TextCraft, 4xH100.
# Apples-to-apples reference for run_textcraft_bigpo_cacb_qstyle_qwen1.5b_4xh100.sh.
# Only the advantage estimator differs (plain gigpo here, no bisim grouping or CACB).

ENGINE=${1:-vllm}
ulimit -u 65536
export VLLM_ATTENTION_BACKEND=XFORMERS

# ---- Logging ------------------------------------------------------------
LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_textcraft_gigpo_1.5b_4xh100_${TS}.log"
ln -sf "$LOG_FILE" "$LOG_DIR/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] TextCraft GiGPO 1.5B (4xH100) run started, logging to $LOG_FILE"
# --------------------------------------------------------------------------

set -x

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}

export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$PWD/swanlog_textcraft_gigpo_1.5b_4xh100}
export WANDB_MODE=offline
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# TextCraft is pure-Python; 0.3 CPU per actor is plenty.
num_cpus_per_env_worker=0.3

train_data_size=16
val_data_size=64
group_size=8
mode="mean_norm" # "mean_norm" or "mean_std_norm"
max_steps=30    # ★ TextCraft horizon cap (depth-2 fits, depth-4 deliberately under-budgeted)

# Build the TextCraft dataset placeholders (splits.json + parquets).
# Skip on HPC if you've already uploaded ~/data/verl-agent/textcraft/.
python3 -m examples.data_preprocess.prepare_textcraft \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2))

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$LOG_DIR/hydra_$TS \
    algorithm.adv_estimator=gigpo \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=textcraft \
    env.seed=0 \
    env.max_steps=$max_steps \
    env.history_length=2 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.textcraft.reward_scale=10.0 \
    env.textcraft.max_distractor=10 \
    env.textcraft.show_only_relevant_recipes=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_textcraft' \
    trainer.experiment_name='gigpo_qwen2.5_1.5b_4xh100' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
