#!/usr/bin/env bash
# TextCraft GiGPO baseline (NO BiGPO bisim, NO CACB) on Qwen2.5-7B-Instruct,
# 4 H100 (GPUs 4-7), capped at ~50 training steps.
#
# Differences vs. the 1.5B GiGPO script:
#   * MODEL_PATH points to Qwen2.5-7B-Instruct
#   * tensor_model_parallel_size=2  (7B fits 2-way TP on 80GB H100s)
#   * CUDA_VISIBLE_DEVICES=4,5,6,7  (designed to run in parallel with the 1.5B
#     script on GPUs 0-3)
#
# Usage: ./run_textcraft_gigpo_qwen7b_4xh100_50steps.sh [vllm]
#        SEEDS="0 1 2" ./...

export CUDA_VISIBLE_DEVICES=0,1,2,3

ENGINE=${1:-vllm}
SEEDS=${SEEDS:-"1 2"}
ulimit -u 65536

LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}

export SWANLAB_MODE=local
export WANDB_MODE=offline
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=64
group_size=12
mode="mean_std_norm"
max_steps=70                    # TextCraft horizon cap (per goal episode)
total_train_steps=50           # hard training-step cap for this short run

for SEED in $SEEDS; do
    TS=$(date +%Y%m%d_%H%M%S)
    TAG="textcraft_gigpo_7b_4xh100_50steps_seed${SEED}"
    LOG_FILE="$LOG_DIR/train_${TAG}_${TS}.log"
    SWANLOG_DIR="$PWD/swanlog_${TAG}"
    DIAG_DIR="$PWD/train_logs/${TAG}_diag"

    echo "============================================================"
    echo "[$(date)] TextCraft GiGPO 7B 4xh100 ${total_train_steps}-step seed=${SEED} (GPUs 4-7) — $LOG_FILE"
    echo "============================================================"

    (
        set -x
        SWANLAB_LOG_DIR=$SWANLOG_DIR \
        python3 -m examples.data_preprocess.prepare_textcraft \
            --train_data_size $train_data_size \
            --val_data_size $val_data_size

        SWANLAB_LOG_DIR=$SWANLOG_DIR BIGPO_LOG_DIR=$DIAG_DIR \
        python3 -m verl.trainer.main_ppo \
            hydra.run.dir=$LOG_DIR/hydra_seed${SEED}_$TS \
            algorithm.adv_estimator=gigpo \
            algorithm.gigpo.bisim_grouping=False \
            algorithm.gigpo.cacb_enabled=False \
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
            env.seed=$SEED \
            env.max_steps=$max_steps \
            env.history_length=4 \
            env.rollout.n=$group_size \
            env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
            env.textcraft.reward_scale=10.0 \
            env.textcraft.max_distractor=10 \
            env.textcraft.show_only_relevant_recipes=True \
            trainer.critic_warmup=0 \
            trainer.logger=['console','swanlab'] \
            trainer.project_name='verl_agent_textcraft_gigpo' \
            trainer.experiment_name="gigpo_qwen2.5_7b_4xh100_textcraft_${total_train_steps}steps_seed${SEED}" \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=1 \
            trainer.save_freq=50 \
            trainer.test_freq=10 \
            trainer.total_epochs=$total_train_steps \
            trainer.total_training_steps=$total_train_steps \
            trainer.val_before_train=True
    ) 2>&1 | tee -a "$LOG_FILE"

    echo "[$(date)] seed=${SEED} done."
done

echo "[$(date)] All seeds finished: $SEEDS"
