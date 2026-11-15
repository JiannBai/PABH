#!/usr/bin/env bash
# BiGPO + CACB Q-style with LINEARLY DECAYING bisim_eps.
#   Qwen2.5-1.5B-Instruct on WebShop, 2 GPUs. Seeds 2 and 10 back-to-back.
#
# Schedule: bisim_eps decays linearly from 0.05 -> 0.02 over global_steps
# 0..100, then stays at 0.02 for the remainder of training.
#
# Rationale: at step 0 the actor_hidden embedder is barely separated, so
# we need a relatively loose eps (0.05) to get cluster sizes that give
# q_style a real same-action peer pool. As training progresses the actor's
# hidden representations sharpen, so the same cosine-distance threshold
# would aggressively over-cluster (mixing reward-heterogeneous trajectories
# into the same bisim group → noisy V_C). Decaying eps to 0.02 by step 100
# tightens clusters as the policy matures.
#
# NOTE: with the new GiGPO floor in build_step_group_bisim (default-on),
# BiGPO is now a strict refinement of GiGPO: same-anchor_obs rows are
# union-merged on top of bisim cosine clustering. So even eps=0 reduces
# to GiGPO (not GRPO). We still bottom out at 0.02 here because going
# all the way to 0 means the cosine-clustering contribution is exactly
# zero, leaving only the string-equality floor — there's no point losing
# the actor_hidden similarity signal entirely.
#
# Usage: ./run_webshop_bigpo_cacb_qstyle_qwen1.5b_2gpu_epsdecay_2seeds.sh [vllm]
#        SEEDS="2 10" ./...

ENGINE=${1:-vllm}
SEEDS=${SEEDS:-"2 10"}
ulimit -u 65536

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

export SWANLAB_MODE=local
export WANDB_MODE=offline
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export CACB_VERBOSE=${CACB_VERBOSE:-1}

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"
# bisim_eps linear schedule: start=0.05, end=0.02, over first 100 global steps.
bisim_eps_start=0.05
bisim_eps_end=0.02
bisim_eps_decay_steps=100
# Static fallback (used before step 0 / if scheduler keys ignored).
bisim_eps=$bisim_eps_start
bisim_embedder=actor_hidden
bisim_layer=-12
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=action_tag
cacb_estimator=q_style

for SEED in $SEEDS; do
    TS=$(date +%Y%m%d_%H%M%S)
    TAG="webshop_bigpo_cacb_qstyle_1.5b_2gpu_epsdecay_0.05to0.02_step100_seed${SEED}"
    LOG_FILE="$LOG_DIR/train_${TAG}_${TS}.log"
    SWANLOG_DIR="$PWD/swanlog_${TAG}"
    DIAG_DIR="$PWD/train_logs/${TAG}_diag"

    echo "============================================================"
    echo "[$(date)] WebShop BiGPO+CACB(q-style, eps 0.05→0.02 over 100 steps, seed=${SEED}) — $LOG_FILE"
    echo "============================================================"

    (
        set -x
        SWANLAB_LOG_DIR=$SWANLOG_DIR \
        python3 -m examples.data_preprocess.prepare_offline \
            --mode 'text' \
            --train_data_size $train_data_size \
            --val_data_size $((val_data_size * 2))

        SWANLAB_LOG_DIR=$SWANLOG_DIR BIGPO_LOG_DIR=$DIAG_DIR \
        python3 -m verl.trainer.main_ppo \
            hydra.run.dir=$LOG_DIR/hydra_seed${SEED}_$TS \
            algorithm.adv_estimator=gigpo \
            algorithm.gigpo.bisim_grouping=True \
            algorithm.gigpo.bisim_eps=$bisim_eps \
            +algorithm.gigpo.bisim_eps_start=$bisim_eps_start \
            +algorithm.gigpo.bisim_eps_end=$bisim_eps_end \
            +algorithm.gigpo.bisim_eps_decay_steps=$bisim_eps_decay_steps \
            algorithm.gigpo.bisim_embedder=$bisim_embedder \
            algorithm.gigpo.bisim_layer=$bisim_layer \
            algorithm.gigpo.cacb_enabled=$cacb_enabled \
            algorithm.gigpo.cacb_n_action_tokens=$cacb_n_action_tokens \
            algorithm.gigpo.cacb_action_key_mode=$cacb_action_key_mode \
            algorithm.gigpo.cacb_estimator=$cacb_estimator \
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
            env.seed=$SEED \
            env.max_steps=15 \
            env.rollout.n=$group_size \
            env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
            trainer.critic_warmup=0 \
            trainer.logger=['console','swanlab'] \
            trainer.project_name='verl_agent_webshop_bigpo' \
            trainer.experiment_name="bigpo_cacb_qwen2.5_1.5b_2gpu_webshop_actorhidden_L-12_epsdecay_0.05to0.02_step100_qstyle_actiontag_seed${SEED}" \
            trainer.n_gpus_per_node=2 \
            trainer.nnodes=1 \
            trainer.save_freq=20 \
            trainer.test_freq=5 \
            trainer.total_epochs=200 \
            trainer.val_before_train=True
    ) 2>&1 | tee -a "$LOG_FILE"

    echo "[$(date)] seed=${SEED} done."
done

echo "[$(date)] All seeds finished: $SEEDS"
