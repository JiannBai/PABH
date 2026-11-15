#!/usr/bin/env bash
# BiGPO + CACB Q-style on HopQA / Search-R1: Qwen2.5-7B-Instruct, 8 GPUs.
#
# Matches the hyperparameters of the original run_search.sh GiGPO baseline:
#   * 8 GPUs, TP=1 (full DP)
#   * train_batch=256, ppo_mini=512, ppo_micro=16/gpu
#   * lr=1e-6, kl_loss_coef=0.001, gpu_mem_util=0.6
#   * env.max_steps=4, env.history_length=4
# Only the algorithm differs:
#   - run_search.sh:  GiGPO + similarity grouping (sim_thresh=0.9)
#   - this script:    GiGPO + bisim grouping + CACB(q-style, tag_verb on search/answer)
#
# Prereq: retrieval server on $SEARCH_URL (default 127.0.0.1:8000).
#   ./examples/search/retriever/retrieval_launch.sh

ENGINE=${1:-vllm}
SEEDS=${SEEDS:-"0 1"}
ulimit -u 65536

LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
SEARCHR1_ROOT=${SEARCHR1_ROOT:-$HOME/data/searchR1}
TRAIN_DATA=${TRAIN_DATA:-$SEARCHR1_ROOT/searchR1_processed_direct/train.parquet}
VAL_DATA=${VAL_DATA:-$SEARCHR1_ROOT/searchR1_processed_direct/test.parquet}
SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:8000/retrieve}

if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" ]]; then
    echo "ERROR: train/val parquet not found under $SEARCHR1_ROOT/searchR1_processed_direct/" >&2
    echo "  TRAIN_DATA=$TRAIN_DATA" >&2
    echo "  VAL_DATA  =$VAL_DATA" >&2
    exit 1
fi

export SWANLAB_MODE=local
export WANDB_MODE=offline
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# ---- Match run_search.sh shape ----
train_data_size=256
val_data_size=512
group_size=5
mode="mean_std_norm"
max_steps=4

# ---- CACB-specific ----
bisim_eps=0.10
bisim_embedder=actor_hidden
bisim_layer=-8                       # 7B has 28 layers; -8 = layer 20
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=tag_verb        # 2(+1) clusters: search / answer / none
cacb_action_tag_name="search,answer"
cacb_estimator=q_style

for SEED in $SEEDS; do
    TS=$(date +%Y%m%d_%H%M%S)
    TAG="search_bigpo_cacb_qstyle_7b_8gpu_seed${SEED}"
    LOG_FILE="$LOG_DIR/train_${TAG}_${TS}.log"
    SWANLOG_DIR="$PWD/swanlog_${TAG}"
    DIAG_DIR="$PWD/train_logs/${TAG}_diag"

    echo "============================================================"
    echo "[$(date)] HopQA BiGPO+CACB(q-style, tag_verb) 7B 8GPU seed=${SEED} — $LOG_FILE"
    echo "============================================================"

    (
        set -x
        SWANLAB_LOG_DIR=$SWANLOG_DIR BIGPO_LOG_DIR=$DIAG_DIR CACB_VERBOSE=${CACB_VERBOSE:-1} \
        python3 -m verl.trainer.main_ppo \
            hydra.run.dir=$LOG_DIR/hydra_seed${SEED}_$TS \
            algorithm.adv_estimator=gigpo \
            algorithm.gigpo.bisim_grouping=True \
            algorithm.gigpo.bisim_eps=$bisim_eps \
            algorithm.gigpo.bisim_embedder=$bisim_embedder \
            algorithm.gigpo.bisim_layer=$bisim_layer \
            algorithm.gigpo.cacb_enabled=$cacb_enabled \
            algorithm.gigpo.cacb_n_action_tokens=$cacb_n_action_tokens \
            algorithm.gigpo.cacb_action_key_mode=$cacb_action_key_mode \
            +algorithm.gigpo.cacb_action_tag_name="'$cacb_action_tag_name'" \
            algorithm.gigpo.cacb_estimator=$cacb_estimator \
            algorithm.gigpo.step_advantage_w=1.0 \
            algorithm.gigpo.mode=$mode \
            algorithm.use_kl_in_reward=False \
            algorithm.gamma=0.95 \
            data.train_files=$TRAIN_DATA \
            data.val_files=$VAL_DATA \
            data.train_batch_size=$train_data_size \
            data.val_batch_size=$val_data_size \
            data.max_prompt_length=4096 \
            data.max_response_length=512 \
            data.filter_overlong_prompts=True \
            data.truncation='left' \
            data.return_raw_chat=True \
            actor_rollout_ref.model.path=$MODEL_PATH \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
            actor_rollout_ref.actor.ppo_mini_batch_size=512 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.actor.use_invalid_action_penalty=True \
            actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.name=$ENGINE \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
            actor_rollout_ref.rollout.enable_chunked_prefill=False \
            actor_rollout_ref.rollout.enforce_eager=False \
            actor_rollout_ref.rollout.free_cache_engine=False \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
            actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            env.env_name=search \
            env.seed=$SEED \
            env.max_steps=$max_steps \
            env.rollout.n=$group_size \
            env.history_length=4 \
            env.search.search_url=$SEARCH_URL \
            trainer.critic_warmup=0 \
            trainer.logger=['console','swanlab'] \
            trainer.project_name='verl_agent_search_bigpo' \
            trainer.experiment_name="bigpo_cacb_qwen2.5_7b_8gpu_search_actorhidden_L${bisim_layer}_eps${bisim_eps}_qstyle_tagverb_seed${SEED}" \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=1 \
            trainer.save_freq=50 \
            trainer.test_freq=50 \
            trainer.total_epochs=1 \
            trainer.val_before_train=False
    ) 2>&1 | tee -a "$LOG_FILE"

    echo "[$(date)] seed=${SEED} done."
done

echo "[$(date)] All seeds finished: $SEEDS"
