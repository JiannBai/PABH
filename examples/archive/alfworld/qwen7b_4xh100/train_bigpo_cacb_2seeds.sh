#!/usr/bin/env bash
# BiGPO + CACB multi-seed wrapper.
#
# Runs the same CACB training recipe as run_alfworld_bigpo_cacb_qwen7b_4xh100.sh
# twice with different env.seed values, sequentially on the same 4xH100 box.
# Each seed gets isolated swanlab/bigpo-diag log dirs and a distinct
# experiment_name so all three runs (the existing seed=0, plus seeds=1,2)
# are easy to aggregate later for mean±std reporting.
#
# Override the seed list with:
#   SEEDS="3 4" bash run_alfworld_bigpo_cacb_qwen7b_4xh100_2seeds.sh
#
# Anything you pass after `--` is forwarded verbatim to main_ppo (handy for
# overriding total_epochs, etc.).

set -euo pipefail

ENGINE=${ENGINE:-vllm}
SEEDS=${SEEDS:-"1 2"}

# ---- User paths (mirror the single-seed script) -------------------------
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}

# ---- Offline tracking ---------------------------------------------------
export SWANLAB_MODE=local
export WANDB_MODE=offline
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# ---- Recipe knobs (kept identical to the single-seed CACB run) ---------
train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"
bisim_eps=0.10
bisim_embedder=actor_hidden
bisim_layer=-8
cacb_enabled=True
cacb_n_action_tokens=8
num_cpus_per_env_worker=0.5

# Forward extra args after `--` to main_ppo.
EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi

# Stage data once — same parquet for every seed.
python3 -m examples.data_preprocess.prepare_offline \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

run_one_seed() {
    local seed=$1
    local tag="seed${seed}"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local log_dir=${LOG_DIR:-$PWD/train_logs}
    mkdir -p "$log_dir"
    local log_file="$log_dir/train_bigpo_cacb_${tag}_${ts}.log"

    # Per-seed isolation: separate swanlab project dir (avoids SQLite races)
    # and separate bigpo-diag dir (so cluster_hist.csv etc. don't get mixed).
    export SWANLAB_LOG_DIR="$PWD/swanlog_bigpo_cacb_${tag}"
    export BIGPO_LOG_DIR="$log_dir/bigpo_cacb_diag_${tag}"
    mkdir -p "$SWANLAB_LOG_DIR" "$BIGPO_LOG_DIR"

    echo "============================================================"
    echo "[$(date)] CACB run seed=${seed}"
    echo "  log         : $log_file"
    echo "  swanlab dir : $SWANLAB_LOG_DIR"
    echo "  diag dir    : $BIGPO_LOG_DIR"
    echo "============================================================"

    # Tee this seed's stdout/stderr into its own file.
    {
        set -x
        python3 -m verl.trainer.main_ppo \
            hydra.run.dir="$log_dir/hydra_${tag}_${ts}" \
            algorithm.adv_estimator=gigpo \
            algorithm.gigpo.bisim_grouping=True \
            algorithm.gigpo.bisim_eps=$bisim_eps \
            algorithm.gigpo.bisim_embedder=$bisim_embedder \
            algorithm.gigpo.bisim_layer=$bisim_layer \
            +algorithm.gigpo.cacb_enabled=$cacb_enabled \
            +algorithm.gigpo.cacb_n_action_tokens=$cacb_n_action_tokens \
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
            env.seed=${seed} \
            env.max_steps=50 \
            env.rollout.n=$group_size \
            env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
            trainer.critic_warmup=0 \
            trainer.logger=['console','swanlab'] \
            trainer.project_name='verl_agent_alfworld_bigpo' \
            trainer.experiment_name="bigpo_cacb_qwen2.5_7b_4xh100_actorhidden_L-8_eps0.10_${tag}" \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=1 \
            trainer.save_freq=20 \
            trainer.test_freq=5 \
            trainer.total_epochs=150 \
            trainer.val_before_train=True \
            "${EXTRA_ARGS[@]}"
        set +x
    } 2>&1 | tee -a "$log_file"
}

for seed in $SEEDS; do
    run_one_seed "$seed"
done

echo "[$(date)] all CACB seeds done: $SEEDS"
