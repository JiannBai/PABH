#!/usr/bin/env bash
# BiPACE embedder ablation: HashNgram / Frozen-ST control vs Actor-Hidden.
#
# This control checks the bisimulation hypothesis: if a
# policy-agnostic similarity grouping (lexical n-gram hash, or a frozen
# sentence encoder) recovers the same gain as Actor-Hidden, the gain
# comes from "any coarsening defeats the singleton tax", not from the
# policy-induced geometry. Everything except the state-side embedder is
# IDENTICAL to the reference 7B Q-style recipe
# (run_alfworld_bigpo_cacb_qstyle_qwen7b_4xh100.sh): same PACE key
# (action_tag), same estimator (q_style), same w/gamma/lr/budget.
#
# Usage (one run per embedder x seed; use 3 seeds each):
#   EMBEDDER=hash_ngram SEED=0 bash examples/bipace/alfworld/qwen7b_4xh100/ablation/embedder.sh
#   EMBEDDER=frozen_st  SEED=0 bash examples/bipace/alfworld/qwen7b_4xh100/ablation/embedder.sh
#
# eps is per-embedder: cosine geometry differs across backends, so the
# Actor-Hidden default (0.10) does NOT transfer. Calibrate first:
#   python3 scripts/calibrate_text_embedder_eps.py --embedder hash_ngram \
#       --qual-dir train_logs/<some>_diag        # real obs if available
# Defaults below come from the synthetic-audit-set calibration
# (median cluster size in [4,8], same criterion as app:eps-tuning):
#   hash_ngram -> 0.25 ; frozen_st -> calibrate, then override EPS.
#
# NOTE: requires the ray_trainer fix that constructs text embedders via
# gigpo.core_bigpo.get_default_embedder — before that fix, frozen_st
# silently fell back to HashNgram.

ENGINE=${1:-vllm}

EMBEDDER=${EMBEDDER:-hash_ngram}        # hash_ngram | frozen_st
SEED=${SEED:-0}
ST_MODEL=${ST_MODEL:-sentence-transformers/all-MiniLM-L6-v2}

case "$EMBEDDER" in
  hash_ngram) EPS=${EPS:-0.25} ;;
  frozen_st)  EPS=${EPS:-0.15} ;;        # placeholder — calibrate & override
  *) echo "EMBEDDER must be hash_ngram or frozen_st (got: $EMBEDDER)"; exit 1 ;;
esac

# Preflight for frozen_st: the encoder must import and load locally
# Set HF_HUB_OFFLINE=1 if the model is pre-downloaded.
if [ "$EMBEDDER" = "frozen_st" ]; then
  python3 - <<EOF || { echo "frozen_st preflight failed: install sentence-transformers and pre-download $ST_MODEL"; exit 1; }
from sentence_transformers import SentenceTransformer
SentenceTransformer("$ST_MODEL")
print("frozen_st preflight OK")
EOF
fi

TAG="embabl_${EMBEDDER}_eps${EPS}_seed${SEED}"

LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${TAG}_${TS}.log"
ln -sf "$LOG_FILE" "$LOG_DIR/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] BiPACE embedder ablation ($TAG) started, logging to $LOG_FILE"

set -x

export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}

export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$PWD/swanlog_${TAG}}
export WANDB_MODE=offline
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# Diagnostics: cluster_hist.csv from this dir feeds tab:singleton-style
# comparisons (sanity-check median size 4-8 after ~5 steps; if the
# partition collapses or shatters, recalibrate EPS).
export BIGPO_LOG_DIR=${BIGPO_LOG_DIR:-$PWD/train_logs/${TAG}_diag}

export CACB_VERBOSE=${CACB_VERBOSE:-0}

num_cpus_per_env_worker=0.5

train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"
cacb_enabled=True
cacb_n_action_tokens=8                 # only used as fallback path
cacb_action_key_mode=action_tag        # same as reference run
cacb_estimator=q_style                 # same as reference run

python3 -m examples.data_preprocess.prepare_offline \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$LOG_DIR/hydra_$TS \
    algorithm.adv_estimator=gigpo \
    algorithm.gigpo.bisim_grouping=True \
    algorithm.gigpo.bisim_eps=$EPS \
    algorithm.gigpo.bisim_embedder=$EMBEDDER \
    +algorithm.gigpo.bisim_st_model=$ST_MODEL \
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
    env.seed=$SEED \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_alfworld_bigpo' \
    trainer.experiment_name="bipace_${TAG}_qstyle_actiontag" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
