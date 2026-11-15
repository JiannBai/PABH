#!/usr/bin/env bash
# BiGPO training: Qwen2.5-7B-Instruct on ALFWorld, 4 x H100-80GB single node.
#
# Key BiGPO knobs (added on top of the standard GiGPO recipe):
#   algorithm.gigpo.bisim_grouping=True   # enable behavioral-fingerprint clustering
#   algorithm.gigpo.bisim_eps=0.20        # cosine-distance threshold for cluster join
#
# Tip: tune bisim_eps per env. Suggested starting points:
#   ALFWorld (text)        : 0.18 - 0.22
#   WebShop  (text)        : 0.20 - 0.25
#   Search-R1 (tool calls) : 0.15 - 0.20
#   Sokoban  (multimodal)  : 0.30 - 0.40 (needs an image-aware embedder)
#
# Layout: 4 GPUs = TP=2 x DP=2 (vllm rollout); FSDP shards across all 4 ranks.

ENGINE=${1:-vllm}
#export VLLM_ATTENTION_BACKEND=XFORMERS

# ---- Logging (tee everything to train.log) ------------------------------
# Both stdout and stderr go to the file AND to the terminal. A timestamped
# copy is also kept so re-runs don't clobber prior logs.
LOG_DIR=${LOG_DIR:-$PWD/train_logs}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_bigpo_${TS}.log"
# Symlink train.log -> latest, for tail -f convenience.
ln -sf "$LOG_FILE" "$LOG_DIR/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] BiGPO run started, logging to $LOG_FILE"
# --------------------------------------------------------------------------

set -x

# ---- User paths ----------------------------------------------------------
# ALFWorld json_2.1.1 dataset root. Must contain json_2.1.1/{train,valid_seen,valid_unseen}.
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
# HuggingFace model id for Qwen2.5-7B-Instruct.
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

# ---- SwanLab offline (network-isolated cluster) -------------------------
# verl reads these in verl/utils/tracking.py — SWANLAB_MODE=local writes
# everything under SWANLAB_LOG_DIR and never contacts swanlab.cn.
export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$PWD/swanlog_bigpo}   # separate logdir → no SQLite corruption when run alongside GiGPO/CACB
# Disable wandb in case any deps still try to phone home.
export WANDB_MODE=offline
# Allow HuggingFace Hub downloads by default; set HF_HUB_OFFLINE=1 if pre-downloaded.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# ---- BiGPO diagnostics (v2 instrumentation) ----------------------------
# When BIGPO_LOG_DIR is set, gigpo.bigpo_logger writes per-step CSVs:
#   cluster_hist.csv     bin counts for the GiGPO-style cluster plot
#   match_distance.csv   Δt distribution within multi-traj clusters
#   cluster_signal.csv   within-cluster return variance (signal quality)
#   qualitative_*.txt    every 25 steps, a sample of 3 clusters' obs
# Unset this var to silence (zero overhead).
export BIGPO_LOG_DIR=${BIGPO_LOG_DIR:-$PWD/train_logs/bigpo_diag}
# --------------------------------------------------------------------------

# Reduce CPU contention with parallel env workers; tune to your box.
num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"
bisim_eps=0.10
bisim_embedder=actor_hidden    # policy-induced hidden state
bisim_layer=-8                 # late-mid layer (layer 20 on Qwen2.5-7B 28L); FFN scan: ~5x lower singleton% than -1

# Stage data (only used as modality + size manifest; no actual labels needed).
# Use the offline variant — upstream prepare.py downloads hiyouga/geometry3k
# just to get a row count, which fails when HF_HUB_OFFLINE=1.
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
    trainer.experiment_name='bigpo_qwen2.5_7b_4xh100_actorhidden_L-8_eps0.10' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
