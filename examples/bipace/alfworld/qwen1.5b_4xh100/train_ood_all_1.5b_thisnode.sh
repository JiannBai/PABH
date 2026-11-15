#!/usr/bin/env bash
# ALFWorld 1.5B · OOD（eval_out_of_distribution）· 四方法统一脚本 · 本机
# MODE=bipaceq | pabh | gigpo | grpo
# 用法：MODE=pabh SEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_ood_all_1.5b_thisnode.sh

ENGINE=${1:-vllm}
MODE=${MODE:-pabh}
SEED=${SEED:-0}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace
unset TRITON_PTXAS_PATH

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$BASE_DIR" || exit 1
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ood_1.5b_alfworld_${MODE}_seed${SEED}_thisnode_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] ALFWorld 1.5B OOD ${MODE} seed=${SEED} THISNODE started"

set -x

export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
num_cpus_per_env_worker=0.5
train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"
total_train_steps=200

project_name="qwen2.5_1.5b_alfworld_bipace_thisnode"

if [ "$MODE" = "gigpo" ]; then
    GROUP_ARGS="algorithm.adv_estimator=gigpo \
        algorithm.gigpo.bisim_grouping=False \
        algorithm.gigpo.enable_similarity=False \
        algorithm.gigpo.step_advantage_w=1.0"
    experiment_name="gigpo_alfworld_ood_1.5b_4gpu_seed${SEED}_thisnode"
elif [ "$MODE" = "grpo" ]; then
    GROUP_ARGS="algorithm.adv_estimator=grpo"
    experiment_name="grpo_alfworld_ood_1.5b_4gpu_seed${SEED}_thisnode"
else
    bisim_eps=0.05
    bisim_layer=-12
    cacb_action_key_mode=first_n
    cacb_estimator=counterfactual
    EXTRA_HEAD_ARGS=""
    if [ "$MODE" = "pabh" ]; then
        bisim_embedder=bisim_head
        EXTRA_HEAD_ARGS="+algorithm.gigpo.bisim_head_dim=256 \
            +algorithm.gigpo.bisim_gamma=0.95 \
            +algorithm.gigpo.bisim_head_lr=1e-4 \
            +algorithm.gigpo.bisim_head_tau=0.005 \
            +algorithm.gigpo.bisim_n_pairs=256 \
            +algorithm.gigpo.bisim_train_steps=10 \
            algorithm.gigpo.bisim_updates_per_sample=3 \
            algorithm.gigpo.bisim_resample_pairs=True \
            algorithm.gigpo.bisim_min_dt=3 \
            algorithm.gigpo.bisim_anchor_mode=prm"
        experiment_name="bisimhead_alfworld_ood_1.5b_4gpu_L-12_eps0.05_g_prmanchor_lr1e-4_tau0.005_ups3_seed${SEED}_thisnode"
    else
        bisim_embedder=actor_hidden
        experiment_name="official_bipaceq_alfworld_ood_1.5b_4gpu_L-12_eps0.05_firstn_seed${SEED}_thisnode"
    fi
    GROUP_ARGS="algorithm.adv_estimator=gigpo \
        algorithm.gigpo.bisim_grouping=True \
        algorithm.gigpo.bisim_eps=$bisim_eps \
        algorithm.gigpo.bisim_embedder=$bisim_embedder \
        algorithm.gigpo.bisim_layer=$bisim_layer \
        algorithm.gigpo.cacb_enabled=True \
        algorithm.gigpo.cacb_n_action_tokens=8 \
        algorithm.gigpo.cacb_action_key_mode=$cacb_action_key_mode \
        algorithm.gigpo.cacb_estimator=$cacb_estimator \
        ${EXTRA_HEAD_ARGS} \
        algorithm.gigpo.step_advantage_w=1.0"
fi

CKPT_DIR=${BASE_DIR}/verl_checkpoints/${project_name}/${experiment_name}

python3 -m examples.data_preprocess.prepare_offline \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$LOG_DIR/hydra_$TS \
    ${GROUP_ARGS} \
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
    env.env_name=alfworld/AlfredTWEnv \
    env.alfworld.eval_dataset=eval_out_of_distribution \
    env.seed=$SEED \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.keep_best_and_latest=True \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_train_steps \
    trainer.total_training_steps=$total_train_steps \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${CKPT_DIR}/rollouts" \
    trainer.validation_data_dir="${CKPT_DIR}/val_rollouts" \
    trainer.val_before_train=True $@
