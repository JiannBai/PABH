#!/usr/bin/env bash
# Sokoban 1.5B 三模式脚本 · 本机（ai-1753446）
# MODE=baseline      -> BiPACE-Q 官方口径（actor_hidden 聚类）
# MODE=bisimhead     -> + bisim_head（无 PRM 锚 -> head 冻结随机投影消融臂）
# MODE=bisimhead_prm -> + bisim_head + box_prm 锚（完整方法臂：win+1.0/新箱上靶+0.5/无效-0.1）
# 第 5 环境（推荐验证集）：确定性转移、长视野、"boxes-on-target"天然单调进度（g_set/PRM 零成本）。
# 注意：仓库版 sokoban 的 'tiny_rgb_array' 模式实为 ASCII 文本网格（命名误导），天然文本管线。
# 难度旋钮：env.sokoban.num_boxes（yaml 默认 1；信号不足时上调）/ dim_room / search_depth。
# 用法：MODE=baseline SEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sokoban_1.5b_node1613826.sh

ENGINE=${1:-vllm}
MODE=${MODE:-baseline}
SEED=${SEED:-0}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace-webshop
unset TRITON_PTXAS_PATH

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}
export VLLM_ALLREDUCE_USE_SYM_MEM=${VLLM_ALLREDUCE_USE_SYM_MEM:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$BASE_DIR" || exit 1
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sokoban_1.5b_${MODE}_seed${SEED}_node1613826_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] Sokoban 1.5B ${MODE} SEED${SEED} NODE1613826 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, MODEL_PATH=$MODEL_PATH) started"

set -x

num_cpus_per_env_worker=0.5
train_data_size=16
val_data_size=64
group_size=8
mode="mean_std_norm"
bisim_eps=0.10                  # 试水值（与 sciworld 1.5B 一致）
bisim_embedder=actor_hidden
bisim_layer=-12
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=action_tag
cacb_estimator=q_style
total_train_steps=200

if [ "$MODE" = "bisimhead" ] || [ "$MODE" = "bisimhead_prm" ]; then
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
    if [ "$MODE" = "bisimhead_prm" ]; then
        EXTRA_HEAD_ARGS="$EXTRA_HEAD_ARGS env.sokoban.process_reward_mode=box_prm"
    fi
else
    EXTRA_HEAD_ARGS=""
fi

# ---- 分组/估计器配置（按 MODE 分派；gigpo=精确锚官方口径）----
if [ "$MODE" = "gigpo" ]; then
    GROUP_ARGS="algorithm.adv_estimator=gigpo \
        algorithm.gigpo.bisim_grouping=False \
        algorithm.gigpo.enable_similarity=False \
        algorithm.gigpo.step_advantage_w=1.0"
elif [ "$MODE" = "grpo" ]; then
    # GRPO 官方口径：episode 级分组（阶梯地板，无步级信用）
    GROUP_ARGS="algorithm.adv_estimator=grpo"
else
    GROUP_ARGS="algorithm.adv_estimator=gigpo \
        algorithm.gigpo.bisim_grouping=True \
        algorithm.gigpo.bisim_eps=$bisim_eps \
        algorithm.gigpo.bisim_embedder=$bisim_embedder \
        algorithm.gigpo.bisim_layer=$bisim_layer \
        algorithm.gigpo.cacb_enabled=$cacb_enabled \
        algorithm.gigpo.cacb_n_action_tokens=$cacb_n_action_tokens \
        algorithm.gigpo.cacb_action_key_mode=$cacb_action_key_mode \
        algorithm.gigpo.cacb_estimator=$cacb_estimator \
        ${EXTRA_HEAD_ARGS} \
        algorithm.gigpo.step_advantage_w=1.0"
fi

project_name="qwen2.5_1.5b_sokoban_bipace_node1613826"
_TAGPART=""
[ -n "$TAG" ] && _TAGPART="_${TAG}"
experiment_name="${MODE}_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed${SEED}${_TAGPART}_node1613826"
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
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=sokoban \
    env.seed=$SEED \
    env.max_steps=50 \
    env.history_length=2 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=${NGPUS:-4} \
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
