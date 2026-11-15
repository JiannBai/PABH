#!/usr/bin/env bash
# WebShop 7B 双组 · 本机（ai-1613826）· MODE 环境变量选 baseline|bisimhead
# baseline = BiPACE-Q 官方口径（仓脚本 train_bipace.sh 逐项一致：eps0.08/L-8/action_tag+q_style/G=12/horizon15/150步/TP2）
# bisimhead = 上述 + 我们的方法三件套（bisim_head + PRM锚 + g_text，1.5B 同款超参）
# 用法：MODE=baseline SEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_webshop7b_node1613826.sh

ENGINE=${1:-vllm}
MODE=${MODE:-baseline}
SEED=${SEED:-0}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace-webshop
unset TRITON_PTXAS_PATH
export JAVA_HOME=${JAVA_HOME:-/root/miniconda3/envs/bipace-webshop}

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
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/bipace_7b_webshop_${MODE}_eps004_seed${SEED}_node1613826_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] WebShop 7B ${MODE} SEED${SEED} NODE1613826 (4gpu TP2, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) started"

set -x

export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=128
group_size=12                 # 官方 WebShop G=12
mode="mean_std_norm"
bisim_eps=0.04                # 论文 Table I.1 口径对照版（仓脚本是 0.08，测哪个是 79.7 的真实配置）
bisim_embedder=actor_hidden
bisim_layer=-8
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=action_tag
cacb_estimator=q_style
total_train_steps=150         # 官方 7B WebShop 预算

# ---- bisimhead 模式追加（唯一算法差）----
if [ "$MODE" = "bisimhead" ]; then
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
else
    EXTRA_HEAD_ARGS=""
fi

project_name="qwen2.5_7b_webshop_bipace_node1613826"
experiment_name="${MODE}_webshop_7b_4gpu_L-8_eps0.04_qstyle_actiontag_seed${SEED}_node1613826"
CKPT_DIR=${BASE_DIR}/verl_checkpoints/${project_name}/${experiment_name}

python3 -m examples.data_preprocess.prepare_offline \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2))

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$LOG_DIR/hydra_$TS \
    algorithm.adv_estimator=gigpo \
    algorithm.gigpo.bisim_grouping=True \
    algorithm.gigpo.bisim_eps=$bisim_eps \
    algorithm.gigpo.bisim_embedder=$bisim_embedder \
    algorithm.gigpo.bisim_layer=$bisim_layer \
    algorithm.gigpo.cacb_enabled=$cacb_enabled \
    algorithm.gigpo.cacb_n_action_tokens=$cacb_n_action_tokens \
    algorithm.gigpo.cacb_action_key_mode=$cacb_action_key_mode \
    algorithm.gigpo.cacb_estimator=$cacb_estimator \
    ${EXTRA_HEAD_ARGS} \
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
    env.env_name=Webshop \
    env.seed=$SEED \
    env.max_steps=15 \
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
    trainer.test_freq=5 \
    trainer.total_epochs=$total_train_steps \
    trainer.total_training_steps=$total_train_steps \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${CKPT_DIR}/rollouts" \
    trainer.validation_data_dir="${CKPT_DIR}/val_rollouts" \
    trainer.val_before_train=True $@
