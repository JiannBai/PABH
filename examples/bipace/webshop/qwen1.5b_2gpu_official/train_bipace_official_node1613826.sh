#!/usr/bin/env bash
# BiPACE-Q 官方论文口径严格复现 · WebShop 1.5B · 2卡 · 本机（node1613826）
# 口径来源 = 论文 Table I.1 WebShop列 + 官方 repo ablation/eps0.05_2seeds.sh（实为论文主表配置）:
#   G=12, horizon=15, eps=0.06, layer=-12, CACB=action_tag+q_style, TP=1, 2×H100, 200 steps
#   官方 seeds = {0, 2, 10}（主脚本 seed0 + 2seeds 脚本 seed 2/10）
# 与官方脚本的唯一非算法差异（已在文档披露）:
#   ① WebshopWorker 分片架构(OOM修复: JVM/Lucene索引每actor一份, 行为等价已验证)
#   ② trainer.logger=console only(官方含swanlab, 纯日志基础设施, 不影响训练)
#   ③ 硬件 H20×2 vs H100×2

ENGINE=${1:-vllm}
SEED=${SEED:-0}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace-webshop
unset TRITON_PTXAS_PATH
export JAVA_HOME=${JAVA_HOME:-/root/miniconda3/envs/bipace-webshop}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
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
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/bipace_1.5b_webshop_official_baseline_seed${SEED}_node1613826_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] BiPACE-Q OFFICIAL 1.5B WebShop seed=${SEED} POD1491319 (2gpu, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) started"

set -x

num_cpus_per_env_worker=0.1   # 官方值（分片后仅~10个actor, 无fork风暴风险）
train_data_size=16
val_data_size=128
group_size=12                 # ← 论文 Table I.1 WebShop: G=12（非旧版脚本的8）
mode="mean_std_norm"
bisim_eps=0.06                # ← 论文 Table I.1 WebShop 1.5B: ε=0.06
bisim_embedder=actor_hidden   # 官方基线：无head
bisim_layer=-12               # 论文 1.5B 设定
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=action_tag
cacb_estimator=q_style

project_name="qwen2.5_1.5b_webshop_bipace_official_node1613826"
experiment_name="bipaceq_official_webshop_1.5b_2gpu_L-12_eps0.06_qstyle_actiontag_seed${SEED}_node1613826"
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
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${CKPT_DIR}/rollouts" \
    trainer.validation_data_dir="${CKPT_DIR}/val_rollouts" \
    trainer.val_before_train=True $@
