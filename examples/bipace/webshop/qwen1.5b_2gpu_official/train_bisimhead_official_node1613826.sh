#!/usr/bin/env bash
# 我们的改进方法(进度感知 bisim_head + PRM锚 + g_text进anchor) · WebShop 1.5B · 2卡 · 官方论文口径
# 与 train_bipace_official_node1613826.sh(官方严格基线)唯一算法差 = 我们的方法三件套:
#   ① bisim_embedder=bisim_head (可训MICo头, lr1e-4 τ0.005 γ0.95 ups3 resample min_dt3)
#   ② bisim_anchor_mode=prm (购物事件PRM: +1正确购买/-0.5错买/+0.2新商品页/-0.1坏动作)
#   ③ g_text 进 anchor (env_manager 自动门控: embedder=bisim_head 时注入; 基线不注入)
# 其余一切(2卡/G=12/eps0.06/L-12/horizon15/200步/test_freq5/seeds{0,2,10})与官方基线逐项一致。
# 用法: SEED=0 bash ... ; SEED=2 ...; SEED=10 ...

ENGINE=${1:-vllm}
SEED=${SEED:-0}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace-webshop
unset TRITON_PTXAS_PATH
export JAVA_HOME=${JAVA_HOME:-/root/miniconda3/envs/bipace-webshop}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
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
LOG_FILE="$LOG_DIR/bipace_1.5b_webshop_bisimhead_official_seed${SEED}_node1613826_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] OUR METHOD (bisimhead+PRM+g_text) 1.5B WebShop OFFICIAL-protocol seed=${SEED} NODE1613826 (2gpu, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) started"

set -x

num_cpus_per_env_worker=0.1   # 官方值
train_data_size=16
val_data_size=128
group_size=12                 # 论文 Table I.1 WebShop: G=12
mode="mean_std_norm"
bisim_eps=0.06                # 论文 Table I.1 WebShop 1.5B: ε=0.06
bisim_embedder=bisim_head     # ← 唯一算法差①
bisim_layer=-12
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=action_tag
cacb_estimator=q_style

# ---- bisim head 超参（ALFWorld 1.5B 3-seed 0.951 同款组合）----
bisim_head_dim=256
bisim_head_gamma=0.95
bisim_head_lr=1e-4
bisim_head_tau=0.005
bisim_head_n_pairs=256
bisim_head_train_steps=10
bisim_head_updates_per_sample=3
bisim_head_resample_pairs=True
bisim_head_min_dt=3

project_name="qwen2.5_1.5b_webshop_bipace_official_node1613826"
experiment_name="bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.06_qstyle_g_prmanchor_seed${SEED}_node1613826"
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
    +algorithm.gigpo.bisim_head_dim=$bisim_head_dim \
    +algorithm.gigpo.bisim_gamma=$bisim_head_gamma \
    +algorithm.gigpo.bisim_head_lr=$bisim_head_lr \
    +algorithm.gigpo.bisim_head_tau=$bisim_head_tau \
    +algorithm.gigpo.bisim_n_pairs=$bisim_head_n_pairs \
    +algorithm.gigpo.bisim_train_steps=$bisim_head_train_steps \
    algorithm.gigpo.bisim_updates_per_sample=$bisim_head_updates_per_sample \
    algorithm.gigpo.bisim_resample_pairs=$bisim_head_resample_pairs \
    algorithm.gigpo.bisim_min_dt=$bisim_head_min_dt \
    algorithm.gigpo.bisim_anchor_mode=prm \
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
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${CKPT_DIR}/rollouts" \
    trainer.validation_data_dir="${CKPT_DIR}/val_rollouts" \
    trainer.val_before_train=True $@
