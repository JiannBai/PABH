#!/usr/bin/env bash
# ScienceWorld 1.5B 四模式脚本 · 本机（ai-1613826/ai-1745017/ai-1753446 沿用名）
# MODE=baseline      -> BiPACE-Q 官方配置（actor_hidden 聚类，eps=0.10 与 GiGPO 论文常用口径一致）
# MODE=gigpo         -> GiGPO 官方口径（bisim_grouping=False + enable_similarity=False，精确锚分组）
#                       —— 新环境基线阶梯用（ScienceWorld 无官方 GiGPO 数字，自跑对照）
# MODE=bisimhead     -> + bisim_head（无 process_reward 锚 -> head 不训练，冻结随机投影消融臂）
# MODE=bisimhead_prm -> + bisim_head + score_delta PRM 锚（完整方法臂：head 在线 MICo 训练）
# 参照：chaochaoxx/scienceworld-verl-agent-posttrain 的 GiGPO 7B 基线（86.5% no-hints eval）
# 注意：ScienceWorld 暂无 g_set/g_text 进度追踪（三层栈的第 1 层缺席），
#       本环境的"方法臂"= 第 2 层度量训练 + 第 3 层消费（详见 EXPERIMENTS_LOG §29.5）。
#
# 用法：MODE=baseline SEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_sciworld_1.5b_node1613826.sh

ENGINE=${1:-vllm}
MODE=${MODE:-baseline}
SEED=${SEED:-0}
START=${START:-sft}             # raw|sft —— 进实验名，防同名 ckpt 覆盖（10/4 事故根因）；
TAG=${TAG:-}                     # 附加实验名后缀（gtext 代次标记等；空=默认名）
EPS=${EPS:-0.10}                 # 聚类阈值（我们的方法需对 eps 寻优）
STEPS=${STEPS:-200}              # 训练步数（eps 寻优短跑可设 100）
                                # 默认 sft 保持与历史 run（seed0/1 已完成的 SFT 起点系列）同名兼容

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace
unset TRITON_PTXAS_PATH
export JAVA_HOME=${JAVA_HOME:-/opt/java}

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
# 关键修复：容器底噪 TRITON_PTXAS_PATH 指向系统 CUDA 13.0 ptxas，
# triton 3.2 无法解析其版本号导致 vllm 编译崩溃（2026-09-28 事故）；改用 triton 自带 ptxas(12.4)。
unset TRITON_PTXAS_PATH

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
# 关键修复：必须 cd 到仓库根——env.sciworld.split_path 等配置是相对路径，
# 接力脚本从其他目录（如 /tmp/codewiz）启动时相对路径解析失败（2026-09-28 事故）。
cd "$BASE_DIR" || exit 1
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sciworld_1.5b_${MODE}_seed${SEED}_node1613826_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] ScienceWorld 1.5B ${MODE} SEED${SEED} NODE1613826 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) started"

set -x

num_cpus_per_env_worker=0.5     # JVM 模拟器较重
train_data_size=16
val_data_size=64                # test split 200 中按 env 采样
group_size=8
mode="mean_std_norm"
bisim_eps=$EPS                   # SciWorld 反馈模板同 ALFWorld；PABH 需按 EPS 寻优
bisim_embedder=actor_hidden
bisim_layer=-12
cacb_enabled=True
cacb_n_action_tokens=8
cacb_action_key_mode=action_tag
cacb_estimator=q_style
total_train_steps=$STEPS

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
        # 完整方法臂：env 暴露 score_delta 规则 PRM -> head 实际在线训练
        # （bisimhead 臂无此键 -> head 冻结随机投影，为"未训练投影"消融）
        EXTRA_HEAD_ARGS="$EXTRA_HEAD_ARGS env.sciworld.process_reward_mode=score_delta_prm"
    fi
else
    EXTRA_HEAD_ARGS=""
fi

# ---- 分组/估计器配置（按 MODE 分派）----
if [ "$MODE" = "gigpo" ]; then
    # GiGPO 官方口径：精确锚分组（无 BiGPO 聚类、无 CACB）
    GROUP_ARGS="algorithm.adv_estimator=gigpo \
        algorithm.gigpo.bisim_grouping=False \
        algorithm.gigpo.enable_similarity=False \
        algorithm.gigpo.step_advantage_w=1.0"
elif [ "$MODE" = "grpo" ]; then
    # GRPO 官方口径：episode 级分组（阶梯地板，无步级信用、无锚分组）
    GROUP_ARGS="algorithm.adv_estimator=grpo"
else
    # BiPACE-Q / 方法臂：BiGPO 聚类 + CACB
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

project_name="qwen2.5_1.5b_sciworld_bipace_node1613826"
_TAGPART=""
[ -n "$TAG" ] && _TAGPART="_${TAG}"
experiment_name="${MODE}_${START}start_sciworld_1.5b_4gpu_L-12_eps${EPS}_qstyle_actiontag_seed${SEED}${_TAGPART}_node1613826"
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
    env.env_name=sciworld \
    env.seed=$SEED \
    env.max_steps=40 \
    env.history_length=2 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.sciworld.prompt_mode=with_hints \
    env.sciworld.reward_mode=terminal \
    env.sciworld.simplifications_preset='' \
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
