#!/usr/bin/env bash
# ScienceWorld SFT 冷启动 · Qwen2.5-1.5B-Instruct · 本机（ai-1613826）
# 数据：AgentTraj-L 2120 轨迹 -> 42178 step 样本（<think>..</think><action>..</action> 格式）
# 输出 checkpoint 作为 RL 阶段统一起点（baseline 与 bisimhead 共用——SFT 是消融轴之一）
# 注意：fsdp_sft_trainer 不走 chat template（纯 prompt/response 拼接），与 RL 阶段的
# raw_chat 格式有差异——但 SFT 数据的 prompt 即 RL 的用户模板（prepare 脚本同源生成），
# Qwen 的 chatml 头部差异对 1.5B 冷启动影响有限；若 RL 验证掉点明显再换 transformers 直训。
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace-webshop
export JAVA_HOME=${JAVA_HOME:-/root/miniconda3/envs/bipace-webshop}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sciworld_sft_1.5b_node1613826_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] ScienceWorld SFT cold-start 1.5B started"

set -x

CKPT_DIR=${BASE_DIR}/verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/verl-agent/sciworld_sft/train.parquet \
    data.val_files=$HOME/data/verl-agent/sciworld_sft/val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.prompt_dict_keys=[] \
    data.response_dict_keys=[] \
    data.max_length=3584 \
    data.micro_batch_size_per_gpu=8 \
    optim.lr=1e-5 \
    model.partial_pretrain=$MODEL_PATH \
    use_remove_padding=true \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.project_name=qwen2.5_1.5b_sciworld_sft_node1613826 \
    trainer.experiment_name=sft_full \
    trainer.total_epochs=3 \
    trainer.logger=['console'] \
    trainer.save_freq=-1 \
