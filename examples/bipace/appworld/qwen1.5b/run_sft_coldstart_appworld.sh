#!/usr/bin/env bash
# AppWorld SFT 冷启动 · Qwen2.5-1.5B-Instruct · 本机（ai-1845503）
# 数据：专家轨迹解 -> 81 train / 9 val 一次性样本（<think> + <code> 格式）
# 输出 checkpoint 作为 RL 阶段 baseline/gigpo/bisimhead_prm 三臂的统一冷启动起点。
# 用法：CUDA_VISIBLE_DEVICES=4,5 bash run_sft_coldstart_appworld.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace
unset TRITON_PTXAS_PATH

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-1.5B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/appworld_sft_1.5b_node1797956_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] AppWorld SFT cold-start 1.5B started"

set -x

CKPT_DIR=${BASE_DIR}/verl_checkpoints/qwen2.5_1.5b_appworld_sft_node1797956/sft_full

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/verl-agent/appworld_sft/train.parquet \
    data.val_files=$HOME/data/verl-agent/appworld_sft/val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.prompt_dict_keys=[] \
    data.response_dict_keys=[] \
    data.max_length=8192 \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=2 \
    optim.lr=1e-5 \
    model.partial_pretrain=$MODEL_PATH \
    use_remove_padding=true \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.project_name=qwen2.5_1.5b_appworld_sft_node1797956 \
    trainer.experiment_name=sft_full \
    trainer.total_epochs=3 \
    trainer.logger=['console'] \
    trainer.save_freq=-1 \
