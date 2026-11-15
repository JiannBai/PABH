#!/usr/bin/env bash
# ScienceWorld SFT 冷启动 · Qwen2.5-7B-Instruct · 本机（ai-1753446）
# 数据：AgentTraj-L 2120 轨迹 -> 42178 step 样本（data_backup/sciworld_sft/ 有共享备份）
# 输出 checkpoint 作为 7B RL 阶段 SFT 消融轴的统一起点（baseline 与方法臂共用）
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bipace-webshop
export JAVA_HOME=${JAVA_HOME:-/root/miniconda3/envs/bipace-webshop}
unset TRITON_PTXAS_PATH

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
MODEL_PATH=${MODEL_PATH:-/mnt/tidalfs-hssh01/dataset/bxd/research/agenticRL/models/Qwen2.5-7B-Instruct}

LOG_DIR=${LOG_DIR:-$BASE_DIR/verl_log}
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sciworld_sft_7b_node1613826_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] ScienceWorld SFT cold-start 7B started (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

set -x

CKPT_DIR=${BASE_DIR}/verl_checkpoints/qwen2.5_7b_sciworld_sft_node1613826/sft_full

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/verl-agent/sciworld_sft/train.parquet \
    data.val_files=$HOME/data/verl-agent/sciworld_sft/val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.prompt_dict_keys=[] \
    data.response_dict_keys=[] \
    data.max_length=3584 \
    data.micro_batch_size_per_gpu=4 \
    optim.lr=1e-5 \
    model.partial_pretrain=$MODEL_PATH \
    use_remove_padding=true \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.project_name=qwen2.5_7b_sciworld_sft_node1613826 \
    trainer.experiment_name=sft_full \
    trainer.total_epochs=3 \
    trainer.logger=['console'] \
    trainer.save_freq=-1
