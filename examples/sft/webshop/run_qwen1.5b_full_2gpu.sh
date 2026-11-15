#!/bin/bash
# Full SFT of Qwen2.5-1.5B-Instruct on the ETO webshop expert trajectories.
# Target: WebShop success_rate ~5% (matches ETO paper's SFT baseline ballpark).
#
# Prereqs:
#   1. python -m examples.data_preprocess.prepare_eto_sft_webshop
#      → writes ~/data/verl-agent/eto_sft_webshop/{train,test}.parquet
#   2. 2 × H100 / A100 (≥ 40GB each). For smaller cards, drop micro_batch_size_per_gpu.
#
# Usage:
#   bash examples/sft/webshop/run_qwen1.5b_full_2gpu.sh [save_path]
#
set -x

save_path=${1:-$PWD/ckpts/eto_sft_webshop_qwen1.5b}
data_dir=${ETO_SFT_DATA_DIR:-$HOME/data/verl-agent/eto_sft_webshop}

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=8192 \
    data.truncation=right \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct \
    model.enable_gradient_checkpointing=true \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.offload_params=false \
    model.lora_rank=0 \
    model.strategy=fsdp2 \
    use_remove_padding=true \
    ulysses_sequence_parallel_size=1 \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.03 \
    optim.weight_decay=0.01 \
    optim.lr_scheduler=cosine \
    trainer.total_epochs=3 \
    trainer.save_freq=40 \
    trainer.default_local_dir=$save_path \
    trainer.default_hdfs_dir=null \
    trainer.project_name=eto-sft-webshop \
    trainer.experiment_name=qwen2.5-1.5b-full-2gpu \
    trainer.logger=['console','wandb'] \
    trainer.seed=0
