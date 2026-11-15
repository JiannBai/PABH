#!/bin/bash
# Full SFT of Qwen2.5-1.5B-Instruct on the ETO webshop expert trajectories,
# using HuggingFace Trainer (NOT verl-agent's fsdp_sft_trainer — that one has
# unreliable mid-epoch checkpointing). save_steps=40 is rock solid here.
#
# Prereqs:
#   1. python -m examples.data_preprocess.prepare_eto_sft_webshop
#      → writes <data_dir>/{train,test}.parquet
#   2. transformers >= 4.41, accelerate, wandb
#
# Usage:
#   bash examples/sft/webshop/run_qwen1.5b_full_2gpu_hf.sh [save_path]
#
set -x

save_path=${1:-$PWD/ckpts/eto_sft_webshop_qwen1.5b_hf}
data_dir=${ETO_SFT_DATA_DIR:-$HOME/data/verl-agent/eto_sft_webshop}
model_path=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    examples/sft/webshop/train_sft_hf.py \
    --model $model_path \
    --train_file $data_dir/train.parquet \
    --eval_file $data_dir/test.parquet \
    --output_dir $save_path \
    --epochs 3 \
    --lr 1e-5 \
    --per_device_bs 2 \
    --grad_accum 8 \
    --max_length 8192 \
    --truncation right \
    --save_steps 40 \
    --logging_steps 5 \
    --project eto-sft-webshop \
    --run_name qwen2.5-1.5b-hf-2gpu
