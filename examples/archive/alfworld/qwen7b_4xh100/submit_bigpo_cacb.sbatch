#!/bin/bash
# =====================================================================
# HPC3 (3期) submission script for BiGPO + CACB on ALFWorld / Qwen2.5-7B.
#
# Submit:  sbatch submit_alfworld_bigpo_cacb.sbatch
# Cancel:  scancel <jobid>
# Status:  squeue -u $USER
# Tail:    tail -f train_logs/train.log    (after the job starts)
# =====================================================================

#SBATCH --job-name=bigpo_cacb_qwen7b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=639G

# >>> EDIT BEFORE FIRST SUBMIT <<<
#SBATCH --partition=gpu                    # confirm via `sinfo` or cluster docs (e.g. gpu / hgx / a100)
#SBATCH --account=your_project_account     # your HPC3 charge code
# >>> END EDIT <<<

#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/bigpo_cacb_%j.out
#SBATCH --error=slurm_logs/bigpo_cacb_%j.err
# Optional: email on job state change
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@institution.edu

set -euo pipefail
mkdir -p slurm_logs

# ---- Banner ---------------------------------------------------------------
echo "========================================================="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "GPUs      : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "GPU info  :"
nvidia-smi -L || true
echo "Started at: $(date)"
echo "Workdir   : ${SLURM_SUBMIT_DIR:-$PWD}"
echo "========================================================="
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ---- Modules + conda ------------------------------------------------------
# Adjust these names if your cluster uses a different module-name scheme.
# `module avail cuda` will show the available CUDA modules; pick one matching
# the verl-agent env's torch build (CUDA 12.1+ is fine for torch 2.4).
module purge || true
module load cuda/12.1 || true        # required for vLLM kernels
module load gcc/11.2  || true        # FSDP build sometimes needs newer libstdc++

# Activate the verl-agent conda env. Adjust path to match your install.
source /share/anaconda3/etc/profile.d/conda.sh
conda activate verl-agent

echo "python    : $(which python3)"
echo "torch     : $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "ray       : $(python3 -c 'import ray; print(ray.__version__)' 2>/dev/null || echo 'n/a')"
echo "---------------------------------------------------------"

# ---- Run ------------------------------------------------------------------
# The training script handles HF / SwanLab / ALFWorld env vars internally
# and writes per-step CSVs + train.log. SwanLab uses a per-run logdir
# (swanlog_bigpo_cacb/) so no SQLite collision with other jobs.
bash examples/bipace/alfworld/qwen7b_4xh100/train_bigpo_cacb.sh

echo "========================================================="
echo "Finished at: $(date)"
echo "Exit code  : $?"
echo "========================================================="
