#!/bin/bash
#SBATCH --job-name=cnfB1a_s2ck
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
# Stage-B fix 1 (vehicle): stage-2 retrain from a stage-1 checkpoint with
# RETAINED periodic checkpoints for validation-based selection.
# usage: sbatch sB1a_s2ckpts.sh <stage1-ckpt> <out-dir> [budget_s]
set -euo pipefail
JT=/home/ntricard/.claude/jobs/3ac3fd02/tmp/confild_improve
S1="${1:?stage1 ckpt}"; OUT="${2:?out dir}"; BUDGET="${3:-21600}"
source ~/envs/jhtdb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 MPLCONFIGDIR=/tmp
echo "node: $(hostname)"
python "$JT/confild_stage2_ckpts.py" \
  --stage1-ckpt "$S1" --out-dir "$OUT" --budget-s "$BUDGET" --ckpt-every 25000
echo "=== s2ckpts done ==="
