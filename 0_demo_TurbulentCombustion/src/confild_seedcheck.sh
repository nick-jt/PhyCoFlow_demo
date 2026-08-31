#!/bin/bash
#SBATCH --job-name=cnf_seedcheck
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
set -euo pipefail
source ~/envs/jhtdb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLCONFIGDIR=/tmp
python /home/ntricard/.claude/jobs/3ac3fd02/tmp/seedcheck.py
