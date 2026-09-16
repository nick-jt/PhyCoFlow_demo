#!/bin/bash
#SBATCH --job-name=cnf_bench_prior
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=64G
set -euo pipefail
source ~/envs/jhtdb
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python /home/ntricard/.claude/jobs/3ac3fd02/tmp/bench_prior.py
