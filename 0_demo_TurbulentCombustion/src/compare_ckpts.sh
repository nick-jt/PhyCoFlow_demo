#!/bin/bash
#SBATCH --job-name=cmp_ckpts
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=64G
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
python compare_ckpts.py > compare_ckpts_out.txt 2>&1
