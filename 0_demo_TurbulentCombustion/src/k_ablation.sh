#!/bin/bash
#SBATCH --job-name=kabl
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
python k_ablation.py > k_ablation_out.txt 2>&1
