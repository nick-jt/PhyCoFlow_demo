#!/bin/bash
#SBATCH --job-name=profile
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
python profile_step.py > profile_step_out.txt 2>&1
