#!/bin/bash
#SBATCH --job-name=spectra_arms
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
python spectra_arms.py > spectra_arms_${SLURM_JOB_ID}.log 2>&1
echo "exit: $?" >> spectra_arms_${SLURM_JOB_ID}.log
