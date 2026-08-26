#!/bin/bash
#SBATCH --job-name=cq_flash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=cq_flash_${SLURM_JOB_ID}.log
python test_cq_flash.py >> $L 2>&1
echo "exit status: $?" >> $L
