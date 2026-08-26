#!/bin/bash
#SBATCH --job-name=qual_figs
#SBATCH --time=4:00:00
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
L=qualitative_${SLURM_JOB_ID}.log
python qualitative_jhu.py --snapshot 3 --K 16 --n-steps 4 --tag jhu >> $L 2>&1
echo "exit status: $?" >> $L
