#!/bin/bash
#SBATCH --job-name=qual_wing
#SBATCH --time=4:00:00
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
L=qualitative_wing_${SLURM_JOB_ID}.log
python qualitative_wing.py --case 0 --K 16 --n-steps 4 --tag wing >> $L 2>&1
echo "exit status: $?" >> $L
