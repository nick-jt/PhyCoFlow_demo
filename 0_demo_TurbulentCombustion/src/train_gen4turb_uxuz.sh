#!/bin/bash
#SBATCH --job-name=g4t_uxuz
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
source ~/envs/jhtdb
cd /projects/ammoniacomb/generative_reconstruction/baselines/Gen4Turbulence/3_flow_reconstruction/dm
L=$SLURM_SUBMIT_DIR/train_gen4turb_uxuz_${SLURM_JOB_ID}.log
python train_model_uxuz.py >> $L 2>&1
echo "exit: $?" >> $L
