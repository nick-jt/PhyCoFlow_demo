#!/bin/bash
#SBATCH --job-name=gen4turb_jhu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=180G
set -u
source ~/envs/jhtdb
# Their loader hardcodes ../data/u.npy, so run from the task's dm/ directory.
cd /projects/ammoniacomb/generative_reconstruction/baselines/Gen4Turbulence/3_flow_reconstruction/dm
LOG=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src/train_gen4turb_jhu_${SLURM_JOB_ID}.log
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST" > $LOG
python train_model.py >> $LOG 2>&1
echo "exit=$?" >> $LOG
