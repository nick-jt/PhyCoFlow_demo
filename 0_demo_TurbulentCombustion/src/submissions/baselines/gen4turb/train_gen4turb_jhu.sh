#!/bin/bash
#SBATCH --job-name=gen4turb_jhu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=180G
set -u
source ~/envs/jhtdb
# Their loader hardcodes ../data/u.npy, so run from the task's dm/ directory.
cd /work/hdd/bilr/ntricard/datasets/baselines/Gen4Turbulence/3_flow_reconstruction/dm
LOG=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/src/train_gen4turb_jhu_${SLURM_JOB_ID}.log
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST" > $LOG
python train_model.py >> $LOG 2>&1
echo "exit=$?" >> $LOG
