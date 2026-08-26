#!/bin/bash
#SBATCH --job-name=cnf_stage2
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=confild_stage2_${SLURM_JOB_ID}.log
python confild_stage2.py \
  --cnf-ckpt ../Save_TrainedModel/JHU/baseline_confild/cnf_stage1/cnf_last.pt \
  --out-dir ../Save_TrainedModel/JHU/baseline_confild/diff_stage2 >> $L 2>&1
echo "exit status: $?" >> $L
