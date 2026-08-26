#!/bin/bash
#SBATCH --job-name=cnf_cond_smoke
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=confild_cond_smoke_${SLURM_JOB_ID}.log
python confild_conditional.py \
  --cnf-ckpt ../Save_TrainedModel/JHU/baseline_confild/cnf_stage1/cnf_last.pt \
  --diff-ckpt ../Save_TrainedModel/JHU/baseline_confild/diff_stage2/diff_last.pt \
  --out-dir ../Save_TrainedModel/JHU/baseline_confild/cond_smoke \
  --snapshots 0 --K 2 >> $L 2>&1
echo "exit status: $?" >> $L
