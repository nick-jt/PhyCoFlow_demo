#!/bin/bash
#SBATCH --job-name=cnf_stage1b
#SBATCH --time=14:00:00
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
L=confild_stage1b_${SLURM_JOB_ID}.log
# Capacity-generous arm: latent 1024, group expansion 8 (each latent visited
# ~6x more often than the 48-way run), latent lr 3e-3, 150k steps.
python confild_baseline.py \
  --out-dir ../Save_TrainedModel/JHU/baseline_confild/cnf_stage1b \
  --latent-dim 1024 --n-group 8 \
  --steps 150000 --batch 8 --lr-lat 3e-3 \
  --save-every 15000 --tta-every 30000 --tta-steps 1000 >> $L 2>&1
echo "exit status: $?" >> $L
