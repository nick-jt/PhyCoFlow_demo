#!/bin/bash
#SBATCH --job-name=cnf_stage1
#SBATCH --time=12:00:00
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
L=confild_stage1_${SLURM_JOB_ID}.log
python confild_baseline.py \
  --out-dir ../Save_TrainedModel/JHU/baseline_confild/cnf_stage1 \
  --steps 100000 --batch 8 --lr-lat 1e-3 \
  --save-every 10000 --tta-every 25000 >> $L 2>&1
echo "exit status: $?" >> $L
