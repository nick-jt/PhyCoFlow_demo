#!/bin/bash
#SBATCH --job-name=fno3d_smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

DEMO_NUM=91
LOG_FILE="train_fno3d_smoke_${SLURM_JOB_ID}.log"
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, python: $(which python)" > "$LOG_FILE"
CUDA_VISIBLE_DEVICES=0 python -u train_pointcloud_ffm.py \
  --config Save_config/config_baseline_fno3d_smoke.yaml \
  --Demo-Num $DEMO_NUM >> "$LOG_FILE" 2>&1
