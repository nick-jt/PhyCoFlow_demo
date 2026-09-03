#!/bin/bash
#SBATCH --job-name=bench_kolm_ffm
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G

set -u
# Trajectory-holdout protocol: frames grouped train-trajs-first in the H5,
# so a gapless block split at train_ratio 0.8 = 32/8 trajectory holdout.
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

DEMO_NUM=101
LOG_FILE="train_bench_kolm_ffm_${SLURM_JOB_ID}_DemoN${DEMO_NUM}.log"
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, python: $(which python)"
CUDA_VISIBLE_DEVICES=0 python train_pointcloud_ffm.py \
    --RELOAD \
    --config Save_config/kolmogorov2d/config_bench_kolm_ffm.yaml \
    --Demo-Num $DEMO_NUM >> "$LOG_FILE" 2>&1
echo "exit status: $?"
