#!/bin/bash
#SBATCH --job-name=bench_cyl_ffm
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G

set -u
# Re-holdout protocol: frames grouped train-Re-first in the H5
# ([0,1200)=Re{60,100,150,200}, [1200,1800)=Re{80,250}), so a gapless block
# split at train_ratio 0.6667 = the 4/2 Reynolds-number holdout.
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

DEMO_NUM=102
LOG_FILE="train_bench_cyl_ffm_${SLURM_JOB_ID}_DemoN${DEMO_NUM}.log"
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, python: $(which python)"
CUDA_VISIBLE_DEVICES=0 python train_pointcloud_ffm.py \
    --RELOAD \
    --config Save_config/cylinder2d/config_bench_cyl_ffm.yaml \
    --Demo-Num $DEMO_NUM >> "$LOG_FILE" 2>&1
echo "exit status: $?"
