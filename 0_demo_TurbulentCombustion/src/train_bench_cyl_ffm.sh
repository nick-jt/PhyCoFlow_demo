#!/bin/bash
#SBATCH --job-name=bench_cyl_ffm
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=64G

set -u
# Re-holdout protocol: frames grouped train-Re-first in the H5
# ([0,1200)=Re{60,100,150,200}, [1200,1800)=Re{80,250}), so a gapless block
# split at train_ratio 0.6667 = the 4/2 Reynolds-number holdout.
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

# env overrides (surface-to-field fleet): CONFIG, DEMO_NUM
DEMO_NUM=${DEMO_NUM:-102}
CONFIG=${CONFIG:-Save_config/cylinder2d/config_bench_cyl_ffm.yaml}
LOG_FILE="train_bench_cyl_ffm_${SLURM_JOB_ID}_DemoN${DEMO_NUM}.log"
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, python: $(which python) config=$CONFIG"
CUDA_VISIBLE_DEVICES=0 python train_pointcloud_ffm.py \
    --RELOAD \
    --config "$CONFIG" \
    --Demo-Num $DEMO_NUM >> "$LOG_FILE" 2>&1
echo "exit status: $?"
