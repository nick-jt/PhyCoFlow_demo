#!/bin/bash
#SBATCH --job-name=iclr_wing_v4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mail-user=ntricard@mit.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=72G

set -u
export WING_AUGMENT=reflect_y
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

DEMO_NUM=19
LOG_FILE="train_iclr_wing_v4_${SLURM_JOB_ID}_DemoN${DEMO_NUM}.log"
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, python: $(which python)"
CUDA_VISIBLE_DEVICES=0 python train_pointcloud_ffm.py \
        --RELOAD \
        --config Save_config/config_iclr_wing_v4.yaml \
        --Demo-Num $DEMO_NUM >> "$LOG_FILE" 2>&1
echo "exit status: $?"
