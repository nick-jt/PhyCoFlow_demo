#!/bin/bash
#SBATCH --job-name=smoke_spec
#SBATCH --time=0:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral_proper,so3,translate
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
python train_pointcloud_ffm.py --config Save_config/config_smoke_spec.yaml --Demo-Num 96 > smoke_spec_${SLURM_JOB_ID}.log 2>&1
echo "exit=$?"
