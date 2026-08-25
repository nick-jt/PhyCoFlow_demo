#!/bin/bash
#SBATCH --job-name=smoke_det
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
python train_Det_Baseline.py --config /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config/config_baseline_Det_bench.yaml > smoke_det_${SLURM_JOB_ID}.log 2>&1
echo "exit=$?"
