#!/bin/bash
#SBATCH --job-name=iclr_senseiver
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mail-user=ntricard@mit.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=72G

set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
LOG_FILE="train_iclr_det_senseiver_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 python train_Det_Baseline.py \
        --config /work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_config/config_baseline_Det.yaml >> "$LOG_FILE" 2>&1
echo "exit status: $?"
