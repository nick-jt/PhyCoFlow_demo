#!/bin/bash
#SBATCH --job-name=s3gm_smoke
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
source ~/envs/jhtdb
cd /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src
CFGROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config
LOG="s3gm_smoke_${SLURM_JOB_ID}.log"

echo "=== SEEDCHECK (canonical helpers.py draw, compute node) ===" >> "$LOG"
python verify_seedcheck.py --config $CFGROOT/config_baseline_S3GM_smoke.yaml \
    --snaps 29 --seed 0 --n-obs 19531 >> "$LOG" 2>&1
echo "seedcheck rc=$?" >> "$LOG"

echo "=== SMOKE TRAIN (6 epochs, figures at 3 and 6) ===" >> "$LOG"
CUDA_VISIBLE_DEVICES=0 python train_s3gm3d.py \
    --config $CFGROOT/config_baseline_S3GM_smoke.yaml \
    --training-stage 1 >> "$LOG" 2>&1
RC=$?
echo "train rc=$RC" >> "$LOG"
exit $RC
