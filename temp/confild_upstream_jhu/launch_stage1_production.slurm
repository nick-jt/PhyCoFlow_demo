#!/bin/bash
#SBATCH --job-name=cnf1_g48
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

set -eu
source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

# Replace the three promoted values below after comparing the screen histories.
LATENT_DIM="${LATENT_DIM:-384}"
DECODER_LR="${DECODER_LR:-1e-4}"
LATENT_LR="${LATENT_LR:-3e-5}"
POINTS_PER_ITEM="${POINTS_PER_ITEM:-65536}"

python train_stage1.py \
  --out-dir "runs/production_g48_d${LATENT_DIM}" \
  --groups 48 \
  --hidden 384 \
  --latent-dim "$LATENT_DIM" \
  --decoder-lr "$DECODER_LR" \
  --latent-lr "$LATENT_LR" \
  --points-per-item "$POINTS_PER_ITEM" \
  --batch-size 8 \
  --epochs 1000 \
  --diagnostic-every 10 \
  --save-every 25

