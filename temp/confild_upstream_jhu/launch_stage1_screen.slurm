#!/bin/bash
#SBATCH --job-name=cnf1_screen
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --array=0-5

set -eu
source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

# Fidelity screen: upstream initialization and optimizer structure are fixed.
# Only scale-sensitive JHU parameters vary. All arms use the reflection subgroup
# first; only promoted configurations should pay for the full 48-element run.
LATENT=(384 384 384 768 768 1024)
DECODER_LR=(1e-4 3e-5 1e-4 1e-4 3e-5 1e-4)
LATENT_LR=(1e-5 3e-5 1e-4 3e-5 1e-4 3e-5)
POINTS=(65536 65536 131072 65536 131072 131072)

ARM="$SLURM_ARRAY_TASK_ID"
OUT="runs/screen_arm${ARM}_d${LATENT[$ARM]}"
python train_stage1.py \
  --out-dir "$OUT" \
  --groups 8 \
  --hidden 384 \
  --latent-dim "${LATENT[$ARM]}" \
  --decoder-lr "${DECODER_LR[$ARM]}" \
  --latent-lr "${LATENT_LR[$ARM]}" \
  --points-per-item "${POINTS[$ARM]}" \
  --batch-size 8 \
  --epochs 500 \
  --diagnostic-every 10 \
  --save-every 50

