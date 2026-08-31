#!/bin/bash
#SBATCH --job-name=cnf_stage1d
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=confild_stage1d_${SLURM_JOB_ID}.log
# Arm D: stabilised conditioning. Arm C (lr-lat 1e-2) lifted the latent-dependence
# ratio 0.010 -> 0.039 but was unstable -- lat_rms ran 0.43 -> 1.90 while the loss
# ROSE 0.026 -> 0.046, i.e. the latents diverged and the decoder chased them. So:
#   lr-lat 3e-3        back off from 1e-2; arm C overshot stability, not capacity.
#   lat-weight-decay   DeepSDF-style L2 prior on the latent table to stop the runaway.
#   nf-warmup 15000    3x longer, so latents organise before the decoder commits.
#   check every 5000   watch the ratio trend from step 5k, but only abort at 30k,
#                      so a slow-but-rising ratio is not killed prematurely.
python confild_baseline.py \
  --out-dir ../Save_TrainedModel/JHU/baseline_confild/cnf_stage1d \
  --latent-dim 1024 --n-group 8 \
  --steps 150000 --batch 8 \
  --lr-lat 3e-3 --latent-init-std 0.05 --nf-warmup 15000 --lat-weight-decay 1e-4 \
  --collapse-check-every 5000 --collapse-min 0.05 --collapse-grace 30000 \
  --save-every 15000 --tta-every 25000 --tta-steps 1000 >> $L 2>&1
echo "exit status: $?" >> $L
