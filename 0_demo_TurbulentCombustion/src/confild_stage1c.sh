#!/bin/bash
#SBATCH --job-name=cnf_stage1c
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
L=confild_stage1c_${SLURM_JOB_ID}.log
# Conditioning-fix arm. Capacity is IDENTICAL to arm B (latent 1024, group 8,
# hidden 384, 15 layers) so this isolates the conditioning collapse: arm B's
# decoder ignored its latent (across-latent output ratio 0.010 -- swapping in a
# different snapshot's latent, or a zero vector, moved the output by 1.3%).
#
#   latent-init-std 0.05  zeros-init makes the FiLM modulation identically zero
#                         for every item at step 0, so the fastest descent is a
#                         latent-independent mean field.
#   lr-lat 1e-2           3.3x arm B; widens the latent:decoder lr ratio to 100:1.
#   nf-warmup 5000        let the latents become informative before the decoder
#                         commits to a solution.
#   collapse-check-every  abort at ~1.9 h if the decoder is still ignoring its
#                         latent, instead of burning 11 h to learn a mean field.
python confild_baseline.py \
  --out-dir ../Save_TrainedModel/JHU/baseline_confild/cnf_stage1c \
  --latent-dim 1024 --n-group 8 \
  --steps 150000 --batch 8 \
  --lr-lat 1e-2 --latent-init-std 0.05 --nf-warmup 5000 \
  --collapse-check-every 25000 --collapse-min 0.05 \
  --save-every 15000 --tta-every 25000 --tta-steps 1000 >> $L 2>&1
echo "exit status: $?" >> $L
