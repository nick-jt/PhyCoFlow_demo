#!/bin/bash
#SBATCH --job-name=cnf_cap1024
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
# Capacity-matched arm. Upstream's case4.yml uses a 384-d lumped latent, but
# their largest 3D case is 32x34x62x3 = 201,984 DOF per snapshot. Ours is
# 125^3x4 = 7,812,500 DOF -- 39x larger -- so holding the latent at 384 imposes
# a compression ratio 39x more aggressive than anything CoNFiLD was validated
# at. An oracle 384-parameter code tops out at relL2 0.535 on this data; at
# 1024 the ceiling is 0.441. Everything else is upstream's recipe: zeros init,
# lr_nf 1e-4, lr_lat 1e-5, nf_accum 50 (decoder steps once per 50 latent steps).
A=50

L=confild_cap1024_${SLURM_JOB_ID}.log
# Diagnostic sweep over the decoder gradient-accumulation window. Upstream steps
# the decoder once per epoch; our epoch is 1200 items / batch 8 = 150 steps, so
# A=150 is the literal reading. But that yields only ~1k decoder updates over our
# budget vs upstream's 30k, so the ratio and the update count disagree -- hence a
# sweep rather than a guess. A=1 is the collapsed control we already ran (ratio 0.010).
# Everything else is upstream's recipe verbatim: zeros init, lr_nf 1e-4,
# lr_lat 1e-5, no weight decay, no warmup, latent tied to hidden width.
python confild_baseline.py \
  --out-dir ../Save_TrainedModel/JHU/baseline_confild/cnf_cap1024 \
  --latent-dim 1024 --n-group 8 --steps 150000 --batch 8 \
  --nf-accum $A --lr-nf 1e-4 --lr-lat 1e-5 \
  --latent-init-std 0.0 --lat-weight-decay 0.0 --nf-warmup 0 \
  --collapse-check-every 5000 --collapse-min 0.05 --collapse-grace 40000 \
  --save-every 15000 --tta-every 15000 --tta-steps 1000 >> $L 2>&1
echo "exit status: $?" >> $L
