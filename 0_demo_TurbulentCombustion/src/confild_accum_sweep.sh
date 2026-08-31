#!/bin/bash
#SBATCH --job-name=cnf_accum
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --array=0-2
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
ACC=(10 50 150)
A=${ACC[$SLURM_ARRAY_TASK_ID]}
L=confild_accum${A}_${SLURM_ARRAY_JOB_ID}.log
# Diagnostic sweep over the decoder gradient-accumulation window. Upstream steps
# the decoder once per epoch; our epoch is 1200 items / batch 8 = 150 steps, so
# A=150 is the literal reading. But that yields only ~1k decoder updates over our
# budget vs upstream's 30k, so the ratio and the update count disagree -- hence a
# sweep rather than a guess. A=1 is the collapsed control we already ran (ratio 0.010).
# Everything else is upstream's recipe verbatim: zeros init, lr_nf 1e-4,
# lr_lat 1e-5, no weight decay, no warmup, latent tied to hidden width.
python confild_baseline.py \
  --out-dir ../Save_TrainedModel/JHU/baseline_confild/cnf_accum${A} \
  --n-group 8 --steps 35000 --batch 8 \
  --nf-accum $A --lr-nf 1e-4 --lr-lat 1e-5 \
  --latent-init-std 0.0 --lat-weight-decay 0.0 --nf-warmup 0 \
  --collapse-check-every 5000 --collapse-min 0.05 --collapse-grace 999999 \
  --save-every 17500 --tta-every 17500 --tta-steps 1000 >> $L 2>&1
echo "exit status: $?" >> $L
