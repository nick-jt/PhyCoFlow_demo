#!/bin/bash
#SBATCH --job-name=kolm_sweep_rest
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
#SBATCH --output=kolm_sweep_rest_%j.log
# latent-FM leg of the 2D density sweep (senseiver/mlp_rbf/geofno/sit already
# landed in job 3116596 before it hit its wall). S3GM is run separately at a
# reduced density set: it costs ~3 min/snapshot, so five densities would be
# 12.5 GPU-h to draw one curve.
set -uo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
STM=../Save_TrainedModel/kolmogorov2d
RUN=$(readlink -f "$(ls -d $STM/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN60_* | tail -1)")
python eval_kolm_ensemble.py --model latent_fm --run-dir "$RUN" --ckpt best --K 8 --nfe 4 \
    --n-obs-list 65 164 655 1965 6554 --cond-fields 0 --expect-val-len 640 \
    --stratify-blocks 1 --out-prefix kolm_sweep --resume --n-frames 50 \
    --seed 0 --op-seed 1000 --no-figs
