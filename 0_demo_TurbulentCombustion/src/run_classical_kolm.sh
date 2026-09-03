#!/bin/bash
#SBATCH --job-name=classical_kolm2d
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
#SBATCH --output=classical_kolm2d_%j.log

set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

H5=/projects/ammoniacomb/generative_reconstruction/kolmogorov2d/Kolmogorov2D_shu_stride4.h5
OUT=../Save_TrainedModel/kolmogorov2d/baseline_classical
mkdir -p "$OUT"

python -u baseline_classical_2d.py \
    --h5 "$H5" \
    --train-frames 2560 --fields vorticity --cond-fields 0 \
    --methods constant kdtree idw gappy_pod --pod-rank 80 --idw-k 8 \
    --no-periodic --verify-percentile --tag main_1pct_nonperiodic \
    --out-dir "$OUT"

python -u baseline_classical_2d.py \
    --h5 "$H5" \
    --train-frames 2560 --fields vorticity --cond-fields 0 \
    --methods constant kdtree idw gappy_pod --pod-rank 80 --idw-k 8 \
    --periodic --tag main_1pct_periodic \
    --out-dir "$OUT"

echo "classical kolm2d done"
