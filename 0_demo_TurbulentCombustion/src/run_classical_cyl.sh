#!/bin/bash
#SBATCH --job-name=classical_cyl2d
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
#SBATCH --output=classical_cyl2d_%j.log

# Classical anchors on the native cylinder mesh (23800 cells). Non-periodic is
# the default and correct here (cylinder wake is not periodic). Sensors default
# to 1% = 238 per observed field; velocities only (--cond-fields 0 1), p never
# observed. The cylinder wake IS low-rank, so gappy-POD is the anchor to beat:
# rank 80 as the fleet default plus a rank-20 tag to probe the POD sweet spot.
# GPU node because the sensor draw runs on cuda:0 (the CUDA sensor-draw guard).
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

H5=/projects/ammoniacomb/generative_reconstruction/cylinder2d/Cylinder2D_mesh.h5
OUT=../Save_TrainedModel/cylinder2d/baseline_classical
mkdir -p "$OUT"

python -u baseline_classical_2d.py \
    --h5 "$H5" \
    --train-frames 1200 --fields Ux Uy p --cond-fields 0 1 \
    --methods constant kdtree idw gappy_pod --pod-rank 80 --idw-k 8 \
    --no-periodic --verify-percentile --tag main_1pct_pod80 \
    --out-dir "$OUT"

python -u baseline_classical_2d.py \
    --h5 "$H5" \
    --train-frames 1200 --fields Ux Uy p --cond-fields 0 1 \
    --methods gappy_pod --pod-rank 20 --idw-k 8 \
    --no-periodic --tag main_1pct_pod20 \
    --out-dir "$OUT"

echo "classical cyl2d done"
