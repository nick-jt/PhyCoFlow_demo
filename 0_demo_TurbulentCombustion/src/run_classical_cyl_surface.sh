#!/bin/bash
#SBATCH --job-name=classical_cylsurf
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=64G
#SBATCH --output=classical_cyl_surface_%j.log

# SURFACE-TO-FIELD variant: sensors only on the 360-cell wall ring (all three
# fields at the taps), tap budget 32/64/128/360. Prediction 2 of the
# pre-registration: IDW/kdtree collapse toward the train-mean floor; the
# honest risk (prediction 3) is that gappy POD still wins.
# Classical anchors on the native cylinder mesh (23800 cells).
# the default and correct here (cylinder wake is not periodic). Sensors default
# to 1% = 238 per observed field; velocities only (--cond-fields 0 1), p never
# observed. The cylinder wake IS low-rank, so gappy-POD is the anchor to beat:
# rank 80 as the fleet default plus a rank-20 tag to probe the POD sweet spot.
# GPU node because the sensor draw runs on cuda:0 (the CUDA sensor-draw guard).
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

H5=/work/hdd/bilr/ntricard/datasets/cylinder2d/Cylinder2D_mesh.h5
OUT=../Save_TrainedModel/cylinder2d_surface/baseline_classical
mkdir -p "$OUT"

python -u baseline_classical_2d.py \
    --h5 "$H5" \
    --train-frames 1200 --fields Ux Uy p --cond-fields 0 1 2 --sensor-pool surface --n-sensors 32 64 128 360 \
    --methods constant kdtree idw gappy_pod --pod-rank 80 --idw-k 8 \
    --no-periodic --verify-percentile --tag surface_pod80 \
    --out-dir "$OUT"

python -u baseline_classical_2d.py \
    --h5 "$H5" \
    --train-frames 1200 --fields Ux Uy p --cond-fields 0 1 2 --sensor-pool surface --n-sensors 32 64 128 360 \
    --methods gappy_pod --pod-rank 20 --idw-k 8 \
    --no-periodic --tag surface_pod20 \
    --out-dir "$OUT"

echo "classical cyl surface done"
