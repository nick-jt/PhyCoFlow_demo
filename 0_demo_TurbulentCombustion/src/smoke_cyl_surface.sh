#!/bin/bash
#SBATCH --job-name=smoke_cylsurf
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=64G
#SBATCH --output=smoke_cyl_surface_%j.log

# Smoke gates for the cylinder SURFACE-TO-FIELD task (P0.5): 3 epochs each of
# (1) DMF-Gen on the mesh with sensor_pool=surface (360-cell wall ring, all
# fields at the taps, compile ON), (2) Senseiver Det on the mesh, (3) latent-FM
# stage 1 on the ROI grid (pool = 62 nearest fluid cells via the sidecar).
# PASS = every trainer prints "[dataset] sensor_pool=surface" and finite,
# descending losses; the surface fleet is --dependency=afterok on this job.
set -euo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

echo "=== [1/3] DMF-Gen (pointcloud FFM, mesh, surface pool) ==="
python train_pointcloud_ffm.py \
    --config Save_config/cylinder2d_surface/config_bench_cyl_surface_ffm_smoke.yaml \
    --Demo-Num 993 2>&1 | grep -vE 'it/s\]|KeOps\] (Compiling|Generating)' | tail -30

echo "=== [2/3] Senseiver (Det, mesh, surface pool) ==="
python train_Det_Baseline.py \
    --config Save_config/cylinder2d_surface/config_baseline_Det_cyl_surface_smoke.yaml 2>&1 | grep -vE 'it/s\]' | tail -25

echo "=== [3/3] latent-FM stage 1 (grid, surface pool) ==="
python train_Gen_Baseline.py \
    --config Save_config/cylinder2d_surface/config_baseline_Gen_cyl_surface_smoke.yaml 2>&1 | grep -vE 'it/s\]' | tail -25

echo "=== surface smoke done ==="
