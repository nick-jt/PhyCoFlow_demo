#!/bin/bash
#SBATCH --job-name=smoke_cyl2d
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
#SBATCH --output=smoke_cyl2d_%j.log

# Smoke gates for the 2D cylinder benchmark: short runs of (1) DMF-Gen
# point-cloud FFM on the native mesh, (2) Senseiver Det baseline on the mesh,
# (3) latent-FM stage 1 on the 200x400 ROI grid, all on the Re-holdout
# protocol. PASS = loss finite+descending, [train] lines carry time/peak_mem,
# no shape errors. The full fleet is submitted --dependency=afterok on this,
# so -e/pipefail make a trainer crash actually block the fleet (the kolm smoke
# piped through tail without pipefail and could not gate anything).
set -euo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

echo "=== [1/3] DMF-Gen (pointcloud FFM, mesh) ==="
python train_pointcloud_ffm.py \
    --config Save_config/cylinder2d/config_bench_cyl_ffm_smoke.yaml \
    --Demo-Num 996 2>&1 | tail -30

echo "=== [2/3] Senseiver (Det, mesh) ==="
python train_Det_Baseline.py \
    --config Save_config/cylinder2d/config_baseline_Det_cyl_smoke.yaml 2>&1 | tail -25

echo "=== [3/3] latent-FM stage 1 (grid) ==="
python train_Gen_Baseline.py \
    --config Save_config/cylinder2d/config_baseline_Gen_cyl_smoke.yaml 2>&1 | tail -25

echo "=== smoke done ==="
