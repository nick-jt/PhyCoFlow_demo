#!/bin/bash
#SBATCH --job-name=smoke_kolm2d
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
#SBATCH --output=smoke_kolm2d_%j.log

# Smoke gates for the 2D Kolmogorov benchmark: 3 epochs each of
# (1) DMF-Gen point-cloud FFM, (2) Senseiver Det baseline, (3) latent-FM
# stage 1, on the trajectory-holdout protocol. PASS = loss finite+descending,
# [train] lines carry time/peak_mem, no shape errors.
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

echo "=== [1/3] DMF-Gen (pointcloud FFM) ==="
python train_pointcloud_ffm.py \
    --config Save_config/kolmogorov2d/config_bench_kolm_ffm_smoke.yaml \
    --Demo-Num 999 2>&1 | tail -30

echo "=== [2/3] Senseiver (Det) ==="
python train_Det_Baseline.py \
    --config Save_config/kolmogorov2d/config_baseline_Det_kolm_smoke.yaml 2>&1 | tail -25

echo "=== [3/3] latent-FM stage 1 ==="
python train_Gen_Baseline.py \
    --config Save_config/kolmogorov2d/config_baseline_Gen_kolm_smoke.yaml 2>&1 | tail -25

echo "=== smoke done ==="
