#!/bin/bash
#SBATCH --job-name=wing_classical
#SBATCH --account=f2pde
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=wing_classical_%j.log

# Training-free classical baselines for SHIFT-WING.  Pure CPU/numpy: no GPU is
# requested and the reported cost fields say device=cpu, gpu_mem_gb=null.
# Peak RSS measured at 8.3 GB, ~100 s per val case with 32 workers.
# Canonical sensor draw + canonical 8 val cases; see baseline_classical_wing.py.

set -euo pipefail
source ~/envs/jhtdb

WT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
cd "$WT/src"

python baseline_classical_wing.py \
    --processed-root /projects/ammoniacomb/generative_reconstruction/shift_wing/processed_v3 \
    --out-dir "$WT/Save_TrainedModel/wing/baseline_classical" \
    --n-cases 8 --n-taps 512 --n-shear 128 --seed 0 \
    --ranks 0 10 20 40 80 160 --idw-k 8 \
    --workers 32 \
    --methods train_mean idw gappy_pod
