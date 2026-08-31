#!/bin/bash
#SBATCH --job-name=fb_merge
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --mem=16G
# Merge the two FireBench case extracts (u10 first, then u12) into the
# training file the configs expect. Streamed, ~9 GB written to scratch.
set -u
export PYTHONUNBUFFERED=1
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
FB=/home/ntricard/orcd/scratch/firebench3d
python -u $DEMO/src/engaging/merge_firebench_cases.py \
    --inputs $FB/firebench3d/FireBench_u10_ramp0_3D_dense.h5 \
             $FB/firebench3d/FireBench_u12_ramp0_3D_dense.h5 \
    --out $FB/FireBench_u10u12_merged.h5
