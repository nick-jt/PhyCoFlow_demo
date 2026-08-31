#!/bin/bash
#SBATCH --job-name=cnf_eval
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#
# usage: sbatch confild_eval_full.sh <stage1-run-dir> <stage2-run-dir>
#
# Canonical protocol, identical to ensemble_eval.py:291-304:
#   seed 0, op-seed 1000, all 50 held-out snapshots, K=8,
#   observed channels 0 and 2, FIXED 19,531 sensors per channel (1%).
# Measured on the smoke run: ~2.7 DPS it/s at window 32 -> ~1.7 h of guided
# sampling + ~0.5 h of full-field decoding.

set -euo pipefail
REPO=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
S1DIR="${1:?usage: sbatch confild_eval_full.sh <stage1-run-dir> <stage2-run-dir>}"
S2DIR="${2:?usage: sbatch confild_eval_full.sh <stage1-run-dir> <stage2-run-dir>}"

source ~/envs/jhtdb
cd "$REPO/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 MPLCONFIGDIR=/tmp
export TQDM_DISABLE=1

for SEL in best last; do            # validation-selected AND budget-matched
  echo "=== checkpoint selection: ${SEL} ==="
  python confild_eval_unified.py \
    --stage1-ckpt "${S1DIR}/${SEL}.pt" \
    --stage2-ckpt "${S2DIR}/${SEL}.pt" \
    --out-dir "${S2DIR}/eval_${SEL}" \
    --seed 0 --op-seed 1000 --n-snapshots 50 --K 8 \
    --cond-fields 0 2 --n-obs 19531 19531 \
    --steps 1000 --row-chunk 4
done
