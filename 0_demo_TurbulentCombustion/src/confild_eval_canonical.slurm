#!/bin/bash
#SBATCH --job-name=cnf_evalcanon
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#
# usage: sbatch --dependency=afterany:<trainjob> confild_eval_canonical.slurm <arm-save-root>
#
# Canonical CoNFiLD evaluation of the unified two-stage checkpoints:
#   seed 0, op-seed 1000, all 50 held-out snapshots, K=8, DPS scale 1.0,
#   cond fields 0 2, FIXED 19,531 sensors per channel, 1000 DDPM steps.
# Fingerprint must read: [seedcheck] snap=29 sensors=39062 idx_sum=37987162596
#
# Evaluates BOTH checkpoint selections:
#   best = diagnostic-selected stage-1 codec + final stage-2 prior
#   last = budget-matched stage-1 codec     + final stage-2 prior
# (stage-2 best.pt and last.pt are written by the same save call and are
# expected identical; md5sums below put that on the record).
# Outputs land in the STAGE-2 run dir's Evaluation/ as
# crps_canonical_{best,last}_snap*.json / _summary.json + zslice figures.

set -euo pipefail
REPO=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
ARM_ROOT="${1:?usage: sbatch confild_eval_canonical.slurm <arm-save-root>}"

source ~/envs/jhtdb
cd "$REPO/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 MPLCONFIGDIR=/tmp TQDM_DISABLE=1

S1DIR=$(ls -d "${ARM_ROOT}"/Baseline_confild_Stage1_DemoN23_* 2>/dev/null | grep -v ABORTED | sort | tail -1)
S2DIR=$(ls -d "${ARM_ROOT}"/Baseline_confild_Stage2_DemoN23_* 2>/dev/null | grep -v ABORTED | sort | tail -1)
[ -n "${S1DIR}" ] || { echo "FATAL: no stage-1 run dir under ${ARM_ROOT}"; exit 1; }
[ -n "${S2DIR}" ] || { echo "FATAL: no stage-2 run dir under ${ARM_ROOT} (stage 2 never started?)"; exit 1; }
echo "node       : $(hostname)"
echo "stage1 dir : ${S1DIR}"
echo "stage2 dir : ${S2DIR}"
md5sum "${S1DIR}/best.pt" "${S1DIR}/last.pt" "${S2DIR}/best.pt" "${S2DIR}/last.pt" || true

for SEL in best last; do
  echo "=== canonical eval: ${SEL} ==="
  python confild_eval_unified.py \
    --stage1-ckpt "${S1DIR}/${SEL}.pt" \
    --stage2-ckpt "${S2DIR}/${SEL}.pt" \
    --out-dir "${S2DIR}" \
    --tag "canonical_${SEL}" \
    --seed 0 --op-seed 1000 --n-snapshots 50 --K 8 \
    --cond-fields 0 2 --n-obs 19531 19531 \
    --dps-scale 1.0 --steps 1000 --row-chunk 4
done
echo "=== eval done ==="
