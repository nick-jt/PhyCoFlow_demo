#!/bin/bash
#SBATCH --job-name=cnf_canon
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=node2906
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --mem=96G
# Canonical conditional (DPS) eval of a finished CoNFiLD stage-2 arm — the
# numbers directly comparable to FLEET_SUMMARY_TABLE (per-channel rel-L2 at 1%
# sensors, Ux/Uz observed, Uy/p unobserved, n=50, K=8).
#   ARM=sweep1024|sweep2048|sweep4096|sweep8192|strict2048
#
# H100 REQUIRED, NOT OPTIONAL. This runs at the canonical operating point
# (snap 29, seed 0, cond_fields [0,2], n_obs [19531,19531]) where
# check_canonical_fingerprint bites: torch.randperm on CUDA is not portable
# across GPU SKU, and the canonical sensor draw (sensors=39062,
# idx_sum=37987162596) is H100-SXM-bound. An H200 would abort here by design.
# Engaging has exactly one h100 node (node2906), hence the nodelist pin.
#
# Ported from src/confild_eval_canonical.slurm (origin-only: gpu-h100/f2pde
# partition, ~/envs/jhtdb, hardcoded origin paths).
set -u
export PYTHONUNBUFFERED=1
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
# confild_eval_unified.py inserts CONFILD_ROOT on sys.path at import time and
# defaults to the origin /projects path, which does not exist here.
export CONFILD_ROOT=/orcd/scratch/orcd/002/ntricard/baselines/CoNFiLD
source ~/envs/phycoflow          # loads cuda + venv; see ~/envs/phycoflow
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
ARM=${ARM:?set ARM=sweep1024|sweep2048|sweep4096|sweep8192|strict2048}
case $ARM in
  sweep1024)  ROOT=sweep_ld1024_hf256 ;;
  sweep2048)  ROOT=sweep_ld2048_hf256 ;;
  sweep4096)  ROOT=sweep_ld4096_hf256 ;;
  sweep8192)  ROOT=sweep_ld8192_hf256 ;;
  strict2048) ROOT=strict_ld2048_hf144 ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
BASE=$DEMO/Save_TrainedModel/JHU/baseline_confild/$ROOT
S1DIR=$(ls -d $BASE/Baseline_confild_Stage1_DemoN23_* 2>/dev/null | tail -1)
S2DIR=$(ls -d $BASE/Baseline_confild_Stage2_DemoN23_* 2>/dev/null | tail -1)
L=$DEMO/src/eval_confild_${ARM}_canonical_${SLURM_JOB_ID}.log
echo "host=$(hostname) arm=$ARM start=$(date)" > $L
echo "stage1=${S1DIR:-NONE}" >> $L; echo "stage2=${S2DIR:-NONE}" >> $L
if [ -z "${S1DIR:-}" ] || [ -z "${S2DIR:-}" ]; then
  echo "GUARD: need BOTH a stage-1 and a stage-2 run for $ARM" >> $L; exit 3
fi
RC=0
for SEL in last best; do
  [ -f "$S1DIR/$SEL.pt" ] && [ -f "$S2DIR/$SEL.pt" ] || { echo "skip $SEL (missing ckpt)" >> $L; continue; }
  echo "=== canonical eval: $SEL ===" >> $L
  python -u confild_eval_unified.py \
    --stage1-ckpt "$S1DIR/$SEL.pt" \
    --stage2-ckpt "$S2DIR/$SEL.pt" \
    --out-dir "$S2DIR" \
    --data "$DEMO/Dataset/JHU_4cubes_stride100.h5" \
    --tag "canonical_${SEL}" \
    --seed 0 --op-seed 1000 --n-snapshots 50 --K 8 \
    --cond-fields 0 2 --n-obs 19531 19531 \
    --dps-scale 1.0 --steps 1000 --row-chunk 4 >> $L 2>&1 || RC=$?
done
echo "end=$(date) rc=$RC" >> $L
exit $RC
