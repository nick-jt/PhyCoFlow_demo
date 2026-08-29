#!/bin/bash
#SBATCH --job-name=eval_xc_match
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# ==========================================================================
# LEGACY QUARANTINE (baseline audit 2026-08-29): this launcher predates the
# canonical sensor-draw protocol (helpers.build_sparse_condition under
# torch.manual_seed(seed*777+snap) on an H100 SXM compute node; fingerprint
# snap=29 sensors=39062 idx_sum=37987162596). Its numbers are NOT
# comparable to canonical results and must not enter the paper table.
echo "WARNING: LEGACY NON-CANONICAL SENSOR DRAW -- numbers not comparable to the canonical fingerprint. Set ALLOW_LEGACY_EVAL=1 to run."
[ -z "${ALLOW_LEGACY_EVAL:-}" ] && exit 1
# ==========================================================================

set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

CFG=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config
STM=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel
LOG=eval_xcube_matched_${SLURM_JOB_ID}.log

# Same 8 val snapshots the posterior-ensemble eval used, so the baseline
# numbers are averaged over the identical held-out fields.
SNAPS="0 1 3 12 14 23 28 36"

for s in $SNAPS; do
  echo "===== SENSEIVER snap $s" >> "$LOG"
  python evaluate_Det_Baseline.py \
      --config "$CFG/config_baseline_Det_xcube_aug.yaml" \
      --run-dir "$STM/JHU/baseline_senseiver/Baseline_senseiver_Stage1_DemoN31_20260818_083446" \
      --split val --snapshot-index $s 2>&1 | grep -E "field_[0-3]:|obs_rel_l2_SenConsis:" >> "$LOG"
done

for s in $SNAPS; do
  echo "===== LATENTFM snap $s" >> "$LOG"
  python evaluate_Gen_Baseline.py \
      --config "$CFG/config_baseline_Gen_xcube_aug.yaml" \
      --run-dir "$STM/JHU/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN23_20260818_153527" \
      --training-stage 2 --split val --snapshot-index $s 2>&1 | grep -E "field_[0-3]:|obs_rel_l2_SenConsis:" >> "$LOG"
done

echo "ALL DONE" >> "$LOG"
