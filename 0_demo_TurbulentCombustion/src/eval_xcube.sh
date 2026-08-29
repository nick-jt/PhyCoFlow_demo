#!/bin/bash
#SBATCH --job-name=eval_xcube
#SBATCH --time=4:00:00
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
# Held-out cube 3 only. Augmentation is train-split-only, so it is inert here,
# but the split vars must match training for the val set to be cube 3.
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

STM=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel
CFG=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config
LOG=eval_xcube_${SLURM_JOB_ID}.log

run_ours () {  # name, run_dir
  echo "===== OURS $1" >> "$LOG"
  python ensemble_eval.py --run-dir "$2" --ckpt best.pt \
      --K 8 --n-steps 16 --n-snapshots 8 \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --out "$2/Evaluation/xcube_ensemble.json" >> "$LOG" 2>&1
  echo "exit=$?" >> "$LOG"
}

mkdir -p "$STM/JHU/pointcloud_ffm/iclr_jhu_xcube_aug_DemoN15_20260818_083446/Evaluation"
mkdir -p "$STM/JHU/pointcloud_ffm/iclr_jhu_xcube_DemoN4_20260817_155642/Evaluation"
run_ours "aug"    "$STM/JHU/pointcloud_ffm/iclr_jhu_xcube_aug_DemoN15_20260818_083446"
run_ours "no-aug" "$STM/JHU/pointcloud_ffm/iclr_jhu_xcube_DemoN4_20260817_155642"

echo "===== SENSEIVER aug" >> "$LOG"
python evaluate_Det_Baseline.py \
    --config "$CFG/config_baseline_Det_xcube_aug.yaml" \
    --run-dir "$STM/JHU/baseline_senseiver/Baseline_senseiver_Stage1_DemoN31_20260818_083446" \
    --split val --snapshot-index 0 >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"

echo "===== LATENT-FM aug" >> "$LOG"
python evaluate_Gen_Baseline.py \
    --config "$CFG/config_baseline_Gen_xcube_aug.yaml" \
    --run-dir "$STM/JHU/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN23_20260818_153527" \
    --training-stage 2 --split val --snapshot-index 0 >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"

echo "ALL DONE" >> "$LOG"
