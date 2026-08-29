#!/bin/bash
#SBATCH --job-name=stats_base
#SBATCH --time=12:00:00
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
LFM=$STM/JHU/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN23_20260818_153527
SEN=$STM/JHU/baseline_senseiver/Baseline_senseiver_Stage1_DemoN31_20260818_083446
N15=$STM/JHU/pointcloud_ffm/iclr_jhu_xcube_aug_DemoN15_20260818_083446
L=eval_stats_base_${SLURM_JOB_ID}.log
export ENSEMBLE_K=8
# All 50 val snapshots for both baselines (skip the 8 already evaluated: JSONs exist)
for S in $(seq 0 49); do
  if [ ! -f "$LFM/Evaluation/crps_snap${S}.json" ]; then
    echo "##### LATENTFM snap $S" >> $L
    ENSEMBLE_OUT=$LFM/Evaluation/crps_snap${S}.json \
    python evaluate_Gen_Baseline.py --config $CFG/config_baseline_Gen_xcube_aug.yaml \
      --run-dir $LFM --training-stage 2 --split val --snapshot-index $S 2>&1 \
      | grep -E "\[ensemble\]" >> $L
  fi
done
for S in $(seq 0 49); do
  if [ ! -f "$SEN/Evaluation/crps_snap${S}.json" ]; then
    echo "##### SENSEIVER snap $S" >> $L
    ENSEMBLE_OUT=$SEN/Evaluation/crps_snap${S}.json \
    python evaluate_Det_Baseline.py --config $CFG/config_baseline_Det_xcube_aug.yaml \
      --run-dir $SEN --split val --snapshot-index $S 2>&1 \
      | grep -E "\[ensemble\]" >> $L
  fi
done
# K=32 calibration check for LFM on the original 8 snapshots
export ENSEMBLE_K=32
for S in 0 1 3 12 14 23 28 36; do
  echo "##### LATENTFM K32 snap $S" >> $L
  ENSEMBLE_OUT=$LFM/Evaluation/crps_K32_snap${S}.json \
  python evaluate_Gen_Baseline.py --config $CFG/config_baseline_Gen_xcube_aug.yaml \
    --run-dir $LFM --training-stage 2 --split val --snapshot-index $S 2>&1 \
    | grep -E "\[ensemble\]" >> $L
done
export ENSEMBLE_K=8
# N15 (lambda=0) expanded snapshots at NFE4 for the ablation row's statistics
echo "##### N15 all-snaps nfe4" >> $L
python ensemble_eval.py --run-dir $N15 --ckpt best.pt \
    --K 8 --n-steps 4 --n-snapshots 50 \
    --cond-fields 0 2 --n-obs 19531 19531 \
    --out $N15/Evaluation/crps_nfe4_K8_all50.json >> $L 2>&1
echo "ALL DONE" >> $L
