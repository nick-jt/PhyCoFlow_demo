#!/bin/bash
#SBATCH --job-name=crps_all
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
LFM=$STM/JHU/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN23_20260818_153527
SEN=$STM/JHU/baseline_senseiver/Baseline_senseiver_Stage1_DemoN31_20260818_083446
L=eval_crps_all_${SLURM_JOB_ID}.log
export ENSEMBLE_K=8
# Same held-out snapshots used for every other comparison in the paper.
for S in 0 1 3 12 14 23 28 36; do
  echo "##### LATENTFM snap $S" >> $L
  ENSEMBLE_OUT=$LFM/Evaluation/crps_snap${S}.json \
  python evaluate_Gen_Baseline.py --config $CFG/config_baseline_Gen_xcube_aug.yaml \
    --run-dir $LFM --training-stage 2 --split val --snapshot-index $S 2>&1 \
    | grep -E "\[ensemble\]" >> $L
done
for S in 0 1 3 12 14 23 28 36; do
  echo "##### SENSEIVER snap $S" >> $L
  ENSEMBLE_OUT=$SEN/Evaluation/crps_snap${S}.json \
  python evaluate_Det_Baseline.py --config $CFG/config_baseline_Det_xcube_aug.yaml \
    --run-dir $SEN --split val --snapshot-index $S 2>&1 \
    | grep -E "\[ensemble\]" >> $L
done
OURS=$STM/JHU/pointcloud_ffm/iclr_jhu_xcube_aug_DemoN15_20260818_083446
for NS in 2 4; do
  echo "##### OURS n_steps=$NS" >> $L
  python ensemble_eval.py --run-dir $OURS --ckpt best.pt \
      --K 8 --n-steps $NS --n-snapshots 8 \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --out $OURS/Evaluation/crps_nfe${NS}_K8.json 2>&1 \
      | grep -E "rel_l2_mean|rel_l2_single|  crps|spread_error|coverage_" >> $L
done
echo "ALL DONE" >> $L
