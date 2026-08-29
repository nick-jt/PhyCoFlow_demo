#!/bin/bash
#SBATCH --job-name=sit_eval
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --array=0-4

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
# evaluate_Gen_Baseline.py resolves a relative --run-dir against the repo root, not
# this directory, so the path must be absolute.
SIT=$(readlink -f "$(ls -d ../Save_TrainedModel/JHU/baseline_sit/Baseline_sit_Stage1_DemoN41_* | tail -1)")
L=eval_sit_arr_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
echo "SIT=$SIT" >> $L
mkdir -p "$SIT/Evaluation"
export ENSEMBLE_K=8
# 50 snapshots over 5 tasks; ~19 min each => ~3.2 h per task instead of 16 h serial.
for S in $(seq $((SLURM_ARRAY_TASK_ID * 10)) $((SLURM_ARRAY_TASK_ID * 10 + 9))); do
  OUT=$SIT/Evaluation/crps_snap${S}.json
  if [ -s "$OUT" ]; then echo "##### SIT snap $S: already present, skipping" >> $L; continue; fi
  echo "##### SIT snap $S" >> $L
  ENSEMBLE_OUT=$OUT \
  python evaluate_Gen_Baseline.py --config $CFG/config_baseline_SiT_xcube.yaml \
    --run-dir $SIT --training-stage 1 --split val --snapshot-index $S 2>&1 \
    | grep -aE "\[ensemble\]|Traceback|Error|Exception|error:" >> $L
done
N=$(ls "$SIT/Evaluation"/crps_snap*.json 2>/dev/null | wc -l)
echo "TASK $SLURM_ARRAY_TASK_ID DONE; $N/50 total metric files present" >> $L
