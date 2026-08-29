#!/bin/bash
#SBATCH --job-name=sit_m_eval
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
CFG=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config
SIT=$(readlink -f "$(ls -d ../Save_TrainedModel/JHU/baseline_sit/matched/Baseline_sit_Stage1_DemoN41_* | tail -1)")
CK=${SIT_CKPT:-best}      # best = validation-selected; last = budget-matched (epoch 6000)
L=eval_sitm_${CK}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
FULL=eval_sitm_${CK}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.full.log
echo "SIT=$SIT CKPT=$CK" >> $L
mkdir -p "$SIT/Evaluation_$CK"
export ENSEMBLE_K=8
for S in $(seq $((SLURM_ARRAY_TASK_ID * 10)) $((SLURM_ARRAY_TASK_ID * 10 + 9))); do
  OUT=$SIT/Evaluation_$CK/crps_snap${S}.json
  if [ -s "$OUT" ]; then echo "##### snap $S already present, skipping" >> $L; continue; fi
  echo "##### snap $S" >> $L
  # tee the FULL stream: piping straight into grep swallows traceback bodies
  # and hides the real failure. grep reads from the tee, nothing is lost.
  ENSEMBLE_OUT=$OUT \
  python evaluate_Gen_Baseline.py --config $CFG/config_baseline_SiT_xcube_matched.yaml \
    --run-dir $SIT --training-stage 1 --checkpoint-name $CK \
    --split val --snapshot-index $S 2>&1 \
    | tee -a $FULL | grep -aE "\[ensemble\]|Traceback|Error|Exception|error:" >> $L
done
N=$(ls "$SIT/Evaluation_$CK"/crps_snap*.json 2>/dev/null | wc -l)
echo "TASK $SLURM_ARRAY_TASK_ID DONE; $N/50 total metric files present" >> $L
