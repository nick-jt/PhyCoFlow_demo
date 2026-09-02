#!/bin/bash
#SBATCH --job-name=confild_orc
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --mem=64G
# Frozen-decoder oracle (codec ceiling) for one CoNFiLD sweep arm.
# evaluate_confild_stage1.py is un-gated and SKU-independent (no canonical
# fingerprint involved) -> h200 is fine. Identical settings across arms:
# script defaults (snaps 150 151 153 162, 3000 steps, 3 restarts, seed 123).
# Submit per arm, held on the training chain tail:
#   sbatch --dependency=afterany:<tail_jobid> --export=ALL,ARM=<arm> eval_confild_stage1_engaging.sh
# Evaluates last.pt (budget-matched, primary) and best.pt (train-selected,
# secondary). Results: <run_dir>/Evaluation/stage1_oracle_{last,best}/stage1_auto_decode.json
set -u
export PYTHONUNBUFFERED=1
module load cuda/12.4.0
source ~/envs/phycoflow          # campaign env; see ~/envs/phycoflow
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
RD=$(ls -d $DEMO/Save_TrainedModel/JHU/baseline_confild/$ROOT/Baseline_confild_Stage1_DemoN23_* 2>/dev/null | tail -1)
L=$DEMO/src/eval_confild_${ARM}_oracle_${SLURM_JOB_ID}.log
echo "host=$(hostname) arm=$ARM run_dir=${RD:-NONE} start=$(date)" > $L
if [ -z "${RD:-}" ] || [ ! -f "$RD/last.pt" ]; then
  echo "GUARD: no stage-1 run/last.pt for $ARM — training incomplete, aborting" >> $L
  exit 3
fi
# Training runs on a preemptable partition, so afterany can fire on a job that
# was preempted mid-budget. Scoring an under-trained arm would silently corrupt
# the sweep comparison (all arms must be compared at the SAME spent budget), so
# require the full 48600 s to have been consumed before evaluating.
BUDGET_TOTAL=${BUDGET_TOTAL:-48600}
CONSUMED=$(python - "$RD/history.jsonl" <<'PY'
import json, sys
total = 0.0; prev = 0.0
try:
    lines = open(sys.argv[1])
except OSError:
    print(0); raise SystemExit
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        continue
    if "elapsed_seconds" not in rec:
        continue
    e = float(rec["elapsed_seconds"])
    if e < prev:
        total += prev
    prev = e
print(int(total + prev))
PY
)
echo "budget_consumed=$CONSUMED / $BUDGET_TOTAL" >> $L
if [ "$CONSUMED" -lt $((BUDGET_TOTAL - 300)) ]; then
  echo "GUARD: $ARM consumed only ${CONSUMED}s of ${BUDGET_TOTAL}s (preempted?) —" >> $L
  echo "       resubmit training to finish the budget, then re-run this eval." >> $L
  exit 4
fi
RC=0
for CK in last best; do
  [ -f "$RD/$CK.pt" ] || { echo "no $CK.pt, skipping" >> $L; continue; }
  CUDA_VISIBLE_DEVICES=0 python -u evaluate_confild_stage1.py \
      --checkpoint "$RD/$CK.pt" \
      --data $DEMO/Dataset/JHU_4cubes_stride100.h5 \
      --confild-root /orcd/scratch/orcd/002/ntricard/baselines/CoNFiLD \
      --out-dir "$RD/Evaluation/stage1_oracle_$CK" \
      --snapshot-indices 150 151 153 162 \
      --steps 3000 --points-per-step 65536 --fit-fraction 0.8 \
      --restarts 3 --lr 1e-4 --seed 123 >> $L 2>&1 || RC=$?
done
echo "end=$(date) rc=$RC" >> $L
exit $RC
