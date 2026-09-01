#!/bin/bash
#SBATCH --job-name=confild_swp
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --mem=96G
# CoNFiLD stage-1 latent-dim sweep on MIT Engaging (2026-08-31). Select arm:
#   ARM=sweep1024|sweep2048|sweep4096|strict2048
#
# Runs on mit_preemptable (2-day limit) so the whole 48600 s stage-1 budget
# fits in ONE job instead of chained 6 h segments — mit_normal_gpu's 6 h cap
# forced 3 segments, and each segment re-queued behind a saturated h200 pool
# (measured 2026-09-01: ~1 day of queue wait per segment). Preemption is cheap
# here because the trainer resumes from last.pt via --reload.
#
# BUDGET ACCOUNTING. confild_upstream_training.py:486 starts the budget clock at
# PROCESS start and never accumulates across resumes, so a naive resume would
# grant a fresh 48600 s and overshoot the protocol. This script therefore sums
# the wall-clock already consumed by previous segments (per-process maxima of
# `elapsed_seconds` in history.jsonl, which resets on each new process) and
# passes only the remainder to the trainer. Total stays 48600 s no matter how
# many times the job is preempted. Exits 0 when the budget is already spent.
#
# Submit one job per arm (NOT a chain):
#   sbatch --export=ALL,ARM=<arm> train_confild_sweep_engaging.sh
# If preempted, resubmit the identical command — it resumes and finishes the
# remaining budget.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
module load cuda/12.4.0
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
ARM=${ARM:?set ARM=sweep1024|sweep2048|sweep4096|strict2048}
BUDGET_TOTAL=${BUDGET_TOTAL:-48600}
case $ARM in
  sweep1024)  ROOT=sweep_ld1024_hf256 ;;
  sweep2048)  ROOT=sweep_ld2048_hf256 ;;
  sweep4096)  ROOT=sweep_ld4096_hf256 ;;
  strict2048) ROOT=strict_ld2048_hf144 ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
DATA=$DEMO/Dataset/JHU_4cubes_stride100.h5
L=$DEMO/src/train_confild_${ARM}_eng_${SLURM_JOB_ID}.log
echo "host=$(hostname) arm=$ARM start=$(date)" > $L

# ---- remaining budget = total - already consumed --------------------------
RD=$(ls -d $DEMO/Save_TrainedModel/JHU/baseline_confild/$ROOT/Baseline_confild_Stage1_DemoN23_* 2>/dev/null | tail -1)
CONSUMED=0
if [ -n "${RD:-}" ] && [ -f "$RD/history.jsonl" ]; then
  CONSUMED=$(python - "$RD/history.jsonl" <<'PY'
import json, sys
total = 0.0; prev = 0.0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        continue          # truncated final line from a preempted process
    if "elapsed_seconds" not in rec:
        continue          # never treat a key-less record as a segment boundary
    e = float(rec["elapsed_seconds"])
    # elapsed_seconds restarts near 0 in each new process; a drop ends a segment
    if e < prev:
        total += prev
    prev = e
print(int(total + prev))
PY
)
fi
BUDGET=$((BUDGET_TOTAL - CONSUMED))
echo "budget_total=$BUDGET_TOTAL consumed=$CONSUMED remaining=$BUDGET run_dir=${RD:-NONE}" >> $L
if [ "$BUDGET" -le 60 ]; then
  echo "budget already spent — stage 1 complete for $ARM, nothing to do" >> $L
  exit 0
fi

STAGE=/tmp/$USER/confild_${SLURM_JOB_ID}; mkdir -p $STAGE
SZ=$(stat -c %s $DATA); AVAIL=$(df -B1 --output=avail /tmp | tail -1); RUNDATA=$DATA
if [ "$AVAIL" -gt $((SZ + 20000000000)) ]; then
  cp $DATA $STAGE/ && DSZ=$(stat -c %s $STAGE/$(basename $DATA)) || DSZ=0
  if [ "$DSZ" = "$SZ" ]; then RUNDATA=$STAGE/$(basename $DATA); else rm -f $STAGE/$(basename $DATA); fi
fi
echo "data=$RUNDATA" >> $L

SRC_CFG=$DEMO/Save_config/config_baseline_CoNFiLD_xcube_${ARM}.yaml
CFG=$DEMO/Save_config/confild_${ARM}_eng.yaml
sed -e "s|$DEMO/Dataset/JHU_4cubes_stride100.h5|$RUNDATA|g" \
    -e "s|^\(      wallclock_budget_s: \)48600|\1$BUDGET|" \
    $SRC_CFG > $CFG
grep -n "wallclock_budget_s" $CFG >> $L
CUDA_VISIBLE_DEVICES=0 python -u train_Gen_Baseline.py --config $CFG \
    --training-stage 1 --reload >> $L 2>&1
RC=$?
rm -rf $STAGE
echo "end=$(date) rc=$RC" >> $L
exit $RC
