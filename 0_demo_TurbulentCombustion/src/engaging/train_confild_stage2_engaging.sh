#!/bin/bash
#SBATCH --job-name=confild_s2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --mem=96G
# CoNFiLD STAGE 2 (latent diffusion prior) for a finished stage-1 sweep arm.
#   ARM=sweep1024|sweep2048|sweep4096|sweep8192|strict2048
#
# WHY STAGE 2 IS THE LEVER (measured 2026-09-02). The stage-1 codec ceiling is
# NOT what limits CoNFiLD's reconstruction quality. For the latent-1024
# architecture the codec reaches Uy 0.326 / p 0.301, while the finished
# pipeline delivers Uy 1.089 / p 1.487 — worse than predicting the training
# mean. That 3-5x loss on the UNOBSERVED channels happens entirely in stage-2
# DPS-guided sampling, so that is where the headroom is.
#
# The stage-1 checkpoint is discovered by CoNFiLDAdapter._stage1_checkpoint_path
# via find_latest_run_dir on this arm's save_root — which is why every arm has
# its own save_root. Stage 2 also runs the +/-10% parameter_budget check, so a
# width-256 2048/4096 arm is REFUSED here by design; strict2048 passes at
# 6,474,165 (-0.49%). Set BUDGET_ENFORCE=0 to override when parameter count is
# explicitly not a constraint.
#
# Budget accounting mirrors stage 1: wallclock_budget_s is measured from PROCESS
# start and never accumulates across resumes, so this subtracts time already
# spent (per-process maxima of elapsed_seconds in the stage-2 history) and
# passes only the remainder. Resubmit the identical command after a preemption.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
source ~/envs/phycoflow          # loads cuda + venv; see ~/envs/phycoflow
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
ARM=${ARM:?set ARM=sweep1024|sweep2048|sweep4096|sweep8192|strict2048}
BUDGET_TOTAL=${BUDGET_TOTAL:-19800}
case $ARM in
  sweep1024)  ROOT=sweep_ld1024_hf256 ;;
  sweep2048)  ROOT=sweep_ld2048_hf256 ;;
  sweep4096)  ROOT=sweep_ld4096_hf256 ;;
  sweep8192)  ROOT=sweep_ld8192_hf256 ;;
  strict2048) ROOT=strict_ld2048_hf144 ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
DATA=$DEMO/Dataset/JHU_4cubes_stride100.h5
L=$DEMO/src/train_confild_${ARM}_stage2_${SLURM_JOB_ID}.log
echo "host=$(hostname) arm=$ARM stage=2 start=$(date)" > $L

BASE=$DEMO/Save_TrainedModel/JHU/baseline_confild/$ROOT
S1=$(ls -d $BASE/Baseline_confild_Stage1_DemoN23_* 2>/dev/null | tail -1)
if [ -z "${S1:-}" ] || { [ ! -f "$S1/last.pt" ] && [ ! -f "$S1/best.pt" ]; }; then
  echo "GUARD: no finished stage-1 run for $ARM — train stage 1 first" >> $L; exit 3
fi
echo "stage1=$S1" >> $L

# ---- remaining stage-2 budget = total - already consumed -------------------
S2=$(ls -d $BASE/Baseline_confild_Stage2_DemoN23_* 2>/dev/null | tail -1)
CONSUMED=0
if [ -n "${S2:-}" ] && [ -f "$S2/history.jsonl" ]; then
  CONSUMED=$(python - "$S2/history.jsonl" <<'PY'
import json, sys
total = 0.0; prev = 0.0
for line in open(sys.argv[1]):
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
fi
BUDGET=$((BUDGET_TOTAL - CONSUMED))
echo "budget_total=$BUDGET_TOTAL consumed=$CONSUMED remaining=$BUDGET stage2_dir=${S2:-NEW}" >> $L
if [ "$BUDGET" -le 60 ]; then
  echo "stage-2 budget already spent for $ARM, nothing to do" >> $L; exit 0
fi

STAGE=/tmp/$USER/confild_s2_${SLURM_JOB_ID}; mkdir -p $STAGE
SZ=$(stat -c %s $DATA); AVAIL=$(df -B1 --output=avail /tmp | tail -1); RUNDATA=$DATA
if [ "$AVAIL" -gt $((SZ + 20000000000)) ]; then
  cp $DATA $STAGE/ && DSZ=$(stat -c %s $STAGE/$(basename $DATA)) || DSZ=0
  if [ "$DSZ" = "$SZ" ]; then RUNDATA=$STAGE/$(basename $DATA); else rm -f $STAGE/$(basename $DATA); fi
fi
echo "data=$RUNDATA" >> $L

SRC_CFG=$DEMO/Save_config/config_baseline_CoNFiLD_xcube_${ARM}.yaml
CFG=$DEMO/Save_config/confild_${ARM}_s2_eng.yaml
sed -e "s|$DEMO/Dataset/JHU_4cubes_stride100.h5|$RUNDATA|g" \
    -e "s|^\(      wallclock_budget_s: \)19800|\1$BUDGET|" \
    $SRC_CFG > $CFG
if [ "${BUDGET_ENFORCE:-1}" = "0" ]; then
  sed -i "s|^\(    enforce: \).*|\1false|" $CFG
  echo "parameter_budget.enforce disabled by BUDGET_ENFORCE=0" >> $L
fi
# PRIOR CAPACITY. The UNet prior is 1-D conv, so its size depends on
# num_channels/channel_mult and NOT on model_image_size — every sweep arm ran
# the SAME 1,441,217-parameter prior while the latent it models grew 1024
# -> 8192. Set PRIOR_CH to give the prior capacity matched to the latent.
if [ -n "${PRIOR_CH:-}" ]; then
  sed -i "s|^\(      num_channels: \).*|\1$PRIOR_CH|" $CFG
  echo "prior num_channels overridden to $PRIOR_CH" >> $L
fi
grep -nE "num_channels|wallclock_budget_s|enforce" $CFG >> $L
CUDA_VISIBLE_DEVICES=0 python -u train_Gen_Baseline.py --config $CFG \
    --training-stage 2 --reload >> $L 2>&1
RC=$?
rm -rf $STAGE
echo "end=$(date) rc=$RC" >> $L
exit $RC
