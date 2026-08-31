#!/bin/bash
#SBATCH --job-name=sen_xsmoke
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# SMOKE TEST for the Senseiver+xattn (local cross-attention) SECONDARY arm AND the
# sweep selection patches (fixed-mask TUNE-split val, SubsetView, patience
# bookkeeping): 20 epochs of training (DemoN99, throwaway), then a 3-snapshot
# eval through eval_senseiver_sweep_wrap.py (canonical draw + splits JSON;
# fingerprint gate fires only if snap 29 is drawn).  Gates the three full-arm
# launches.

set -u
set -o pipefail

export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1

export BASELINE_MAX_HOURS=0          # epochs=20 terminates the run
export SEN_VAL_FIXED=1
export SEN_VAL_SEED=0
export SEN_VAL_NOBS=19531
export SEN_VAL_SUBSET=odd
export SEN_ES_PATIENCE_EPOCHS=800
export SEN_LOCAL_IDW=0
export SEN_LOCAL_XATTN=1
export SEN_IDW_K=8

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
SWEEP=/home/ntricard/.claude/jobs/3ac3fd02/tmp/senseiver_sweep
CFG=$SWEEP/config_baseline_Senseiver_iclr_xattn_smoke.yaml
LOG=smoke_senseiver_xattn_${SLURM_JOB_ID}.log
SRC_H5=/projects/ammoniacomb/generative_reconstruction/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5

STAGE_DIR=/tmp/${USER}/jhu_${SLURM_JOB_ID}
RUN_CFG=$CFG
mkdir -p "$STAGE_DIR"
if cp "$SRC_H5" "$STAGE_DIR/" 2>>"$LOG"; then
  LOCAL_H5="$STAGE_DIR/$(basename $SRC_H5)"
  if [ "$(stat -c %s "$LOCAL_H5")" = "$(stat -c %s "$SRC_H5")" ]; then
    RUN_CFG=$STAGE_DIR/config_run.yaml
    sed "s#^\( *data_path: \).*#\1\"$LOCAL_H5\"#" "$CFG" > "$RUN_CFG"
    echo "[stage] OK -> $LOCAL_H5" >> "$LOG"
  fi
fi
cleanup() { rm -rf "$STAGE_DIR"; }
trap cleanup EXIT

echo "[smoke] training 20 epochs..." >> "$LOG"
CUDA_VISIBLE_DEVICES=0 python -u "$SWEEP/train_det_sweep.py" --config "$RUN_CFG" >> "$LOG" 2>&1
tstatus=$?
echo "train exit status: $tstatus" >> "$LOG"
if [ $tstatus -ne 0 ]; then exit $tstatus; fi

RUN_DIR=$(ls -d "$ROOT"/Save_TrainedModel/JHU/baseline_senseiver/Baseline_senseiver_Stage1_DemoN98_* 2>/dev/null | tail -1)
echo "[smoke] run dir: $RUN_DIR" >> "$LOG"

# ---- numeric sanity gate: finite decreasing loss, gate params present -----
python - "$LOG" "$RUN_DIR" >> "$LOG" 2>&1 <<'PYEOF'
import math, re, sys
import torch
log, run_dir = sys.argv[1], sys.argv[2]
losses = [float(m.group(1)) for m in
          re.finditer(r"\[train\] epoch=\d+ loss=([0-9.e+-]+)", open(log).read())]
assert len(losses) >= 20, f"only {len(losses)} train epochs logged"
assert all(math.isfinite(x) for x in losses), "non-finite train loss"
assert min(losses[-3:]) < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-3:]}"
ck = torch.load(f"{run_dir}/best.pt", map_location="cpu", weights_only=False)
gate = ck["model"]["local_gate"]
print(f"[smoke-gate] train loss {losses[0]:.4e} -> {losses[-1]:.4e} over {len(losses)} epochs")
print(f"[smoke-gate] local_gate after 20 epochs: {gate.tolist()}")
print("[smoke-gate] SANITY PASSED")
PYEOF
gstatus=$?
echo "sanity exit status: $gstatus" >> "$LOG"
if [ $gstatus -ne 0 ]; then exit $gstatus; fi
echo "[smoke] eval (3 snapshots, no sweep)..." >> "$LOG"
# The selection-time patches (fixed val masks, TUNE odd subset) must NOT leak
# into the canonical eval path: the eval seeds per ORIGINAL snapshot id.
env -u SEN_VAL_FIXED -u SEN_VAL_SUBSET -u SEN_ES_PATIENCE_EPOCHS JHU_AUGMENT= \
python -u "$SWEEP/eval_senseiver_sweep_wrap.py" \
  --run-dir "$RUN_DIR" --ckpt best.pt --split val \
  --seed 0 --n-snapshots 3 --n-obs 19531 --sweep \
  --out "$RUN_DIR/Evaluation/smoke_eval_best.json" >> "$LOG" 2>&1
estatus=$?
echo "eval exit status: $estatus" >> "$LOG"
exit $estatus
