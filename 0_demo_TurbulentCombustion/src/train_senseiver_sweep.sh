#!/bin/bash
#SBATCH --time=21:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# Senseiver improvement-arm launcher (PLAN_IMPROVE_2026-08-30.md section 1).
# Usage: sbatch -J <name> train_senseiver_sweep.sh <config.yaml> [extra env
# via --export or exported at submit time].  Identical NVMe-staging pattern to
# train_senseiver_iclr.sh; trains via train_det_sweep.py (sen_sweep_fixes
# monkey-patches: fixed-mask TUNE-split validation, patience early stop, and
# -- when SEN_LOCAL_IDW=1 -- the IDW residual path).
#
# NOTE: no trailing `echo "exit status: $?"` -- SLURM records the real python
# exit code, so a crash reports FAILED rather than COMPLETED.

set -u
set -o pipefail

# Shared protocol
export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1

# Wall-clock budget backstop (early stopping is the intended terminator).
export BASELINE_MAX_HOURS=${BASELINE_MAX_HOURS:-19.5}
export BASELINE_ARCHIVE_N=${BASELINE_ARCHIVE_N:-10}

# Selection policy (sen_sweep_fixes.py): fixed seed-0 val masks at 1%,
# TUNE split (odd cube-3 indices), patience 800 epochs on the fixed metric.
export SEN_VAL_FIXED=${SEN_VAL_FIXED:-1}
export SEN_VAL_SEED=${SEN_VAL_SEED:-0}
export SEN_VAL_NOBS=${SEN_VAL_NOBS:-19531}
export SEN_VAL_SUBSET=${SEN_VAL_SUBSET:-odd}
export SEN_ES_PATIENCE_EPOCHS=${SEN_ES_PATIENCE_EPOCHS:-800}
# Enhanced variant only; default off for the capacity arms.
export SEN_LOCAL_IDW=${SEN_LOCAL_IDW:-0}
export SEN_LOCAL_XATTN=${SEN_LOCAL_XATTN:-0}
export SEN_IDW_K=${SEN_IDW_K:-8}

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

SWEEP=/home/ntricard/.claude/jobs/3ac3fd02/tmp/senseiver_sweep
CFG=${1:?usage: sbatch -J name train_senseiver_sweep.sh <config.yaml>}
LOG=train_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
SRC_H5=/projects/ammoniacomb/generative_reconstruction/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5

# ---------------------------------------------------------------------------
# Stage the dataset onto node-local NVMe (audit 16/16g: unstaged, 45% duty
# cycle under fleet contention; staged, 88.6%).  Falls back to the shared
# path if staging fails, so this can only help.
# ---------------------------------------------------------------------------
STAGE_DIR=/tmp/${USER}/jhu_${SLURM_JOB_ID}
RUN_CFG=$CFG
mkdir -p "$STAGE_DIR"
echo "[stage] copying $(du -h $SRC_H5 | cut -f1) to $STAGE_DIR ..." >> "$LOG"
if cp "$SRC_H5" "$STAGE_DIR/" 2>>"$LOG"; then
  LOCAL_H5="$STAGE_DIR/$(basename $SRC_H5)"
  if [ "$(stat -c %s "$LOCAL_H5")" = "$(stat -c %s "$SRC_H5")" ]; then
    RUN_CFG=$STAGE_DIR/config_run.yaml
    sed "s#^\( *data_path: \).*#\1\"$LOCAL_H5\"#" "$CFG" > "$RUN_CFG"
    echo "[stage] OK -> $LOCAL_H5" >> "$LOG"
    grep -n "data_path" "$RUN_CFG" >> "$LOG"
  else
    echo "[stage] SIZE MISMATCH, falling back to shared path" >> "$LOG"
  fi
else
  echo "[stage] copy FAILED, falling back to shared path" >> "$LOG"
fi

cleanup() { rm -rf "$STAGE_DIR"; }
trap cleanup EXIT

echo "config=$RUN_CFG budget=${BASELINE_MAX_HOURS}h patience=${SEN_ES_PATIENCE_EPOCHS}ep local_idw=${SEN_LOCAL_IDW}" >> "$LOG"
CUDA_VISIBLE_DEVICES=0 python -u "$SWEEP/train_det_sweep.py" --config "$RUN_CFG" >> "$LOG" 2>&1
status=$?
echo "python exit status: $status" >> "$LOG"
exit $status
