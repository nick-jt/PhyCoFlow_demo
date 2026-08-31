#!/bin/bash
#SBATCH --job-name=sen_iclr
#SBATCH --time=21:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# Senseiver, ICLR cross-cube JHU arm.  Post-fidelity-audit configuration.
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

# Wall-clock budget matched across baselines (~20 h on one H100), and ~10
# archived checkpoints across the final 5% for checkpoint-noise measurement.
export BASELINE_MAX_HOURS=${BASELINE_MAX_HOURS:-19.5}
export BASELINE_ARCHIVE_N=${BASELINE_ARCHIVE_N:-10}

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
CFG=$ROOT/Save_config/config_baseline_Senseiver_iclr.yaml
LOG=train_senseiver_iclr_${SLURM_JOB_ID}.log
SRC_H5=/projects/ammoniacomb/generative_reconstruction/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5

# ---------------------------------------------------------------------------
# Stage the dataset onto node-local NVMe.
#
# One training epoch re-reads all 150 train snapshots = 4.7 GB, i.e. ~780 MB/s
# sustained off shared Lustre, and ~10 sibling jobs hit the same filesystem.
# That contention stalled an earlier run for >12 min at a stretch, which eats
# the wall-clock budget the comparison is supposed to hold fixed.  6.3 GB onto
# a 3.2 TB local disk removes the coupling entirely.  Falls back to the shared
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

echo "config=$RUN_CFG budget=${BASELINE_MAX_HOURS}h" >> "$LOG"
CUDA_VISIBLE_DEVICES=0 python -u train_Det_Baseline.py --config "$RUN_CFG" >> "$LOG" 2>&1
status=$?
echo "python exit status: $status" >> "$LOG"
exit $status
