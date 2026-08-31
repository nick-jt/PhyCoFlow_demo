#!/bin/bash
#SBATCH --job-name=dpn_iclr
#SBATCH --time=21:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# DeepONet, ICLR cross-cube JHU arm.  Deterministic operator learner: plain
# MSE regression, no generative wrapper, K=1 at evaluation.
#
# NOTE: the launcher ends with an explicit `exit $status`, never a bare
# `echo "exit status: $?"` -- the latter forces exit 0 and a crash would be
# recorded as COMPLETED.

set -u
set -o pipefail

# Shared protocol
export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Compute-hours, not wall-clock, is the fleet's budget unit: wall-clock was
# found to measure filesystem and DataLoader luck rather than compute.  With
# the default (persistent_workers=False) PyTorch tears down and respawns the
# 4 worker processes EVERY epoch; measured on this arm that is ~3.5 s of a
# ~6.2 s epoch, i.e. a duty cycle of 0.33 against 0.87-0.99 for every other
# baseline -- a 2.7x compute handicap with no methodological content.
# JHU_PERSISTENT_WORKERS=1 (opt-in, model_baseline.py:4869) removes it and
# changes no other method's behaviour.
export JHU_PERSISTENT_WORKERS=1
export TQDM_DISABLE=1

# Wall-clock budget matched across baselines, plus ~10 archived checkpoints
# across the final 5% for checkpoint-noise measurement.  Both the archive
# window and the figure cadence are driven by MEASURED elapsed wall-clock,
# not by a projected epoch rate.
export BASELINE_MAX_HOURS=${BASELINE_MAX_HOURS:-19.5}
export BASELINE_ARCHIVE_N=${BASELINE_ARCHIVE_N:-10}

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
CFG=$ROOT/Save_config/config_baseline_DeepONet_iclr.yaml
LOG=train_deeponet_${SLURM_JOB_ID}.log
SRC_H5=/projects/ammoniacomb/generative_reconstruction/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5

# Stage the dataset onto node-local NVMe (same rationale as the Senseiver arm:
# ~10 sibling jobs share Lustre and contention was measured to eat >50% of the
# wall-clock budget).  Falls back to the shared path if staging fails.
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
  else
    echo "[stage] SIZE MISMATCH, falling back to shared path" >> "$LOG"
  fi
else
  echo "[stage] copy FAILED, falling back to shared path" >> "$LOG"
fi
cleanup() { rm -rf "$STAGE_DIR"; }
trap cleanup EXIT

echo "config=$RUN_CFG budget=${BASELINE_MAX_HOURS}h" >> "$LOG"
CUDA_VISIBLE_DEVICES=0 python -u train_deeponet.py --config "$RUN_CFG" >> "$LOG" 2>&1
status=$?
echo "python exit status: $status" >> "$LOG"

# Chain the two reported evaluations only if training really succeeded.
if [ "$status" -eq 0 ]; then
  RUN_DIR=$(grep -m1 '^\[run\] dir=' "$LOG" | sed 's#^\[run\] dir=##')
  echo "[chain] run_dir=$RUN_DIR" >> "$LOG"
  # The staged config records the /tmp data path, which disappears with the
  # node.  Rewrite the saved run_config.yaml back to the shared path so the
  # evaluation job can open the dataset.
  if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/run_config.yaml" ]; then
    sed -i "s#^\( *data_path: \).*#\1$SRC_H5#" "$RUN_DIR/run_config.yaml"
    for CK in budget.pt best.pt; do
      if [ -f "$RUN_DIR/$CK" ]; then
        jid=$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR",CKPT=$CK eval_deeponet.sh)
        echo "[chain] submitted eval $CK job=$jid" >> "$LOG"
      else
        echo "[chain] MISSING $RUN_DIR/$CK -- no eval submitted" >> "$LOG"
      fi
    done
  fi
fi
exit $status
