#!/bin/bash
#SBATCH --job-name=dpnpp_train
#SBATCH --time=21:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# DeepONet++ (structured branch), ICLR cross-cube JHU arm.  One script serves
# all p-sweep arms: pass CFG and TAG via --export.  Early stopping at the
# fixed-validation (TUNE odd cube-3, canonical seed-0 masks) optimum;
# BASELINE_MAX_HOURS=19.5 is the wall backstop.
#
# Chains, on success: canonical 50-snapshot evals (best.pt + budget.pt) and
# the TUNE-odd / TEST-even subset evals on best.pt.

set -u
set -o pipefail

export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Persistent workers: same measured rationale as the vanilla DeepONet arm
# (worker respawn was ~3.5 s of a ~6.2 s epoch = 2.7x compute handicap).
export JHU_PERSISTENT_WORKERS=1
export TQDM_DISABLE=1
export BASELINE_MAX_HOURS=${BASELINE_MAX_HOURS:-19.5}
export BASELINE_ARCHIVE_N=${BASELINE_ARCHIVE_N:-10}

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

TMPD=/home/ntricard/.claude/jobs/3ac3fd02/tmp/deeponetpp
CFG=${CFG:-$TMPD/config_baseline_DeepONetPP_iclr.yaml}
TAG=${TAG:-}
LOG=$TMPD/train_deeponetpp${TAG}_${SLURM_JOB_ID}.log
SRC_H5=/projects/ammoniacomb/generative_reconstruction/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5

# Stage the dataset onto node-local NVMe (same rationale as every other arm:
# sibling jobs share Lustre; contention measured to eat >50% of wall budget).
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

echo "config=$RUN_CFG tag=$TAG budget=${BASELINE_MAX_HOURS}h" >> "$LOG"
CUDA_VISIBLE_DEVICES=0 python -u "$TMPD/train_deeponetpp.py" \
  --config "$RUN_CFG" --tag "$TAG" >> "$LOG" 2>&1
status=$?
echo "python exit status: $status" >> "$LOG"

# NAMED subset tokens, not comma lists: sbatch --export splits on commas and
# would truncate "1,3,5,..." to "1" (bug found 2026-08-30, jobs 17117990/91).
# eval_deeponetpp.py expands TUNE_ODD / TEST_EVEN itself.
TUNE_IDS=TUNE_ODD
TEST_IDS=TEST_EVEN

if [ "$status" -eq 0 ]; then
  RUN_DIR=$(grep -m1 '^\[run\] dir=' "$LOG" | sed 's#^\[run\] dir=##')
  echo "[chain] run_dir=$RUN_DIR" >> "$LOG"
  # The staged config records the /tmp data path; rewrite the saved
  # run_config.yaml back to the shared path for the evaluation jobs.
  if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/run_config.yaml" ]; then
    sed -i "s#^\( *data_path: \).*#\1$SRC_H5#" "$RUN_DIR/run_config.yaml"
    # canonical 50-snapshot rows, best + budget
    for CK in best.pt budget.pt; do
      if [ -f "$RUN_DIR/$CK" ]; then
        jid=$(sbatch --parsable \
              --export=ALL,RUN_DIR="$RUN_DIR",CKPT=$CK "$TMPD/eval_deeponetpp.sh")
        echo "[chain] submitted canonical eval $CK job=$jid" >> "$LOG"
      else
        echo "[chain] MISSING $RUN_DIR/$CK -- no eval submitted" >> "$LOG"
      fi
    done
    # TUNE (odd, selection evidence) and TEST (even, primary) subsets, best.pt
    if [ -f "$RUN_DIR/best.pt" ]; then
      jid=$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR",CKPT=best.pt,NSNAP=25,SNAPIDS="$TUNE_IDS",LABEL=TUNE_odd,OUTNAME=iclr_protocol_eval_best_TUNEodd \
            "$TMPD/eval_deeponetpp.sh")
      echo "[chain] submitted TUNE-odd eval job=$jid" >> "$LOG"
      jid=$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR",CKPT=best.pt,NSNAP=25,SNAPIDS="$TEST_IDS",LABEL=TEST_even,OUTNAME=iclr_protocol_eval_best_TESTeven \
            "$TMPD/eval_deeponetpp.sh")
      echo "[chain] submitted TEST-even eval job=$jid" >> "$LOG"
    fi
  fi
fi
exit $status
