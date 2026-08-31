#!/bin/bash
#SBATCH --job-name=sen_sweval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# Usage: sbatch eval_senseiver_sweep.sh <run_dir_or_glob> [ckpt]
# Canonical Senseiver ICLR eval (eval_senseiver_iclr.py, fingerprint gate +
# compute-node guard intact) via eval_senseiver_sweep_wrap.py, which ALSO
# writes iclr_protocol_eval_<ckpt>_splits.json with TEST(even)/TUNE(odd)
# aggregates.  Export SEN_LOCAL_IDW=1 at submit time for the
# Senseiver+local arm (its checkpoints carry the local_gate parameter).

set -u
set -o pipefail

export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=            # never augment the evaluation split
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1
export SEN_LOCAL_IDW=${SEN_LOCAL_IDW:-0}
export SEN_LOCAL_XATTN=${SEN_LOCAL_XATTN:-0}
export SEN_IDW_K=${SEN_IDW_K:-8}

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

SWEEP=/home/ntricard/.claude/jobs/3ac3fd02/tmp/senseiver_sweep

# $1 may be a literal run dir OR a glob; the newest match wins.  This lets the
# eval be chained with --dependency before the training run dir exists.
RUN_DIR=$(ls -d $1 2>/dev/null | tail -1)
if [ -z "$RUN_DIR" ]; then echo "no run dir matching: $1" >&2; exit 2; fi
CKPT=${2:-best.pt}
echo "resolved run dir: $RUN_DIR  ckpt: $CKPT  local_idw: $SEN_LOCAL_IDW"
LOG=eval_senseiver_sweep_${SLURM_JOB_ID}.log

# Canonical protocol, matching ensemble_eval.py's driver exactly:
#   --seed 0, 50 snapshots, FIXED 19531 sensors per observed channel (1%).
python -u "$SWEEP/eval_senseiver_sweep_wrap.py" \
  --run-dir "$RUN_DIR" --ckpt "$CKPT" --split val \
  --seed 0 --n-snapshots 50 --n-obs 19531 \
  --out "$RUN_DIR/Evaluation/iclr_protocol_eval_${CKPT%.pt}.json" >> "$LOG" 2>&1
status=$?
echo "python exit status: $status" >> "$LOG"
exit $status
