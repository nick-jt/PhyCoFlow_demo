#!/bin/bash
#SBATCH --job-name=sen_eval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# Usage: sbatch eval_senseiver_iclr.sh <run_dir> [ckpt]
# Evaluates on all 50 held-out (cube 3) snapshots with seeded sensor draws and
# writes Evaluation/iclr_protocol_eval_<ckpt>.json.

set -u
set -o pipefail

export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=            # never augment the evaluation split
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

# $1 may be a literal run dir OR a glob; the newest match wins.  This lets the
# eval be chained with --dependency before the training run dir exists.
RUN_DIR=$(ls -d $1 2>/dev/null | tail -1)
if [ -z "$RUN_DIR" ]; then echo "no run dir matching: $1" >&2; exit 2; fi
CKPT=${2:-best.pt}
echo "resolved run dir: $RUN_DIR"
LOG=eval_senseiver_iclr_${SLURM_JOB_ID}.log

# Canonical protocol, matching ensemble_eval.py's driver exactly:
#   --seed 0, 50 snapshots, FIXED 19531 sensors per observed channel (1%).
# (K=8 is not applicable: Senseiver is deterministic, K=1 and CRPS = MAE.)
python -u eval_senseiver_iclr.py \
  --run-dir "$RUN_DIR" --ckpt "$CKPT" --split val \
  --seed 0 --n-snapshots 50 --n-obs 19531 \
  --out "$RUN_DIR/Evaluation/iclr_protocol_eval_${CKPT%.pt}.json" >> "$LOG" 2>&1
status=$?
echo "python exit status: $status" >> "$LOG"
exit $status
