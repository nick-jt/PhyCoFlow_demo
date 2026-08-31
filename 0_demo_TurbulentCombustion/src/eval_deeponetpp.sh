#!/bin/bash
#SBATCH --job-name=dpnpp_eval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# DeepONet++ evaluation = canonical eval_deeponet_iclr.py via the
# eval_deeponetpp.py wrapper (builder swap only).  MUST run on a compute node
# (canonical fingerprint snap=29 sensors=39062 idx_sum=37987162596 is
# H100-SXM-specific).  Optional env:
#   SNAPIDS  comma-separated val indices (TUNE odd / TEST even subsets)
#   LABEL    subset label stamped into the JSON
#   OUTNAME  output JSON basename (default derived from CKPT)

set -u
set -o pipefail

export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

TMPD=/home/ntricard/.claude/jobs/3ac3fd02/tmp/deeponetpp
RUN_DIR=${RUN_DIR:?set RUN_DIR}
CKPT=${CKPT:-best.pt}
NSNAP=${NSNAP:-50}
SNAPIDS=${SNAPIDS:-}
LABEL=${LABEL:-}
OUTNAME=${OUTNAME:-iclr_protocol_eval_${CKPT%.pt}}
EXTRA=${EXTRA:-}
LOG=$TMPD/eval_deeponetpp_$(basename "$RUN_DIR")_${OUTNAME}_${SLURM_JOB_ID}.log

DPNPP_SNAPSHOT_IDS="$SNAPIDS" DPNPP_SUBSET_LABEL="$LABEL" \
python -u "$TMPD/eval_deeponetpp.py" \
  --run-dir "$RUN_DIR" \
  --ckpt "$CKPT" \
  --split val \
  --seed 0 \
  --n-obs 19531 \
  --n-snapshots "$NSNAP" \
  --cond-fields 0 2 \
  --sweep 1953 3906 7812 15625 19531 \
  --sweep-snapshots 8 \
  --out "$RUN_DIR/Evaluation/${OUTNAME}.json" \
  $EXTRA >> "$LOG" 2>&1
status=$?
echo "python exit status: $status" >> "$LOG"
exit $status
