#!/bin/bash
#SBATCH --job-name=dpn_eval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# DeepONet evaluation on the shared ICLR protocol.
#
# MUST run on a compute node: torch.randperm on CUDA is not portable across
# GPU SKU, and the canonical fingerprint (snap=29 sensors=39062
# idx_sum=37987162596) only reproduces on the H100 SXM compute nodes.
#
# No trailing `echo "exit status: $?"` without an explicit `exit $status`:
# a bare echo would force exit 0 and a crash would report COMPLETED.

set -u
set -o pipefail

export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=octahedral          # train-split only inside the dataset
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

RUN_DIR=${RUN_DIR:?set RUN_DIR}
CKPT=${CKPT:-best.pt}
NSNAP=${NSNAP:-50}
EXTRA=${EXTRA:-}
LOG=eval_deeponet_$(basename "$RUN_DIR")_${CKPT%.pt}_${SLURM_JOB_ID}.log

python -u eval_deeponet_iclr.py \
  --run-dir "$RUN_DIR" \
  --ckpt "$CKPT" \
  --split val \
  --seed 0 \
  --n-obs 19531 \
  --n-snapshots "$NSNAP" \
  --cond-fields 0 2 \
  --sweep 1953 3906 7812 15625 19531 \
  --sweep-snapshots 8 \
  --out "$RUN_DIR/Evaluation/iclr_protocol_eval_${CKPT%.pt}.json" \
  $EXTRA >> "$LOG" 2>&1
status=$?
echo "python exit status: $status" >> "$LOG"
exit $status
