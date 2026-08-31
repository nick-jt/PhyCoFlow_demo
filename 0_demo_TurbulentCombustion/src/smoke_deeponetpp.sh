#!/bin/bash
#SBATCH --job-name=dpnpp_smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# End-to-end smoke for the DeepONet++ (structured branch) arm, COMPUTE node.
# Arms, by RUNNING rather than by reading:
#   0. parameter counts of all three p-arms are in the +/-10% band
#   1. 20-epoch training: loss decreasing, fixed-mask TUNE validation fires,
#      figure written, cost json
#   2. canonical eval wrapper WITH the TUNE/TEST subset override, snap 29
#      included -> the canonical fingerprint gate must fire and pass
#   3. canonical eval wrapper WITHOUT override (default rng.choice path)

set -u
set -o pipefail

export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_PERSISTENT_WORKERS=1
export TQDM_DISABLE=1
export BASELINE_MAX_HOURS=0.5        # never reached in 20 epochs; sets fig cadence
export BASELINE_ARCHIVE_N=0

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

TMPD=/home/ntricard/.claude/jobs/3ac3fd02/tmp/deeponetpp
LOG=$TMPD/smoke_deeponetpp_${SLURM_JOB_ID}.log

# (0) params in band, all three arms
python -u "$TMPD/deeponetpp.py" >> "$LOG" 2>&1

# (1) 20 epochs; eval_every lowered to 5 so the fixed-val path fires 4x
SMOKE_CFG=$TMPD/config_smoke_${SLURM_JOB_ID}.yaml
sed 's/^    eval_every: 40.*/    eval_every: 5/' \
    "$TMPD/config_baseline_DeepONetPP_iclr.yaml" > "$SMOKE_CFG"
python -u "$TMPD/train_deeponetpp.py" --config "$SMOKE_CFG" \
    --max-epochs 20 --tag _SMOKE >> "$LOG" 2>&1
status=$?
echo "train exit status: $status" >> "$LOG"
[ "$status" -ne 0 ] && exit $status

RUN_DIR=$(grep -m1 '^\[run\] dir=' "$LOG" | sed 's#^\[run\] dir=##')
echo "[smoke] run_dir=$RUN_DIR" >> "$LOG"
ls -1 "$RUN_DIR" "$RUN_DIR/Evaluation/figures" >> "$LOG" 2>&1

# (2) subset override incl. snap 29 -> canonical fingerprint gate must pass
DPNPP_SNAPSHOT_IDS="29,1,3" DPNPP_SUBSET_LABEL="smoke_subset" \
python -u "$TMPD/eval_deeponetpp.py" --run-dir "$RUN_DIR" --ckpt last.pt \
  --split val --seed 0 --n-obs 19531 --n-snapshots 3 --cond-fields 0 2 \
  --sweep 1953 19531 --sweep-snapshots 2 \
  --out "$RUN_DIR/Evaluation/smoke_subset.json" >> "$LOG" 2>&1
status=$?
echo "eval-subset exit status: $status" >> "$LOG"
[ "$status" -ne 0 ] && exit $status

# (3) default rng.choice path, floor+sweep off for speed
python -u "$TMPD/eval_deeponetpp.py" --run-dir "$RUN_DIR" --ckpt last.pt \
  --split val --seed 0 --n-obs 19531 --n-snapshots 50 --cond-fields 0 2 \
  --skip-nn-floor --skip-sweep \
  --out "$RUN_DIR/Evaluation/smoke_fingerprint.json" >> "$LOG" 2>&1
status=$?
echo "eval-default exit status: $status" >> "$LOG"
exit $status
