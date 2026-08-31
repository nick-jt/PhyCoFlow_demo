#!/bin/bash
#SBATCH --job-name=dpn_smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# End-to-end smoke for the DeepONet arm, on a COMPUTE node.
# Arms, by RUNNING rather than by reading:
#   1. training loop, budget stop -> budget.pt, tail archive, figures, cost json
#   2. evaluation on all 50 held-out snapshots -> canonical sensor fingerprint
#   3. evaluation with the nearest-sensor floor and the sensor sweep enabled
# A short budget (0.04 h = 144 s) is used so the archive/figure/budget paths all
# fire inside the smoke AND the steady-state duty cycle is measured over enough
# epochs to be meaningful (the first epoch carries CUDA + worker warm-up).

set -u
set -o pipefail

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
export BASELINE_MAX_HOURS=0.04
export BASELINE_ARCHIVE_N=3
export BASELINE_ARCHIVE_TAIL_FRAC=0.35

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
CFG=$ROOT/Save_config/config_baseline_DeepONet_iclr.yaml
LOG=smoke_deeponet_${SLURM_JOB_ID}.log

python -u train_deeponet.py --config "$CFG" --max-epochs 200 --tag _SMOKE >> "$LOG" 2>&1
status=$?
echo "train exit status: $status" >> "$LOG"
[ "$status" -ne 0 ] && exit $status

RUN_DIR=$(grep -m1 '^\[run\] dir=' "$LOG" | sed 's#^\[run\] dir=##')
echo "[smoke] run_dir=$RUN_DIR" >> "$LOG"
ls -1 "$RUN_DIR" "$RUN_DIR/archive" "$RUN_DIR/Evaluation/figures" >> "$LOG" 2>&1

# (2) fingerprint gate over all 50 snapshots, floor and sweep off for speed
python -u eval_deeponet_iclr.py --run-dir "$RUN_DIR" --ckpt budget.pt \
  --split val --seed 0 --n-obs 19531 --n-snapshots 50 --cond-fields 0 2 \
  --skip-nn-floor --skip-sweep \
  --out "$RUN_DIR/Evaluation/smoke_fingerprint.json" >> "$LOG" 2>&1
status=$?
echo "eval-fingerprint exit status: $status" >> "$LOG"
[ "$status" -ne 0 ] && exit $status

# (3) full path: nearest-sensor floor + sensor sweep, on a few snapshots
python -u eval_deeponet_iclr.py --run-dir "$RUN_DIR" --ckpt budget.pt \
  --split val --seed 0 --n-obs 19531 --n-snapshots 3 --cond-fields 0 2 \
  --sweep 1953 19531 --sweep-snapshots 2 \
  --out "$RUN_DIR/Evaluation/smoke_full.json" >> "$LOG" 2>&1
status=$?
echo "eval-full exit status: $status" >> "$LOG"
exit $status
