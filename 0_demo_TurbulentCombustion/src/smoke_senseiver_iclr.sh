#!/bin/bash
#SBATCH --job-name=sen_smoke
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# Senseiver, ICLR cross-cube JHU arm.  Post-fidelity-audit configuration.
#
# NOTE: no trailing `echo "exit status: $?"` -- the real exit code of the
# python process is what SLURM records, so a crash reports FAILED, not
# COMPLETED.

set -u
set -o pipefail

# Shared protocol
export JHU_SPLIT_MODE=block
export JHU_SPLIT_GAP=0
export JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wall-clock budget matched across baselines (~20 h on one H100).
export BASELINE_MAX_HOURS=${BASELINE_MAX_HOURS:-0.08}

source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"

ROOT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
CFG=$ROOT/Save_config/config_baseline_Senseiver_iclr_smoke.yaml
LOG=smoke_senseiver_iclr_${SLURM_JOB_ID}.log

echo "config=$CFG budget=${BASELINE_MAX_HOURS}h" >> "$LOG"
CUDA_VISIBLE_DEVICES=0 python -u train_Det_Baseline.py --config "$CFG" >> "$LOG" 2>&1
status=$?
echo "python exit status: $status" >> "$LOG"
exit $status
