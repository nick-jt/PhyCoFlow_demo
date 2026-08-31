#!/bin/bash
#SBATCH --job-name=sen_arch
#SBATCH --time=01:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --account=f2pde
#SBATCH --mem=16G
set -u
set -o pipefail
source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"
L=archive_checkpoints_${SLURM_JOB_ID}.log
# 10 snapshots x 351 s = 58.5 min = the last 5% of a 19.5 h budget.
python -u archive_checkpoints.py \
  --run-dir "$1" --n 10 --interval-s 351 >> "$L" 2>&1
status=$?
echo "python exit status: $status" >> "$L"
exit $status
