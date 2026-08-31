#!/bin/bash
#SBATCH --job-name=classical_anchor
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
OUT=../Save_TrainedModel/JHU/baseline_classical
L=classical_anchor_${SLURM_JOB_ID}.log
{
  nvidia-smi -L
  echo "##### sensor fingerprints (canonical helpers.build_sparse_condition, cuda:0)"
  python -u sensor_fingerprints.py --out $OUT/sensor_fingerprints.json
  echo "##### main table, 1% sensors per observed channel, 50 snapshots"
  python -u baseline_classical_jhu.py --methods constant kdtree idw gappy_pod \
      --n-obs 19531 --n-snapshots 50 --verify-percentile \
      --tag main_n19531 --out-dir $OUT
  echo "##### sensor-count sweep"
  python -u baseline_classical_jhu.py --methods kdtree idw gappy_pod \
      --n-obs 1953 4883 9766 19531 39062 97656 195312 --n-snapshots 50 \
      --tag sweep --out-dir $OUT
  echo "ALL DONE"
} >> $L 2>&1
