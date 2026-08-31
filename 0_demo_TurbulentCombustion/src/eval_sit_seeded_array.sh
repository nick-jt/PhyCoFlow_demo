#!/bin/bash
#SBATCH --job-name=sit_seval
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --array=0-4
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RUN=${SIT_RUN:?set SIT_RUN}
CK=${SIT_CKPT:-best}
L=eval_sitseed_${CK}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
# Full stream is kept: no grep in the pipeline, so tracebacks survive intact.
python eval_sit_ensemble.py \
  --run-dir "$RUN" --ckpt "$CK" --split val \
  --seed 0 --op-seed 1000 --n-snapshots 50 --K 8 \
  --cond-fields 0 2 --n-obs 19531 19531 \
  --shard $SLURM_ARRAY_TASK_ID --num-shards 5 > $L 2>&1
echo "rc=$?" >> $L
