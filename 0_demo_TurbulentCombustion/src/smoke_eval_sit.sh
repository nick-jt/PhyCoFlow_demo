#!/bin/bash
#SBATCH --job-name=sit_esmoke
#SBATCH --time=00:25:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH --partition=gpu-h100 --account=f2pde --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
# Exercises the SAME code path and the SAME launcher flags as the chained
# evals, against the already-complete 7.47M reference. Writes to a scratch
# out-dir so it cannot collide with the real Evaluation_seeded_* results.
R=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/baseline_sit/Baseline_sit_Stage1_DemoN41_20260827_145610
O=/home/ntricard/.claude/jobs/3ac3fd02/tmp/eval_smoke_${SLURM_JOB_ID}
L=smoke_eval_${SLURM_JOB_ID}.log
python eval_sit_ensemble.py \
  --run-dir "$R" --ckpt best --split val \
  --seed 0 --op-seed 1000 --n-snapshots 1 --K 2 \
  --cond-fields 0 2 --n-obs 19531 19531 \
  --shard 0 --num-shards 1 --out-dir "$O" > $L 2>&1
echo "rc=$?" >> $L
echo "--- artifacts ---" >> $L
ls -l "$O" >> $L 2>&1
