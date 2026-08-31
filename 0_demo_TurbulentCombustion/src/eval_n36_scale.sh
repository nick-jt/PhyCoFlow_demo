#!/bin/bash
#SBATCH --job-name=eval_n36
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=eval_n36_${SLURM_JOB_ID}.log
RD=../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_scale_DemoN29_20260827_014226
mkdir -p "$RD/Evaluation"
# Matched protocol against N29, unclamped so observed-channel numbers are not
# flattered by the observation projection. Validation block is bit-identical to
# N29's, so this isolates training-set size (150 -> 1200 snapshots).
for NS in 2 4; do
  echo "##### N36 scale n_steps=$NS" >> $L
  python ensemble_eval.py --run-dir "$RD" --ckpt best.pt \
      --K 8 --n-steps $NS --n-snapshots 50 --no-clamp \
      --cond-fields 0 2 --n-obs 19531 19531 --fig-every 10 \
      --out "$RD/Evaluation/matched_nfe${NS}_K8_all50_NOCLAMP.json" >> $L 2>&1
done
echo "ALL DONE" >> $L
