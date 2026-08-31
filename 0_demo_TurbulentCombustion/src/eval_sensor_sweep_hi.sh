#!/bin/bash
#SBATCH --job-name=sens_hi
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=200G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=eval_sensor_hi_${SLURM_JOB_ID}.log
RD=$(ls -d ../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_* | tail -1)
mkdir -p "$RD/Evaluation"
# Push sensor density across the gradient-resolution threshold.
# Recovering Uy from Ux,Uz needs d/dx and d/dz, which needs spacing below ~3 eta.
#   density  spacing   spacing/eta
#     5%      2.71 cells   6.0    - constraint NOT resolvable
#    10%      2.15         4.7    - not resolvable
#    20%      1.71         3.8    - not resolvable
#    40%      1.36         3.0    - THRESHOLD
#    60%      1.19         2.6    - resolvable
# Prediction: Uy stays flat below 40%. If it stays flat at 40-60% too, the model
# cannot use the constraint even when the data supports it -- an architectural
# limit, not an information limit. Observed channels are the control: if they
# keep improving, the model is handling the out-of-distribution density fine
# (training max was 19531 = 1%).
for NOBS in 195312 390625 781250 1171875; do
  echo "##### n_obs=$NOBS per observed field" >> $L
  python ensemble_eval.py --run-dir "$RD" --ckpt best.pt \
      --K 8 --n-steps 4 --n-snapshots 8 --no-clamp \
      --cond-fields 0 2 --n-obs $NOBS $NOBS --fig-every 4 \
      --out "$RD/Evaluation/sensor_sweep_n${NOBS}.json" >> $L 2>&1 \
    || echo "  [n_obs=$NOBS FAILED - likely OOM; continuing]" >> $L
done
echo "ALL DONE" >> $L
