#!/bin/bash
#SBATCH --job-name=eval_noclamp
#SBATCH --time=6:00:00
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
L=eval_noclamp_${SLURM_JOB_ID}.log
# Fairness check. The standard path runs clamp_hard=True, which scatters the
# TRUE observed values into our prediction at every ODE step, so our sensor
# consistency is 0 by construction while every baseline is scored without any
# such projection (Senseiver measures 0.71). Quantify what the projection is
# worth to us, per channel, so the paper can either report the unprojected
# number or grant the same projection to the baselines.
RD=$(ls -d ../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_* | tail -1)
for NS in 4; do
  python ensemble_eval.py --run-dir "$RD" --ckpt best.pt \
      --K 8 --n-steps $NS --n-snapshots 50 --no-clamp \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --out "$RD/Evaluation/matched_nfe${NS}_K8_all50_NOCLAMP.json" >> $L 2>&1
done
echo "ALL DONE" >> $L
