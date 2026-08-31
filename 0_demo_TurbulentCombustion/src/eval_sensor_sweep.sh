#!/bin/bash
#SBATCH --job-name=sens_sweep
#SBATCH --time=10:00:00
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
L=eval_sensor_sweep_${SLURM_JOB_ID}.log
RD=$(ls -d ../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_* | tail -1)
mkdir -p "$RD/Evaluation"
# Does denser sampling of the OBSERVED channels improve the UNOBSERVED ones?
# Incompressibility ties Uy to Ux/Uz, and the pressure Poisson equation ties p
# to the whole velocity field, so the information exists. Whether the model
# exploits it is the question -- our divergence is 9x worse than DNS, which
# predicts a weak response. 1953=0.1%, 19531=1% (training max), beyond = OOD.
for NOBS in 1953 4883 9766 19531 39062 97656; do
  echo "##### n_obs=$NOBS per observed field" >> $L
  python ensemble_eval.py --run-dir "$RD" --ckpt best.pt \
      --K 8 --n-steps 4 --n-snapshots 12 --no-clamp \
      --cond-fields 0 2 --n-obs $NOBS $NOBS --fig-every 6 \
      --out "$RD/Evaluation/sensor_sweep_n${NOBS}.json" >> $L 2>&1
done
echo "ALL DONE" >> $L
