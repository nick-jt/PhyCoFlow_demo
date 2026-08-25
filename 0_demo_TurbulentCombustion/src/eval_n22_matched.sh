#!/bin/bash
#SBATCH --job-name=n22_matched
#SBATCH --time=3:00:00
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
L=eval_n22_matched_${SLURM_JOB_ID}.log
for NS in 2 4; do
  echo "##### N22 n_steps=$NS" >> $L
  python ensemble_eval.py --run-dir ../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec_DemoN22_20260819_122008 --ckpt best.pt \
      --K 8 --n-steps $NS --n-snapshots 8 \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --out ../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec_DemoN22_20260819_122008/Evaluation/matched_nfe${NS}_K8.json 2>&1 \
      | grep -E "rel_l2_mean|rel_l2_single|  crps|spread_error|coverage_" >> $L
done
echo "ALL DONE" >> $L
