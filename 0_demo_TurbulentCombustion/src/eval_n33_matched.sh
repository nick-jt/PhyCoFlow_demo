#!/bin/bash
#SBATCH --job-name=n33_matched
#SBATCH --time=12:00:00
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
L=eval_n33_matched_${SLURM_JOB_ID}.log
N33=$(ls -d ../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_cq_DemoN33_* | tail -1)
mkdir -p "$N33/Evaluation"
echo "N33=$N33" >> $L
# Same protocol as the frozen model's all-50 eval (K=8, NFE 2 & 4, 50 snapshots)
for NS in 4 2; do
  echo "##### N33 all-50 n_steps=$NS" >> $L
  python ensemble_eval.py --run-dir "$N33" --ckpt best.pt \
      --K 8 --n-steps $NS --n-snapshots 50 \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --out "$N33/Evaluation/matched_nfe${NS}_K8_all50.json" >> $L 2>&1
done
echo "ALL DONE" >> $L
