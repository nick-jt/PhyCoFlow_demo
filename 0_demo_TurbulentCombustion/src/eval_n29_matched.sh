#!/bin/bash
#SBATCH --job-name=n29_matched
#SBATCH --time=4:00:00
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
L=eval_n29_matched_${SLURM_JOB_ID}.log
N29=../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
N22=../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec_DemoN22_20260819_122008
mkdir -p "$N29/Evaluation" "$N22/Evaluation"
for NS in 2 4; do
  echo "##### N29 n_steps=$NS" >> $L
  python ensemble_eval.py --run-dir "$N29" --ckpt best.pt \
      --K 8 --n-steps $NS --n-snapshots 8 \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --out "$N29/Evaluation/matched_nfe${NS}_K8.json" >> $L 2>&1
done
# Regenerate N22 per-snapshot JSONs (lost in job 16434155: missing dir)
for NS in 2 4; do
  echo "##### N22 rerun n_steps=$NS" >> $L
  python ensemble_eval.py --run-dir "$N22" --ckpt best.pt \
      --K 8 --n-steps $NS --n-snapshots 8 \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --out "$N22/Evaluation/matched_nfe${NS}_K8.json" >> $L 2>&1
done
echo "ALL DONE" >> $L
