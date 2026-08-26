#!/bin/bash
#SBATCH --job-name=stats_ours
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
L=eval_stats_ours_${SLURM_JOB_ID}.log
N29=../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
mkdir -p "$N29/Evaluation"
# Expanded statistics: ALL 50 held-out cube-3 snapshots (seed 0 -> superset of the 8 used before)
for NS in 4 2; do
  echo "##### N29 all-snaps n_steps=$NS" >> $L
  python ensemble_eval.py --run-dir "$N29" --ckpt best.pt \
      --K 8 --n-steps $NS --n-snapshots 50 \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --out "$N29/Evaluation/matched_nfe${NS}_K8_all50.json" >> $L 2>&1
done
# Ensemble-size sensitivity: K=32 on the original 8 snapshots (calibration metrics vs K)
echo "##### N29 K=32 nfe4" >> $L
python ensemble_eval.py --run-dir "$N29" --ckpt best.pt \
    --K 32 --n-steps 4 --n-snapshots 8 \
    --cond-fields 0 2 --n-obs 19531 19531 \
    --out "$N29/Evaluation/matched_nfe4_K32.json" >> $L 2>&1
echo "ALL DONE" >> $L
