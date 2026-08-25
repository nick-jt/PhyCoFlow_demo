#!/bin/bash
#SBATCH --job-name=nfe_sweep
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
RUN=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_aug_DemoN15_20260818_083446
L=eval_nfe_sweep_${SLURM_JOB_ID}.log
# Accuracy/cost frontier: does the 16-step reconstruction cost buy anything?
for NS in 2 4 8 16; do
  echo "########## n_steps=$NS" >> $L
  python ensemble_eval.py --run-dir $RUN --ckpt best.pt \
      --K 4 --n-steps $NS --n-snapshots 4 \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --out $RUN/Evaluation/nfe_${NS}.json >> $L 2>&1
done
# Wall-clock for a single full field at each NFE
for NS in 2 4 8 16; do
  echo "########## timing n_steps=$NS" >> $L
  python benchmark_cost.py --mode scale --run-dir $RUN --ckpt best.pt \
      --resolutions 125 --n-steps $NS --chunk 131072 >> $L 2>&1
done
echo "ALL DONE" >> $L
