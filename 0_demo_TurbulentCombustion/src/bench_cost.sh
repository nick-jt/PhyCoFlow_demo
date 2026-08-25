#!/bin/bash
#SBATCH --job-name=bench_cost
#SBATCH --time=2:00:00
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
OUT=$RUN/Evaluation
L=bench_cost_${SLURM_JOB_ID}.log
mkdir -p $OUT
#echo "########## GRID (ConvAE3D) ##########" >> $L
#python benchmark_cost.py --mode grid --out $OUT/cost_grid.json >> $L 2>&1
echo "########## OURS scale ##########" >> $L
python benchmark_cost.py --mode scale --run-dir $RUN --ckpt best.pt \
    --n-steps 16 --chunk 131072 --out $OUT/cost_scale.json >> $L 2>&1
#echo "########## OURS train step ##########" >> $L
#python benchmark_cost.py --mode train --run-dir $RUN --ckpt best.pt \
    --batch 20 --n-query 39062 --out $OUT/cost_train.json >> $L 2>&1
echo "ALL DONE" >> $L
