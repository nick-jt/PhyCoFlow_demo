#!/bin/bash
#SBATCH --job-name=scaling_fixup
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
OUT=../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100/Evaluation
L=scaling_fixup_${SLURM_JOB_ID}.log
# Re-time ConvAE inference with warmup (memory unchanged)
python benchmark_cost.py --mode grid \
    --resolutions 64 125 192 256 384 512 768 1024 \
    --out $OUT/scaling_convae_infer.json >> $L 2>&1
# Measure (not extrapolate) the ConvAE training wall
python benchmark_cost.py --mode grid_train --iters 3 \
    --resolutions 640 768 \
    --out $OUT/scaling_convae_train_wall.json >> $L 2>&1
echo "ALL DONE" >> $L
