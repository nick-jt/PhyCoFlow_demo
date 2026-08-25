#!/bin/bash
#SBATCH --job-name=scaling_baselines
#SBATCH --time=4:00:00
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
L=scaling_baselines_${SLURM_JOB_ID}.log
python benchmark_scaling_baselines.py --mode senseiver \
    --resolutions 64 125 192 256 384 512 768 1024 \
    --out $OUT/scaling_senseiver.json >> $L 2>&1
python benchmark_scaling_baselines.py --mode gen4turb \
    --resolutions 64 120 128 192 256 320 384 512 \
    --out $OUT/scaling_gen4turb.json >> $L 2>&1
echo "ALL DONE" >> $L
