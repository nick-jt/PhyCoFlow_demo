#!/bin/bash
#SBATCH --job-name=scaling_bench
#SBATCH --time=5:00:00
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
N29=../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
OUT=$N29/Evaluation
mkdir -p $OUT
L=scaling_bench_${SLURM_JOB_ID}.log
# Our full-field inference, NFE=2 (rectified-flow operating point), ODE cache on.
python benchmark_cost.py --mode scale --run-dir $N29 \
    --resolutions 64 125 192 256 384 512 768 1024 \
    --n-steps 2 --out $OUT/scaling_ours_infer.json >> $L 2>&1
# ConvAE3D inference (encode+decode) — lower bound on latent-FM sampling cost.
python benchmark_cost.py --mode grid \
    --resolutions 64 125 192 256 384 512 768 1024 \
    --out $OUT/scaling_convae_infer.json >> $L 2>&1
# ConvAE3D TRAINING step (stage-1 lower bound), batch 1 (generous).
python benchmark_cost.py --mode grid_train --iters 5 \
    --resolutions 64 125 192 256 320 384 512 \
    --out $OUT/scaling_convae_train.json >> $L 2>&1
# Our training step vs dataset resolution (flat by construction; measured).
python benchmark_cost.py --mode train_scale --run-dir $N29 \
    --resolutions 64 125 256 512 1024 --iters 5 \
    --out $OUT/scaling_ours_train.json >> $L 2>&1
echo "ALL DONE" >> $L
