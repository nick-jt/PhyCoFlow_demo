#!/bin/bash
#SBATCH --job-name=calib_hi16
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=200G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RD=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
L=calib_nfe16_hi_${SLURM_JOB_ID}.log
echo "n_obs=390625 nfe=16 host=$(hostname) start=$(date)" > $L
# Single query chunk (subset == chunk): the sensor-side encoding over 781k
# tokens is recomputed per chunk per ODE step, so one chunk halves the cost
# of the 2-chunk 400k configuration. Still an exact subsample of the field.
python ensemble_eval.py --run-dir "$RD" --ckpt best.pt \
    --K 8 --n-steps 16 --n-snapshots 50 --no-clamp \
    --cond-fields 0 2 --n-obs 390625 390625 \
    --query-subset 262144 --chunk 262144 --fig-every 25 \
    --out "$RD/Evaluation/calib_sweep_nfe16_n390625_K8.json" >> $L 2>&1
RC=$?
echo "end=$(date) rc=$RC" >> $L
exit $RC
