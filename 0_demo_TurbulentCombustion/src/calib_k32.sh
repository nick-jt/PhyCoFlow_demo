#!/bin/bash
#SBATCH --job-name=calib_k32
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=200G
#SBATCH --array=0-1
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RD=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
NOBS_LIST=(19531 195312)
NOBS=${NOBS_LIST[$SLURM_ARRAY_TASK_ID]}
L=calib_k32_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
echo "task=$SLURM_ARRAY_TASK_ID n_obs=$NOBS K=32 nfe=4 host=$(hostname) start=$(date)" > $L
python ensemble_eval.py --run-dir "$RD" --ckpt best.pt \
    --K 32 --n-steps 4 --n-snapshots 50 --no-clamp \
    --cond-fields 0 2 --n-obs $NOBS $NOBS \
    --query-subset 400000 --chunk 262144 --fig-every 25 \
    --out "$RD/Evaluation/calib_sweep_nfe4_n${NOBS}_K32.json" >> $L 2>&1
RC=$?
echo "end=$(date) rc=$RC" >> $L
exit $RC
