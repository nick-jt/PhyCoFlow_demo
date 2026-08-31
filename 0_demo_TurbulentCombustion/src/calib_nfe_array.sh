#!/bin/bash
#SBATCH --job-name=calib_nfe
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=200G
#SBATCH --array=0-7%4
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RD=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
mkdir -p "$RD/Evaluation"
# H1: does the spread floor persist at NFE 16? Task i in 0-3 = NFE 16, 4-7 = NFE 4
# (matched reference at 50 snapshots -- the published sweep mixed 12- and
# 8-snapshot sets across densities, so it is not internally matched).
NOBS_LIST=(1953 19531 195312 390625 1953 19531 195312 390625)
NFE_LIST=(16 16 16 16 4 4 4 4)
NOBS=${NOBS_LIST[$SLURM_ARRAY_TASK_ID]}
NFE=${NFE_LIST[$SLURM_ARRAY_TASK_ID]}
L=calib_nfe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
echo "task=$SLURM_ARRAY_TASK_ID n_obs=$NOBS nfe=$NFE host=$(hostname) start=$(date)" > $L
# Query subset of 400k points: the GL_rbf velocity is pointwise in the query
# (no cross-query mixing, see Model.py forward), and the RFF prior weights do
# not depend on N, so a fixed random subset is an exact subsample of the
# full-field ensemble -- 20M points over 50 snapshots.
python ensemble_eval.py --run-dir "$RD" --ckpt best.pt \
    --K 8 --n-steps $NFE --n-snapshots 50 --no-clamp \
    --cond-fields 0 2 --n-obs $NOBS $NOBS \
    --query-subset 400000 --chunk 262144 --fig-every 25 \
    --out "$RD/Evaluation/calib_sweep_nfe${NFE}_n${NOBS}_K8.json" >> $L 2>&1
RC=$?
echo "end=$(date) rc=$RC" >> $L
exit $RC
