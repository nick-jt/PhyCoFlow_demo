#!/bin/bash
#SBATCH --job-name=calib_dump
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=200G
#SBATCH --array=0-2
# Route-2 recalibration dump: ONE cheap eval job, one array task per density
# (0.1% / 1% / 10% per observed channel), all 50 canonical snapshots, K=8,
# NFE=4, canonical fingerprint enforced inside dump_calib_points.py.
# Origin-HPC only (needs Save_TrainedModel). After it lands, run
# conformal_recalib.py on each dump dir via a CPU srun (login-node rule),
# and recalibrate_spread.py on the existing calib_sweep JSONs (login OK).
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RD=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
NOBS_LIST=(1953 19531 195312)
NOBS=${NOBS_LIST[$SLURM_ARRAY_TASK_ID]}
OUT="$RD/Evaluation/calib_points_n${NOBS}_K8_nfe4"
L=calib_dump_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
echo "task=$SLURM_ARRAY_TASK_ID n_obs=$NOBS host=$(hostname) start=$(date)" > $L
python dump_calib_points.py --run-dir "$RD" --ckpt best.pt \
    --K 8 --n-steps 4 --n-snapshots 50 \
    --cond-fields 0 2 --n-obs $NOBS $NOBS \
    --query-subset 200000 --chunk 262144 \
    --out-dir "$OUT" >> $L 2>&1
RC=$?
echo "end=$(date) rc=$RC" >> $L
exit $RC
