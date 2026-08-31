#!/bin/bash
#SBATCH --job-name=ldw_ffm_s1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=200G
#SBATCH --array=0-2
# LDW-FFM Stage-1 blend test (PLAN_IMPROVE_2026-08-30 section 5).
# One single-GPU task per sensor density; canonical seed-0 protocol,
# N29 best.pt, K=8, NFE 4, hard clamp, all 50 cube-3 val snapshots.
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
JOBDIR=/home/ntricard/.claude/jobs/3ac3fd02/tmp/ldw_ffm
cd $JOBDIR
RD=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
NOBS_LIST=(1953 19531 195312)
NOBS=${NOBS_LIST[$SLURM_ARRAY_TASK_ID]}
L=$JOBDIR/ldw_ffm_s1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
echo "task=$SLURM_ARRAY_TASK_ID n_obs=$NOBS host=$(hostname) start=$(date)" > $L
mkdir -p "$RD/Evaluation"
python -u ldw_ffm_stage1.py --n-obs $NOBS --run-dir "$RD" --ckpt best.pt \
    --K 8 --n-steps 4 --n-snapshots 50 --chunk 262144 --seed 0 \
    --out "$RD/Evaluation/ldw_ffm_stage1_n${NOBS}.json" >> $L 2>&1
RC=$?
echo "end=$(date) rc=$RC" >> $L
exit $RC
