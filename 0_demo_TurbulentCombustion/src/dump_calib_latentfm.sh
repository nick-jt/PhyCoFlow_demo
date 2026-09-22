#!/bin/bash
#SBATCH --job-name=lfm_calib
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=200G
#SBATCH --array=0-2
# Latent-FM side of the Route-2 conformal comparison (paper TODO: "same
# treatment for the latent-FM dumps, so the calibration comparison is two-sided
# as the scalar one already is").
#
# One array task per density, matching dump_calib_points.sh's DMF-Gen densities
# exactly: 0.1% / 1% / 10% of the 125^3 grid per observed channel. The eval and
# the dump come from ONE sampling pass -- eval_latentfm_ensemble.py writes both
# -- so this costs no more than the eval alone.
#
# Output dirs are density-qualified. That is not cosmetic: writing a
# density-varying result to a fixed name is exactly how the canonical
# Kolmogorov S3GM row got overwritten (see DELTA_STATUS sec.20).
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RD=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN24_20260828_164541
NOBS_LIST=(1953 19531 195312)
NOBS=${NOBS_LIST[$SLURM_ARRAY_TASK_ID]}
OUT="$RD/Evaluation/calib_points_n${NOBS}_K8_nfe4"
L=lfm_calib_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
echo "task=$SLURM_ARRAY_TASK_ID n_obs=$NOBS host=$(hostname) start=$(date)" > $L
python eval_latentfm_ensemble.py --run-dir "$RD" --ckpt best \
    --K 8 --nfe 4 --n-snapshots 50 \
    --cond-fields 0 2 --n-obs $NOBS $NOBS \
    --query-subset 200000 --no-figs \
    --out "$RD/Evaluation/lfm_calib_n${NOBS}_K8_nfe4.json" \
    --dump-calib-dir "$OUT" >> $L 2>&1
RC=$?
echo "end=$(date) rc=$RC" >> $L
exit $RC
