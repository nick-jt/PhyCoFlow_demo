#!/bin/bash
#SBATCH --job-name=calib_probe
#SBATCH --time=6:00:00
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
L=calib_probe_${SLURM_JOB_ID}.log
echo "host=$(hostname) start=$(date)" > $L
python calib_probe.py >> $L 2>&1
RC=$?
echo "end=$(date) rc=$RC" >> $L
exit $RC
