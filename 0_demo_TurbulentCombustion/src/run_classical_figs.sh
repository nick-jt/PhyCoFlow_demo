#!/bin/bash
#SBATCH --job-name=classical_figs
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export MPLCONFIGDIR=/tmp
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=classical_figs_${SLURM_JOB_ID}.log
{
  nvidia-smi -L
  python -u baseline_classical_figs.py --snapshots 11 33 --n-obs 19531 \
      --with-model --model-K 4 --model-nfe 4 \
      --out-dir ../Save_TrainedModel/JHU/baseline_classical/Figures
  echo "FIGS DONE"
} >> $L 2>&1
