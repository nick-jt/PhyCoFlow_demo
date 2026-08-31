#!/bin/bash
#SBATCH --job-name=ckptvar_smoke
#SBATCH --time=0:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export CKPTVAR_RUN=/home/ntricard/.claude/jobs/3ac3fd02/tmp/smoke_run
export CKPTVAR_MIN=2 CKPTVAR_NSNAP=2 CKPTVAR_QSUB=50000
exec bash calib_ckptvar_eval.sh
