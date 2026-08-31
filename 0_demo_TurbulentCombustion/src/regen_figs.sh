#!/bin/bash
#SBATCH --job-name=s3gm_figs
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
source ~/envs/jhtdb
D=/home/ntricard/.claude/jobs/3ac3fd02/tmp/s3gm_improve
python -u "$D/regen_figs.py"
echo "regen_figs rc=$?"
