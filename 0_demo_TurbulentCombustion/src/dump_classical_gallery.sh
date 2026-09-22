#!/bin/bash
#SBATCH --job-name=cls_gallery
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
#SBATCH --output=cls_gallery_%j.log
# Classical gallery dumps (GPU node: exact CUDA sensor draw = fleet draw).
set -uo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RC=0
for DS in kolmogorov2d cylinder2d cylinder2d_surface; do
  echo "=== $DS ==="; python -u dump_classical_gallery.py --dataset $DS || RC=1
done
exit $RC
