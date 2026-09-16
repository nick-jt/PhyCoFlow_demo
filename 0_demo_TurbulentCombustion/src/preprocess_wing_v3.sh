#!/bin/bash
#SBATCH --job-name=wing_preproc
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=128G

set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
python dataset_shiftwing.py \
    --out-root /work/hdd/bilr/ntricard/datasets/shift_wing_processed_v3 \
    --n-workers 32 --train-cases 700 > preprocess_wing_v3_${SLURM_JOB_ID}.log 2>&1
echo "exit status: $?"
