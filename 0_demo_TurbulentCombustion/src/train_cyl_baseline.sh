#!/bin/bash
#SBATCH --job-name=cyl2d_baseline
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G

# Generic 2D-cylinder baseline launcher.
# env: TRAINER (train_Det_Baseline.py | train_Gen_Baseline.py),
#      CONFIG (path relative to repo root), LFM_STAGE (Gen only, default 1)
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

TRAINER=${TRAINER:-train_Det_Baseline.py}
CONFIG=${CONFIG:?set CONFIG=Save_config/cylinder2d/...yaml}
STAGE=${LFM_STAGE:-1}
TAG=$(basename "$CONFIG" .yaml)
LOG_FILE="train_cyl_${TAG}_stage${STAGE}_${SLURM_JOB_ID}.log"
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST trainer=$TRAINER config=$CONFIG stage=$STAGE"

if [[ "$TRAINER" == "train_Gen_Baseline.py" ]]; then
    CUDA_VISIBLE_DEVICES=0 python "$TRAINER" --config "$CONFIG" \
        --training-stage "$STAGE" >> "$LOG_FILE" 2>&1
else
    CUDA_VISIBLE_DEVICES=0 python "$TRAINER" --config "$CONFIG" >> "$LOG_FILE" 2>&1
fi
echo "exit status: $?"
