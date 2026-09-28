#!/bin/bash
#SBATCH --job-name=sit_xcube
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
LOG_FILE="train_sit_xcube_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 python train_Gen_Baseline.py \
        --config /work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_config/config_baseline_SiT_xcube.yaml \
        --training-stage 1 >> "$LOG_FILE" 2>&1
RC=$?
echo "train rc=$RC" >> "$LOG_FILE"
exit $RC
