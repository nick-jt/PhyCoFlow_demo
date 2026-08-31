#!/bin/bash
#SBATCH --job-name=iclr_jhu_k64
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mail-user=ntricard@mit.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=96G

set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR

DEMO_NUM=38
LOG_FILE="train_iclr_jhu_k64_${SLURM_JOB_ID}_DemoN${DEMO_NUM}.log"
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, python: $(which python)"
SD=$(grep -oP '^save_dir\s*:\s*"\K[^"]+' ../Save_config/config_iclr_jhu_k64.yaml)
if ls -d ../${SD}_DemoN${DEMO_NUM}_* >/dev/null 2>&1; then
  echo "ABORT: ${SD}_DemoN${DEMO_NUM}_* already exists" >&2; exit 1
fi
CUDA_VISIBLE_DEVICES=0 python train_pointcloud_ffm.py --config Save_config/config_iclr_jhu_k64.yaml         --Demo-Num $DEMO_NUM >> "$LOG_FILE" 2>&1
echo "exit status: $?"
