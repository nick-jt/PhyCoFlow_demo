#!/bin/bash
#SBATCH --job-name=sen_gnorm
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
set -u
set -o pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"
L=diag_senseiver_gradnorm_${SLURM_JOB_ID}.log
python -u diag_senseiver_gradnorm.py \
  --config /work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_config/config_baseline_Senseiver_iclr.yaml \
  --steps 80 \
  --out /home/ntricard/.claude/jobs/3ac3fd02/tmp/gradnorm/gradnorm.json >> "$L" 2>&1
status=$?
echo "python exit status: $status" >> "$L"
exit $status
