#!/bin/bash
#SBATCH --job-name=sen_seedck
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=64G
set -u
set -o pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"
L=verify_seedcheck_${SLURM_JOB_ID}.log
python -u verify_seedcheck.py \
  --config /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config/config_baseline_Senseiver_iclr.yaml \
  --seed 0 --n-obs 19531 --snaps 29 >> "$L" 2>&1
status=$?
echo "python exit status: $status" >> "$L"
exit $status
