#!/bin/bash
#SBATCH --job-name=cnf_figsmoke
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=128G
set -euo pipefail
REPO=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
source ~/envs/jhtdb
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral MPLCONFIGDIR=/tmp
export CONFILD_JOB_START=$(date +%s)
CFG=Save_config/config_baseline_CoNFiLD_figsmoke.yaml
run_stage () {
  local stage=$1
  local t0=$(date +%s)
  export CONFILD_SLURM_START=$t0
  python src/train_Gen_Baseline.py --config "$CFG" --training-stage "$stage" --device cuda:0
  echo "[confild:slurm] stage=${stage} slurm_stage_elapsed_s=$(( $(date +%s) - t0 ))"
}
run_stage 1
run_stage 2
echo "[confild:slurm] job_total_elapsed_s=$(( $(date +%s) - CONFILD_JOB_START ))"
