#!/bin/bash
#SBATCH --job-name=rng_probe
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=48G
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
python rng_parity_probe.py > rng_probe_${SLURM_JOB_ID}.log 2>&1
echo "rc=$?" >> rng_probe_${SLURM_JOB_ID}.log
