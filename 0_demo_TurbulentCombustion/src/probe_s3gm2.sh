#!/bin/bash
#SBATCH --job-name=s3gm_probe
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
export SRC=$SLURM_SUBMIT_DIR
python /home/ntricard/.claude/jobs/3ac3fd02/tmp/probe_s3gm2.py > "probe2_s3gm_${SLURM_JOB_ID}.log" 2>&1
RC=$?
echo "probe rc=$RC" >> "probe2_s3gm_${SLURM_JOB_ID}.log"
exit $RC
