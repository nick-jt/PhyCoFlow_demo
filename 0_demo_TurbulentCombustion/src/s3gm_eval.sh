#!/bin/bash
#SBATCH --job-name=s3gm_eval
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
#SBATCH --array=0-1
set -u
# Canonical cross-cube evaluation. MUST run on a compute node: torch.randperm
# on CUDA is not portable between H100 PCIe (login) and H100 SXM (compute).
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
source ~/envs/jhtdb
cd /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src
CKPTS=(best last)
CK=${CKPTS[$SLURM_ARRAY_TASK_ID]}
RUN=${S3GM_RUN:?set S3GM_RUN to the run directory}
LOG="s3gm_eval_${CK}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
python eval_s3gm3d.py --run-dir "$RUN" --ckpt "$CK" \
    --seed 0 --op-seed 1000 --n-snapshots 50 --K 8 \
    --cond-fields 0 2 --n-obs 19531 19531 >> "$LOG" 2>&1
RC=$?
echo "eval rc=$RC" >> "$LOG"
exit $RC
