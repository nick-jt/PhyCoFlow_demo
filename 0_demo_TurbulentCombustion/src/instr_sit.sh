#!/bin/bash
#SBATCH --job-name=sit_instr
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RUN=${SIT_RUN:?set SIT_RUN to the run dir}
L=instr_sit_${SLURM_JOB_ID}.log
echo "RUN=$RUN" >> $L
for CK in best last; do
  echo "##### checkpoint=$CK" >> $L
  python instrument_sit.py --run-dir "$RUN" --checkpoint $CK \
    --out "$RUN/instrumentation_${CK}.json" >> $L 2>&1
  RC=$?
  echo "rc=$RC for $CK" >> $L
done
