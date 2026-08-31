#!/bin/bash
#SBATCH --job-name=fno3d_esmoke
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=128G
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RD="../Save_TrainedModel/JHU/baseline_fno/fno3d_smoke_DemoN91_20260828_171251"
L="eval_fno3d_smoke_${SLURM_JOB_ID}.log"
python -u ensemble_eval.py --run-dir "$RD" --ckpt best.pt --K 2 --n-steps 4 \
  --n-snapshots 2 --n-obs 19531 19531 --cond-fields 0 2 --seed 0 --op-seed 1000 \
  --chunk 1953125 --out "$RD/smoke_eval.json" > "$L" 2>&1
python -u bench_fno3d.py --run-dir "$RD" --ckpt best.pt --nfe 4 --batch-size 8 \
  --out "$RD/cost_instrumentation.json" >> "$L" 2>&1
