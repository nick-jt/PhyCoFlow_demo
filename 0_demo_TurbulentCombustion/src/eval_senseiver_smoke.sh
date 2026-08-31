#!/bin/bash
#SBATCH --job-name=sen_evsmk
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
set -o pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
source ~/envs/jhtdb
cd "$SLURM_SUBMIT_DIR"
D=../Save_TrainedModel/JHU/baseline_senseiver/Baseline_senseiver_Stage1_DemoN41_20260828_152643
python -u eval_senseiver_iclr.py --run-dir "$D" --ckpt budget.pt --split val \
  --seed 0 --n-snapshots 3 --n-obs 19531 \
  --sweep 1953 19531 --sweep-snapshots 2 \
  --out /home/ntricard/.claude/jobs/3ac3fd02/tmp/eval_smoke.json >> eval_senseiver_smoke_${SLURM_JOB_ID}.log 2>&1
status=$?
echo "python exit status: $status" >> eval_senseiver_smoke_${SLURM_JOB_ID}.log
exit $status
