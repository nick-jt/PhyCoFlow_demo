#!/bin/bash
#SBATCH --job-name=eval_n34n35
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=eval_n34n35_${SLURM_JOB_ID}.log
# N34 = k^-5/3 GP source prior, N35 = windowed spectral objective. Both are
# aimed at the unconstrained-channel spectral deficit, so the decisive number
# is the band ratio, not the flow-matching loss (whose spectral term differs
# between arms and is therefore not comparable across them).
for TAG in DemoN34 DemoN35; do
  RD=$(ls -d ../Save_TrainedModel/JHU/pointcloud_ffm/*${TAG}_* 2>/dev/null | tail -1)
  if [ -z "$RD" ] || [ ! -f "$RD/best.pt" ]; then
    echo "SKIP $TAG: no run dir or best.pt" >> $L; continue
  fi
  echo "##### $TAG  $RD" >> $L
  mkdir -p "$RD/Evaluation"
  for NS in 2 4; do
    echo "### $TAG n_steps=$NS (50 snapshots, matched protocol)" >> $L
    python ensemble_eval.py --run-dir "$RD" --ckpt best.pt \
        --K 8 --n-steps $NS --n-snapshots 50 \
        --cond-fields 0 2 --n-obs 19531 19531 \
        --out "$RD/Evaluation/matched_nfe${NS}_K8.json" >> $L 2>&1
  done
done
echo "ALL DONE" >> $L
