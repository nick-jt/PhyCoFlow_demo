#!/bin/bash
#SBATCH --job-name=eval_k32
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=eval_k32_${SLURM_JOB_ID}.log
RD=$(ls -d ../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_k32_DemoN37_2* | tail -1)
echo "RD=$RD" >> $L
mkdir -p "$RD/Evaluation"
# Matched to N29's protocol, unclamped. Validation loss says K=32 is worse;
# this checks whether that holds for reconstruction accuracy AND for the
# uncertainty metrics, which the flow-matching loss cannot speak to.
for NS in 2 4; do
  echo "##### K=32 n_steps=$NS" >> $L
  python ensemble_eval.py --run-dir "$RD" --ckpt best.pt \
      --K 8 --n-steps $NS --n-snapshots 50 --no-clamp \
      --cond-fields 0 2 --n-obs 19531 19531 --fig-every 25 \
      --out "$RD/Evaluation/matched_nfe${NS}_K8_all50_NOCLAMP.json" >> $L 2>&1
done
echo "ALL DONE" >> $L
