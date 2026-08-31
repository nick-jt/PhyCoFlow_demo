#!/bin/bash
#SBATCH --job-name=s3gm_esmoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
source ~/envs/jhtdb
cd /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src
RUN=$(ls -dt ../Save_TrainedModel/JHU/baseline_s3gm/smoke/Baseline_s3gm_Stage1_DemoN92_* | head -1)
L="s3gm_evalsmoke_${SLURM_JOB_ID}.log"
echo "RUN=$RUN" >> "$L"
for N in 25 100 200; do
  echo "===== n_steps=$N =====" >> "$L"
  python -u eval_s3gm3d.py --run-dir "$RUN" --ckpt last --n-snapshots 2 --K 2 \
      --n-steps $N --out "$RUN/evalsmoke_N${N}.json" >> "$L" 2>&1
  echo "rc=$?" >> "$L"
done
exit 0
