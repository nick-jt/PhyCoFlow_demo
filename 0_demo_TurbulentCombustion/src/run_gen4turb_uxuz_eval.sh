#!/bin/bash
#SBATCH --job-name=g4t_uxuz_eval
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
OUT=/projects/ammoniacomb/generative_reconstruction/baselines/Gen4Turbulence/3_flow_reconstruction/eval
L=gen4turb_uxuz_eval_${SLURM_JOB_ID}.log
for CK in models_uxuz/model_4930.pt models_uxuz/best_model.pt; do
  N=$(basename $CK .pt)
  python gen4turb_eval.py --ckpt $CK --K 8 --coverage 0.01 \
      --out $OUT/cube3_uxuz_${N}_K8.json >> $L 2>&1
done
echo "ALL DONE" >> $L
