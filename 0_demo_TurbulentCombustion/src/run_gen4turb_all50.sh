#!/bin/bash
#SBATCH --job-name=g4t_all50
#SBATCH --time=6:00:00
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
export G4T_ALL50=1
L=gen4turb_all50_${SLURM_JOB_ID}.log
python gen4turb_eval.py --ckpt models_uxuz/model_4930.pt --K 8 \
  --out /projects/ammoniacomb/generative_reconstruction/baselines/Gen4Turbulence/3_flow_reconstruction/eval/cube3_uxuz_model_4930_K8_all50.json >> $L 2>&1
echo "exit status: $?" >> $L
