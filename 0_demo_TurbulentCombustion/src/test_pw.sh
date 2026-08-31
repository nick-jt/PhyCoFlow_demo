#!/bin/bash
#SBATCH --job-name=sit_pw
#SBATCH --time=00:25:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH --partition=gpu-h100 --account=f2pde --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
python test_persistent_workers.py /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config/config_baseline_SiT_xcube_matched.yaml > test_pw_${SLURM_JOB_ID}.log 2>&1
echo "rc=$?" >> test_pw_${SLURM_JOB_ID}.log
