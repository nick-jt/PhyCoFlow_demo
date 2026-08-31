#!/bin/bash
#SBATCH --job-name=cache_upgrade
#SBATCH --time=1:00:00
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
L=cache_upgrade_${SLURM_JOB_ID}.log
python test_ode_cache.py >> $L 2>&1
N29=../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100
python benchmark_cost.py --mode scale --run-dir $N29 --resolutions 125 --n-steps 4 \
    --out /tmp/scale_check_$SLURM_JOB_ID.json >> $L 2>&1
echo "ALL DONE" >> $L
