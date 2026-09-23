#!/bin/bash
#SBATCH --job-name=sit_wing
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=96G
# NOT YET LAUNCHED. Wing baselines run only after the JHU comparison is
# settled and our model is final; see notes in dataset_wing_baseline.py about
# conditioning from the surface pool rather than random volume points.
set -u
export WING_AUGMENT=reflect_y
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
LOG=train_sit_wing_${SLURM_JOB_ID}.log
python train_Gen_Baseline.py \
    --config /work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_config/config_baseline_SiT_wing.yaml \
    --training-stage 1 >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
