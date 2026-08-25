#!/bin/bash
#SBATCH --job-name=bench_bl
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 BENCH_SAMPLE=1
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
CFG=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config
STM=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel
L=bench_baseline_${SLURM_JOB_ID}.log
echo "### LATENT-FM full sampler" >> $L
python evaluate_Gen_Baseline.py --config $CFG/config_baseline_Gen_xcube_aug.yaml \
  --run-dir $STM/JHU/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN23_20260818_153527 \
  --training-stage 2 --split val --snapshot-index 0 2>&1 | grep -E "\[bench\]|field_0" >> $L
echo "### SENSEIVER" >> $L
python evaluate_Det_Baseline.py --config $CFG/config_baseline_Det_xcube_aug.yaml \
  --run-dir $STM/JHU/baseline_senseiver/Baseline_senseiver_Stage1_DemoN31_20260818_083446 \
  --split val --snapshot-index 0 2>&1 | grep -E "\[bench\]|field_0" >> $L
echo "ALL DONE" >> $L
