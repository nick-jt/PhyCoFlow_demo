#!/bin/bash
#SBATCH --job-name=sit_eval
#SBATCH --time=12:00:00
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
CFG=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config
SIT=$(readlink -f "$(ls -d ../Save_TrainedModel/JHU/baseline_sit/Baseline_sit_Stage1_DemoN41_* | tail -1)")
L=eval_sit_xcube_${SLURM_JOB_ID}.log
echo "SIT=$SIT" >> $L
mkdir -p "$SIT/Evaluation"
export ENSEMBLE_K=8
for S in $(seq 0 49); do
  echo "##### SIT snap $S" >> $L
  ENSEMBLE_OUT=$SIT/Evaluation/crps_snap${S}.json \
  python evaluate_Gen_Baseline.py --config $CFG/config_baseline_SiT_xcube.yaml \
    --run-dir $SIT --training-stage 1 --split val --snapshot-index $S 2>&1 \
    | grep -aE "\[ensemble\]|Traceback|Error|Exception|error:" >> $L
done
N=$(ls "$SIT/Evaluation"/crps_snap*.json 2>/dev/null | wc -l)
echo "ALL DONE: $N/50 snapshot metric files written" >> $L
[ "$N" -eq 50 ] || echo "WARNING: expected 50 crps_snap*.json, found $N" >> $L
