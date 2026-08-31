#!/bin/bash
#SBATCH --job-name=fb_bl_eng
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --mem=96G
# FireBench baselines on MIT Engaging. Select with env vars:
#   BL=lfm  LFM_STAGE=1|2   -> train_Gen_Baseline.py (latent FM; stage 1 = AE)
#   BL=det                  -> train_Det_Baseline.py (Senseiver)
# Both trainers resume via --reload; submit via submit_chain.sh. Chain
# stage-2 LFM only after stage 1 finishes (separate chains, afterok).
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10 JHU_AUGMENT=reflect_y AUG_GRID_SHAPE=152,126,192
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
module load cuda/12.4.0
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
BL=${BL:?set BL=lfm or BL=det}
DATA=${FB_DATA:-/home/ntricard/orcd/scratch/firebench3d/FireBench_u10u12_merged.h5}
L=$DEMO/src/train_fb_${BL}_eng_${SLURM_JOB_ID}.log
echo "host=$(hostname) bl=$BL start=$(date)" > $L

STAGE=/tmp/$USER/fbbl_${SLURM_JOB_ID}; mkdir -p $STAGE
SZ=$(stat -c %s $DATA); AVAIL=$(df -B1 --output=avail /tmp | tail -1); RUNDATA=$DATA
if [ "$AVAIL" -gt $((SZ + 20000000000)) ]; then
  cp $DATA $STAGE/ && DSZ=$(stat -c %s $STAGE/$(basename $DATA)) || DSZ=0
  if [ "$DSZ" = "$SZ" ]; then RUNDATA=$STAGE/$(basename $DATA); else rm -f $STAGE/$(basename $DATA); fi
fi
echo "data=$RUNDATA" >> $L

if [ "$BL" = lfm ]; then
  SRC_CFG=$DEMO/Save_config/config_baseline_Gen_firebench.yaml
  CFG=$DEMO/Save_config/fb_lfm_eng.yaml
  sed "s|/projects/ammoniacomb/generative_reconstruction/firebench3d/FireBench_u10u12_merged.h5|$RUNDATA|g" $SRC_CFG > $CFG
  CUDA_VISIBLE_DEVICES=0 python -u train_Gen_Baseline.py --config $CFG \
      --training-stage ${LFM_STAGE:-1} --reload >> $L 2>&1
else
  SRC_CFG=$DEMO/Save_config/config_baseline_Det_firebench.yaml
  CFG=$DEMO/Save_config/fb_det_eng.yaml
  sed "s|/projects/ammoniacomb/generative_reconstruction/firebench3d/FireBench_u10u12_merged.h5|$RUNDATA|g" $SRC_CFG > $CFG
  CUDA_VISIBLE_DEVICES=0 python -u train_Det_Baseline.py --config $CFG \
      --reload >> $L 2>&1
fi
RC=$?
rm -rf $STAGE
echo "end=$(date) rc=$RC" >> $L
exit $RC
