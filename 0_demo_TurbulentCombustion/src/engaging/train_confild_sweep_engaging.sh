#!/bin/bash
#SBATCH --job-name=confild_swp
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --mem=96G
# CoNFiLD stage-1 latent-dim sweep on MIT Engaging (2026-08-31). Select arm:
#   ARM=sweep1024|sweep2048|sweep4096|strict2048
# Each config carries wallclock_budget_s=16200 (4.5 h/segment, graceful exit
# inside the 6 h wall). Chain EXACTLY 3 segments per arm:
#   ARM=<arm> ./submit_chain.sh train_confild_sweep_engaging.sh 3 --export=ALL,ARM=<arm>
# 3 x 16200 s = 48600 s, budget-matched to the origin 13.5 h stage-1 protocol.
# More segments would keep training past the matched budget (epochs cap of
# 100000 is never reached) — do NOT over-provision this chain.
# Stage 2 is NOT trained here (gated on the oracle sweep outcome).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
module load cuda/12.4.0
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
ARM=${ARM:?set ARM=sweep1024|sweep2048|sweep4096|strict2048}
DATA=$DEMO/Dataset/JHU_4cubes_stride100.h5
L=$DEMO/src/train_confild_${ARM}_eng_${SLURM_JOB_ID}.log
echo "host=$(hostname) arm=$ARM start=$(date)" > $L

STAGE=/tmp/$USER/confild_${SLURM_JOB_ID}; mkdir -p $STAGE
SZ=$(stat -c %s $DATA); AVAIL=$(df -B1 --output=avail /tmp | tail -1); RUNDATA=$DATA
if [ "$AVAIL" -gt $((SZ + 20000000000)) ]; then
  cp $DATA $STAGE/ && DSZ=$(stat -c %s $STAGE/$(basename $DATA)) || DSZ=0
  if [ "$DSZ" = "$SZ" ]; then RUNDATA=$STAGE/$(basename $DATA); else rm -f $STAGE/$(basename $DATA); fi
fi
echo "data=$RUNDATA" >> $L

SRC_CFG=$DEMO/Save_config/config_baseline_CoNFiLD_xcube_${ARM}.yaml
CFG=$DEMO/Save_config/confild_${ARM}_eng.yaml
sed "s|$DEMO/Dataset/JHU_4cubes_stride100.h5|$RUNDATA|g" $SRC_CFG > $CFG
CUDA_VISIBLE_DEVICES=0 python -u train_Gen_Baseline.py --config $CFG \
    --training-stage 1 --reload >> $L 2>&1
RC=$?
rm -rf $STAGE
echo "end=$(date) rc=$RC" >> $L
exit $RC
