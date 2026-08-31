#!/bin/bash
#SBATCH --job-name=jhu_tmp_eng
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --mem=180G
# Temporally-blocked SAME-REGION companion training (paper pending item vi):
# N29 architecture config (spec02) retrained on the single-cutout 617-frame
# consecutive JHU dataset with a time-forward block split (last 25% = val,
# 100-frame guard gap; frame correlation ~0.67 at separation 100, so the
# residual correlation must be disclosed with the result). Budget-matched to
# the canonical 48k steps: 363 train frames / batch 20 -> ~19 steps/epoch,
# epochs 2500. Labelled DemoN33 so it can never be confused with the
# canonical cross-cube N29. Submit via submit_chain.sh (5 segments).
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=100 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
module load cuda/12.4.0
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
DATA=${JHU_DATA:-$DEMO/Dataset/JHU_TurbulenceDataset.h5}
L=$DEMO/src/train_jhu_temporal_eng_${SLURM_JOB_ID}.log
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date)" > $L

# ---- stage the H5 to node-local /tmp, size-checked, with fallback ----------
STAGE=/tmp/$USER/jhu_${SLURM_JOB_ID}
mkdir -p $STAGE
SZ=$(stat -c %s $DATA)
AVAIL=$(df -B1 --output=avail /tmp | tail -1)
RUNDATA=$DATA
if [ "$AVAIL" -gt $((SZ + 20000000000)) ]; then
  cp $DATA $STAGE/ && DSZ=$(stat -c %s $STAGE/$(basename $DATA)) || DSZ=0
  if [ "$DSZ" = "$SZ" ]; then RUNDATA=$STAGE/$(basename $DATA);
  else rm -f $STAGE/$(basename $DATA); fi
fi
[ -f "${DATA%.h5}.stats.pt" ] && cp "${DATA%.h5}.stats.pt" "${RUNDATA%.h5}.stats.pt" 2>/dev/null
echo "data=$RUNDATA" >> $L

# ---- job config: spec02 verbatim except data/save_dir/epochs ---------------
CFG=Save_config/jhu_temporal_eng.yaml
sed -e "s|^data:.*|data: \"$RUNDATA\"|" \
    -e "s|^save_dir:.*|save_dir: \"Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_temporal_spec02\"|" \
    -e "s|^epochs.*|epochs : 2500|" \
    -e "s|^Demo_Num:.*|Demo_Num: 33|" \
    $DEMO/Save_config/config_iclr_jhu_xcube_spec02.yaml > $DEMO/$CFG

CUDA_VISIBLE_DEVICES=0 python -u train_pointcloud_ffm.py \
    --RELOAD --config $DEMO/$CFG --Demo-Num 33 >> $L 2>&1
RC=$?
[ -f "${RUNDATA%.h5}.stats.pt" ] && cp "${RUNDATA%.h5}.stats.pt" "${DATA%.h5}.stats.pt" 2>/dev/null
rm -rf $STAGE
echo "end=$(date) rc=$RC" >> $L
exit $RC
