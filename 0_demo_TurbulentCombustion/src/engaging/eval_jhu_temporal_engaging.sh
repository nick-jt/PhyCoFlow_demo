#!/bin/bash
#SBATCH --job-name=jhu_tmp_ev
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=node2906
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --mem=96G
# Canonical n=50 eval of the DemoN33 temporally-blocked same-region companion
# (paper item at main.tex:142). Reports the numbers that accompany the
# disclosure of ~0.67 residual frame correlation at split gap 100.
#
# JHU_SPLIT_GAP MUST be 100, matching training (363 train / 154 val). The gap
# defines the val block, so evaluating at a different gap scores a different
# split than the model was trained against.
#
# H100 REQUIRED: this runs at the canonical operating point (seed 0,
# cond_fields [0,2], n_obs 19531+19531) where check_canonical_fingerprint
# bites, and torch.randperm on CUDA is not portable across GPU SKU. Engaging
# has exactly one h100 node, hence the nodelist pin.
#
# --data is passed explicitly: the Engaging launchers stage the H5 to
# node-local /tmp, so the run's args.json records a path that no longer exists.
set -u
export PYTHONUNBUFFERED=1
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=100
source ~/envs/phycoflow
DEMO=/home/ntricard/projects/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $DEMO/src
RD=$(ls -dt $DEMO/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_temporal_spec02_DemoN33_* 2>/dev/null | head -1)
L=$DEMO/src/eval_jhu_temporal_${SLURM_JOB_ID}.log
echo "host=$(hostname) run_dir=${RD:-NONE} start=$(date)" > $L
if [ -z "${RD:-}" ] || [ ! -f "$RD/best.pt" ]; then
  echo "GUARD: no finished DemoN33 run" >> $L; exit 3
fi
grep -aq "Training complete" $DEMO/src/train_jhu_temporal_eng_*.log 2>/dev/null \
  || echo "WARNING: no 'Training complete' marker found in any temporal log" >> $L
RC=0
for NS in 4 2; do
  echo "=== canonical n=50, NFE $NS ===" >> $L
  python -u ensemble_eval.py \
    --run-dir "$RD" --ckpt best.pt \
    --data "$DEMO/Dataset/JHU_TurbulenceDataset.h5" \
    --K 8 --n-steps $NS --n-snapshots 50 \
    --cond-fields 0 2 --n-obs 19531 19531 \
    --seed 0 --op-seed 1000 \
    --out "$RD/Evaluation/canonical_all50_nfe${NS}_K8.json" >> $L 2>&1 || RC=$?
done
echo "end=$(date) rc=$RC" >> $L
exit $RC
