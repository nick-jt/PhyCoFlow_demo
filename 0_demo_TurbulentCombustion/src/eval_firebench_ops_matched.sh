#!/bin/bash
#SBATCH --job-name=fb_ops_matched
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G

# ==========================================================================
# FRAME-MATCHED FireBench operator-matrix re-eval (audit 2026-09-05).
# See FIREBENCH_FRAME_AUDIT_2026-09-05.md.
#
# Problem being fixed: jobs 16726561/16764760 passed the SAME --snapshot-index
# values (3 9 10 5 2 1 0 4) to ours (train_ratio 0.9, val = abs frames 109-119)
# and to the baselines (train_ratio 0.75, val = abs frames 90-119). Under the
# baselines' split those indices map to abs frames 90-95,99,100 (u12 t=90-110s)
# while ours was scored on abs frames 109-114,118,119 (u12 t=128-148s):
# zero frame overlap, materially different fire age.
#
# This job rescores ONLY the 5-op x 2-baseline (LFM, Senseiver) cells on the
# SAME ABSOLUTE FRAMES ours was scored on. Ours' cells are untouched.
#
# Remap table (baseline val index = absolute frame - 90; verified against
# src/helpers_baseline.py block split with train_ratio 0.75, gap 10, N=120;
# int() and round() boundary semantics give the identical baseline split, so
# the remap is valid under both):
#   ours val idx   abs frame   u12 t[s]   baseline --snapshot-index
#        0            109        128              19
#        1            110        130              20
#        2            111        132              21
#        3            112        134              22
#        4            113        136              23
#        5            114        138              24
#        9            118        146              28
#       10            119        148              29
#
# Operator-seed matching: ours used per-snapshot op seed = 1000 + (ours val
# idx). We keep ENSEMBLE_OP_SEED = 1000 + OURS_IDX (not 1000 + baseline idx)
# so each baseline sees the same operator realization ours saw on that frame,
# exactly as the original job matched seeds by shared index.
#
# Outputs: ops_<op>_matched_snap<OURS_IDX>.json in each baseline run dir's
# Evaluation/, keyed by OURS' val index for cell-by-cell comparability with
# ours' ops_<op>_K8.json snapshot entries. Legacy ops_<op>_snap<S>.json files
# are left in place, unmodified.
#
# Protocol caveat (deliberate): this reuses evaluate_Gen/Det_Baseline.py with
# the same env-var conditioning pipeline as the original jobs, i.e. the
# LEGACY sensor-draw protocol that the 2026-08-29 audit quarantined. That is
# intentional: ours' Table-2 cells are NOT rerun here, so only the frame
# identity may change relative to jobs 16726561/16764760. A migration of the
# whole matrix to the canonical sensor draw is a separate job.
#
# Estimated runtime: the original job ran all 80 baseline evals in ~20 min on
# one H100; 3:30 wall is a large margin.
# ==========================================================================
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10
source ~/envs/jhtdb

WT=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
MAIN=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
STM=$MAIN/Save_TrainedModel
CFG=$MAIN/Save_config
cd $WT/src

LFM=$(ls -d $STM/firebench/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN35_* | tail -1)
SEN=$(ls -d $STM/firebench/baseline_senseiver/Baseline_senseiver_Stage1_DemoN36_* | tail -1)
L=$WT/src/eval_firebench_ops_matched_${SLURM_JOB_ID}.log
echo "LFM=$LFM  SEN=$SEN" >> $L

# ours-val-idx : baseline-val-idx pairs (see remap table above)
PAIRS="0:19 1:20 2:21 3:22 4:23 5:24 9:28 10:29"

export ENSEMBLE_K=8
run_base () {  # $1 model  $2 op name  $3 noise  $4 occl  $5 drop
  MODEL=$1; OP=$2
  for P in $PAIRS; do
    OURS_IDX=${P%%:*}; BASE_IDX=${P##*:}
    export ENSEMBLE_NOISE=$3 ENSEMBLE_OCCLUDE=$4 ENSEMBLE_DROPFIELD=$5 \
           ENSEMBLE_OP_SEED=$((1000+OURS_IDX))
    echo "##### $MODEL $OP ours_idx=$OURS_IDX -> base_idx=$BASE_IDX (abs frame $((90+BASE_IDX)))" >> $L
    if [ "$MODEL" = lfm ]; then
      ENSEMBLE_OUT=$LFM/Evaluation/ops_${OP}_matched_snap${OURS_IDX}.json \
      python evaluate_Gen_Baseline.py --config $CFG/config_baseline_Gen_firebench.yaml \
        --run-dir $LFM --training-stage 2 --split val --snapshot-index $BASE_IDX >> $L 2>&1
    else
      ENSEMBLE_OUT=$SEN/Evaluation/ops_${OP}_matched_snap${OURS_IDX}.json \
      python evaluate_Det_Baseline.py --config $CFG/config_baseline_Det_firebench.yaml \
        --run-dir $SEN --split val --snapshot-index $BASE_IDX >> $L 2>&1
    fi
  done
  unset ENSEMBLE_NOISE ENSEMBLE_OCCLUDE ENSEMBLE_DROPFIELD
}
for MODEL in lfm sen; do
  run_base $MODEL clean   0 ""        ""
  run_base $MODEL noise01 0.1 ""      ""
  run_base $MODEL noise03 0.3 ""      ""
  run_base $MODEL slab25  0 "slab:0.25" ""
  run_base $MODEL dropvw  0 ""        "1,2"
done
echo "ALL DONE (matched frames)" >> $L
