#!/bin/bash
#SBATCH --job-name=fb_ops_matrix
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
STM=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel
CFG=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_config
N18=$(ls -d $STM/firebench/pointcloud_ffm/iclr_firebench_v4_DemoN18_* | tail -1)
N11=$(ls -d $STM/firebench/pointcloud_ffm/iclr_firebench_v3_DemoN11_* $STM/iclr_firebench_v3_* 2>/dev/null | tail -1)
LFM=$(ls -d $STM/firebench/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN35_* | tail -1)
SEN=$(ls -d $STM/firebench/baseline_senseiver/Baseline_senseiver_Stage1_DemoN36_* | tail -1)
L=eval_firebench_ops_${SLURM_JOB_ID}.log
echo "N18=$N18  N11=$N11  LFM=$LFM  SEN=$SEN" >> $L

# Operator matrix: name | ensemble_eval extra args | env for baselines
run_ours () {  # $1 run_dir  $2 tag  $3 op name  $4.. extra args
  RD=$1; TAG=$2; OP=$3; shift 3
  mkdir -p "$RD/Evaluation"
  echo "##### $TAG $OP" >> $L
  python ensemble_eval.py --run-dir "$RD" --ckpt best.pt --K 8 --n-steps 4 \
      --n-snapshots 8 --op-seed 1000 "$@" \
      --out "$RD/Evaluation/ops_${OP}_K8.json" >> $L 2>&1
}
for M in "$N18 N18" "$N11 N11"; do
  set -- $M
  run_ours $1 $2 clean
  run_ours $1 $2 noise01 --noise-sigma 0.1
  run_ours $1 $2 noise03 --noise-sigma 0.3
  run_ours $1 $2 slab25 --occlude slab:0.25
  run_ours $1 $2 dropvw --drop-fields 1 2
done

# Snapshot ids chosen by ensemble_eval (seed 0) -> reuse for baselines
SNAPS=$(python -c "
import json; d=json.load(open('$N18/Evaluation/ops_clean_K8.json'))
print(' '.join(str(c.get('snapshot',c.get('case'))) for c in (d.get('cases') or d.get('snapshots'))))")
echo "matched snapshots: $SNAPS" >> $L

export ENSEMBLE_K=8
run_base () {  # $1 model  $2 op name  $3 noise  $4 occl  $5 drop
  MODEL=$1; OP=$2
  for S in $SNAPS; do
    export ENSEMBLE_NOISE=$3 ENSEMBLE_OCCLUDE=$4 ENSEMBLE_DROPFIELD=$5 ENSEMBLE_OP_SEED=$((1000+S))
    if [ "$MODEL" = lfm ]; then
      ENSEMBLE_OUT=$LFM/Evaluation/ops_${OP}_snap${S}.json \
      python evaluate_Gen_Baseline.py --config $CFG/config_baseline_Gen_firebench.yaml \
        --run-dir $LFM --training-stage 2 --split val --snapshot-index $S >> $L 2>&1
    else
      ENSEMBLE_OUT=$SEN/Evaluation/ops_${OP}_snap${S}.json \
      python evaluate_Det_Baseline.py --config $CFG/config_baseline_Det_firebench.yaml \
        --run-dir $SEN --split val --snapshot-index $S >> $L 2>&1
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
echo "ALL DONE" >> $L
