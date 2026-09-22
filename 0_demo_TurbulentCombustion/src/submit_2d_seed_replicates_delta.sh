#!/bin/bash
# P1-6 training-seed replicates: the canonical 2D fleets re-trained with seeds
# 7 and 1337 (configs in Save_config/<ds>_seed<seed>/, identical otherwise).
# Run from src/:  bash submit_2d_seed_replicates_delta.sh [kolm|cyl|all] [7|1337|both]
# Evals: DATASET=kolmogorov2d_seed7 sbatch eval_kolm_fleet.sh  (prefix kolm_fleet_seed7)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
WHICH=${1:-all}; SEEDS=${2:-both}; [ "$SEEDS" = both ] && SEEDS="7 1337"
MAP=fleet_jobs_delta.txt
sub() { local label=$1; shift; local jid; jid=$(sbatch --parsable "$@"); echo "$(date +%F_%T) $label $jid" | tee -a "$MAP"; echo "$jid"; }
for SEED in $SEEDS; do
  if [[ $WHICH == kolm || $WHICH == all ]]; then
    C=Save_config/kolmogorov2d_seed$SEED; L=train_kolm_baseline.sh; T=$SEED
    sub kolm_s${T}_dmfgen    --export=ALL,CONFIG=$C/config_bench_kolm_ffm_s$SEED.yaml,DEMO_NUM=$((101 + (SEED==7 ? 100 : 200))) train_bench_kolm_ffm.sh
    sub kolm_s${T}_senseiver --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=$C/config_baseline_Det_kolm_s$SEED.yaml    $L
    sub kolm_s${T}_mlp_rbf   --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=$C/config_baseline_MLPRBF_kolm_s$SEED.yaml $L
    sub kolm_s${T}_geofno    --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=$C/config_baseline_GeoFNO_kolm_s$SEED.yaml $L
    sub kolm_s${T}_sit       --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_SiT_kolm_s$SEED.yaml    $L
    sub kolm_s${T}_s3gm      --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_S3GM_kolm_s$SEED.yaml $L
    S1=$(sub kolm_s${T}_lfm_s1 --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_Gen_kolm_s$SEED.yaml,LFM_STAGE=1 $L | tail -1)
    sub kolm_s${T}_lfm_s2    --dependency=afterok:$S1 --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_Gen_kolm_s$SEED.yaml,LFM_STAGE=2 $L
  fi
  if [[ $WHICH == cyl || $WHICH == all ]]; then
    C=Save_config/cylinder2d_seed$SEED; L=train_cyl_baseline.sh; T=$SEED
    sub cyl_s${T}_dmfgen     --export=ALL,CONFIG=$C/config_bench_cyl_ffm_s$SEED.yaml,DEMO_NUM=$((102 + (SEED==7 ? 100 : 200))) train_bench_cyl_ffm.sh
    sub cyl_s${T}_senseiver  --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=$C/config_baseline_Det_cyl_s$SEED.yaml    $L
    sub cyl_s${T}_mlp_rbf    --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=$C/config_baseline_MLPRBF_cyl_s$SEED.yaml $L
    sub cyl_s${T}_geofno     --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=$C/config_baseline_GeoFNO_cyl_s$SEED.yaml $L
    sub cyl_s${T}_sit        --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_SiT_cyl_s$SEED.yaml    $L
    sub cyl_s${T}_s3gm       --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_S3GM_cyl_s$SEED.yaml $L
    S1=$(sub cyl_s${T}_lfm_s1 --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_Gen_cyl_s$SEED.yaml,LFM_STAGE=1 $L | tail -1)
    sub cyl_s${T}_lfm_s2     --dependency=afterok:$S1 --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_Gen_cyl_s$SEED.yaml,LFM_STAGE=2 $L
  fi
done
