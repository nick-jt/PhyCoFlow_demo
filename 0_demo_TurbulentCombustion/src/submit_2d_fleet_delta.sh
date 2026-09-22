#!/bin/bash
# Submit the full 2D learned fleet (Kolmogorov + cylinder) on DeltaAI, gated
# on the smoke jobs (--dependency=afterok). Run from src/:
#     bash submit_2d_fleet_delta.sh [kolm|cyl|all]   (default all)
# Env: SMOKE=0 skips the smoke gate (fleet submitted ungated).
# Every job prints its id; the map is appended to fleet_jobs_delta.txt so the
# eval launchers / handoff can trace run dirs to jobs.
#
# Why everything is retrained here: no checkpoints were transferred from
# Kestrel, and the GPU SKU changed (H100 -> GH200), so canonical evals must be
# re-run on ONE SKU per dataset anyway (handoff sec.3.4 item 6). Classical
# anchors are re-run too (same seeded draws, same SKU as the learned rows).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
WHICH=${1:-all}
SMOKE=${SMOKE:-1}
MAP=fleet_jobs_delta.txt
sub() {  # sub <label> <sbatch args...>
  local label=$1; shift
  local jid
  jid=$(sbatch --parsable "$@")
  echo "$(date +%F_%T) $label $jid" | tee -a "$MAP"
  echo "$jid"
}
dep() { [ -n "${1:-}" ] && echo "--dependency=afterok:$1" || true; }

if [[ $WHICH == kolm || $WHICH == all ]]; then
  SK=""; [ "$SMOKE" = 1 ] && SK=$(sub smoke_kolm2d smoke_kolm2d.sh | tail -1)
  D=$(dep "$SK")
  sub kolm_dmfgen      $D train_bench_kolm_ffm.sh
  sub kolm_senseiver   $D --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=Save_config/kolmogorov2d/config_baseline_Det_kolm.yaml     train_kolm_baseline.sh
  sub kolm_mlp_rbf     $D --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=Save_config/kolmogorov2d/config_baseline_MLPRBF_kolm.yaml  train_kolm_baseline.sh
  sub kolm_geofno      $D --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=Save_config/kolmogorov2d/config_baseline_GeoFNO_kolm.yaml  train_kolm_baseline.sh
  sub kolm_sit         $D --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=Save_config/kolmogorov2d/config_baseline_SiT_kolm.yaml     train_kolm_baseline.sh
  sub kolm_s3gm        $D --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=Save_config/kolmogorov2d/config_baseline_S3GM_kolm.yaml    train_kolm_baseline.sh
  S1=$(sub kolm_lfm_s1 $D --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=Save_config/kolmogorov2d/config_baseline_Gen_kolm.yaml,LFM_STAGE=1 train_kolm_baseline.sh | tail -1)
  sub kolm_lfm_s2      --dependency=afterok:$S1 --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=Save_config/kolmogorov2d/config_baseline_Gen_kolm.yaml,LFM_STAGE=2 train_kolm_baseline.sh
  sub kolm_classical   $D run_classical_kolm.sh
fi

if [[ $WHICH == cyl || $WHICH == all ]]; then
  SC=""; [ "$SMOKE" = 1 ] && SC=$(sub smoke_cyl2d smoke_cyl2d.sh | tail -1)
  D=$(dep "$SC")
  sub cyl_dmfgen       $D train_bench_cyl_ffm.sh
  sub cyl_senseiver    $D --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=Save_config/cylinder2d/config_baseline_Det_cyl.yaml     train_cyl_baseline.sh
  sub cyl_mlp_rbf      $D --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=Save_config/cylinder2d/config_baseline_MLPRBF_cyl.yaml  train_cyl_baseline.sh
  sub cyl_geofno       $D --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=Save_config/cylinder2d/config_baseline_GeoFNO_cyl.yaml  train_cyl_baseline.sh
  sub cyl_sit          $D --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=Save_config/cylinder2d/config_baseline_SiT_cyl.yaml     train_cyl_baseline.sh
  sub cyl_s3gm         $D --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=Save_config/cylinder2d/config_baseline_S3GM_cyl.yaml    train_cyl_baseline.sh
  S1=$(sub cyl_lfm_s1  $D --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=Save_config/cylinder2d/config_baseline_Gen_cyl.yaml,LFM_STAGE=1 train_cyl_baseline.sh | tail -1)
  sub cyl_lfm_s2       --dependency=afterok:$S1 --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=Save_config/cylinder2d/config_baseline_Gen_cyl.yaml,LFM_STAGE=2 train_cyl_baseline.sh
  sub cyl_classical    $D run_classical_cyl.sh
fi
echo "queue:"; squeue -u "$USER" -o "%.10i %.16j %.8T %.10M %.20E" | head -40
