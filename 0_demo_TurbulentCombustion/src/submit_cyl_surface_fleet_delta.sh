#!/bin/bash
# Cylinder SURFACE-TO-FIELD fleet (P0.5), DeltaAI. Run from src/:
#     bash submit_cyl_surface_fleet_delta.sh        # smoke-gated
#     SMOKE=0 bash submit_cyl_surface_fleet_delta.sh
# Same launchers as the canonical cylinder fleet; only the configs differ
# (Save_config/cylinder2d_surface/*, sensor_pool: surface, cond_fields [0,1,2],
# taps U{32..360}). Runs land in Save_TrainedModel/cylinder2d_surface/.
# Pre-registered predictions (handoff sec.2 P0.5) are in
# SURFACE_TASK_PREREGISTRATION.md -- do not edit them after results land.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
SMOKE=${SMOKE:-1}
MAP=fleet_jobs_delta.txt
sub() { local label=$1; shift; local jid; jid=$(sbatch --parsable "$@"); echo "$(date +%F_%T) $label $jid" | tee -a "$MAP"; echo "$jid"; }
dep() { [ -n "${1:-}" ] && echo "--dependency=afterok:$1" || true; }
C=Save_config/cylinder2d_surface
SK=""; [ "$SMOKE" = 1 ] && SK=$(sub smoke_cylsurf smoke_cyl_surface.sh | tail -1)
D=$(dep "$SK")
sub cylsurf_dmfgen     $D --export=ALL,CONFIG=$C/config_bench_cyl_surface_ffm.yaml,DEMO_NUM=103 train_bench_cyl_ffm.sh
sub cylsurf_senseiver  $D --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=$C/config_baseline_Det_cyl_surface.yaml    train_cyl_baseline.sh
sub cylsurf_mlp_rbf    $D --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=$C/config_baseline_MLPRBF_cyl_surface.yaml train_cyl_baseline.sh
sub cylsurf_geofno     $D --export=ALL,TRAINER=train_Det_Baseline.py,CONFIG=$C/config_baseline_GeoFNO_cyl_surface.yaml train_cyl_baseline.sh
sub cylsurf_sit        $D --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_SiT_cyl_surface.yaml    train_cyl_baseline.sh
sub cylsurf_s3gm       $D --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_S3GM_cyl_surface.yaml   train_cyl_baseline.sh
S1=$(sub cylsurf_lfm_s1 $D --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_Gen_cyl_surface.yaml,LFM_STAGE=1 train_cyl_baseline.sh | tail -1)
sub cylsurf_lfm_s2     --dependency=afterok:$S1 --export=ALL,TRAINER=train_Gen_Baseline.py,CONFIG=$C/config_baseline_Gen_cyl_surface.yaml,LFM_STAGE=2 train_cyl_baseline.sh
sub cylsurf_classical  $D run_classical_cyl_surface.sh
echo "queue:"; squeue -u "$USER" -o "%.10i %.16j %.8T %.10M %.20E" | grep -i 'cylsurf\|cyl2d\|classical' | head -20
