#!/bin/bash
# Post-transfer resume: submit ONLY the evaluation work that the transferred
# artifacts do not already contain. Trains nothing, recomputes nothing.
#
#   bash resume_after_transfer.sh            # inventory + submit what is missing
#   DRY=1 bash resume_after_transfer.sh      # inventory only, submit nothing
#
# Rules:
#  * a model leg is submitted only if its run dir with best.pt EXISTS and its
#    fleet JSON is MISSING (a transferred JSON is taken as final);
#  * eval_kolm_ensemble.py runs with --resume, so any transferred
#    crps_snapN.json short-circuits that snapshot instead of recomputing it;
#  * sensor layouts are bit-identical across H100/GH200 (verified: Kolmogorov
#    frame 256 idx_sum 22128955 reproduced here), so mixing transferred and
#    locally computed accuracy rows is sound. COST fields are not portable --
#    every JSON records its own `gpu`, and any row recomputed here must be
#    reported as GH200 timing, never merged into an H100 cost column.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT=$(cd .. && pwd)
STM=$ROOT/Save_TrainedModel
MAP=fleet_jobs_delta.txt
DRY=${DRY:-0}

sub() {  # sub <label> <sbatch args...>
  local label=$1; shift
  if [ "$DRY" = 1 ]; then echo "  [dry] would submit $label: $*"; return; fi
  local jid; jid=$(sbatch --parsable "$@") || { echo "  [!] submit failed: $label"; return 1; }
  echo "$(date +%F_%T) $label $jid (post-transfer resume)" | tee -a "$MAP"
}

run_dir() {  # run_dir <dataset> <model> -> path or empty
  local ds=$1 m=$2 pat
  case $m in
    dmfgen)     pat="$STM/$ds/pointcloud_ffm/bench_*_DemoN*" ;;
    latent_fm)  pat="$STM/$ds/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN*" ;;
    sit)        pat="$STM/$ds/baseline_sit/Baseline_sit_Stage1_DemoN*" ;;
    senseiver)  pat="$STM/$ds/baseline_det/Baseline_senseiver_Stage1_DemoN*" ;;
    mlp_rbf)    pat="$STM/$ds/baseline_mlp_rbf/Baseline_mlp_rbf_Stage1_DemoN*" ;;
    geofno)     pat="$STM/$ds/baseline_geofno/Baseline_geofno_Stage1_DemoN*" ;;
    s3gm)       pat="$STM/$ds/baseline_s3gm/Baseline_s3gm_Stage1_DemoN*" ;;
    *) echo ""; return ;;
  esac
  local d; d=$(ls -d $pat 2>/dev/null | tail -1)
  [ -n "$d" ] && [ -f "$d/best.pt" ] && echo "$d" || echo ""
}

has_json() {  # has_json <run_dir> <prefix> <model>
  local d=$1 p=$2 m=$3 tag
  tag=$(echo "$m" | sed -e 's/latent_fm/latentfm/' -e 's/mlp_rbf/mlprbf/')
  ls "$d"/Evaluation/${p}_${tag}_K*.json >/dev/null 2>&1
}

echo "=== inventory of transferred artifacts ==="
declare -A TODO FOUND
for ds in kolmogorov2d cylinder2d; do
  pre=$([ "$ds" = kolmogorov2d ] && echo kolm_fleet || echo cyl_fleet)
  todo=""; found=0
  for m in dmfgen senseiver mlp_rbf geofno sit s3gm latent_fm; do
    d=$(run_dir "$ds" "$m")
    if [ -z "$d" ]; then
      printf '  %-14s %-10s MISSING (no run dir with best.pt)\n' "$ds" "$m"
    elif found=1; has_json "$d" "$pre" "$m"; then
      n=$(ls "$d"/Evaluation/crps_snap*.json 2>/dev/null | wc -l)
      printf '  %-14s %-10s JSON present -> skip (%s crps snaps, %s)\n' "$ds" "$m" "$n" "$(basename "$d")"
    else
      n=$(ls "$d"/Evaluation/crps_snap*.json 2>/dev/null | wc -l)
      printf '  %-14s %-10s EVAL NEEDED (%s crps snaps already resumable, %s)\n' "$ds" "$m" "$n" "$(basename "$d")"
      todo="$todo $m"
    fi
  done
  TODO[$ds]=$(echo "$todo" | xargs)
  FOUND[$ds]=$found
done

echo
echo "=== submissions ==="
for ds in kolmogorov2d cylinder2d; do
  if [ -n "${TODO[$ds]}" ]; then
    sub "eval_${ds}_posttransfer" --time=06:00:00 \
        --export=ALL,DATASET=$ds,MODELS="${TODO[$ds]}" eval_kolm_fleet.sh
  elif [ "${FOUND[$ds]}" = 1 ]; then
    echo "  $ds: nothing to evaluate -- every model's fleet JSON was transferred"
  else
    echo "  $ds: SKIPPED -- no run dirs arrived for this dataset (nothing to evaluate yet)"
  fi
done

# --- protocol ablation: only the arms whose JSONs are absent -----------------
LP=$STM/kolmogorov2d/litprotocol
NEED_ARMS=""
for arm in b_dmfgen b_idw uniform b_senseiver b_sit uniform_b_dmfgen; do
  ls "$LP"/*${arm}*.json >/dev/null 2>&1 || NEED_ARMS="$NEED_ARMS $arm"
done
NEED_ARMS=$(echo "$NEED_ARMS" | xargs)
if [ -n "$NEED_ARMS" ] && [ -n "$(run_dir kolmogorov2d dmfgen)" ]; then
  sub eval_litprotocol_posttransfer --time=04:00:00 --export=ALL,ARMS="$NEED_ARMS" eval_kolm_litprotocol.sh
else
  echo "  litprotocol: ${NEED_ARMS:-all arms present} ${NEED_ARMS:+(needs the kolm DMF-Gen run dir)}"
fi

# --- K-sweep and gallery dumps (both self-skip existing outputs) -------------
KD=$(run_dir kolmogorov2d dmfgen)
if [ -n "$KD" ] && ! ls "$KD"/Evaluation/*K64*.json >/dev/null 2>&1; then
  sub eval_ksweep_posttransfer --time=04:00:00 eval_kolm_ksweep.sh
else
  echo "  ksweep: ${KD:+present or done}${KD:-no kolm DMF-Gen run dir}"
fi
if [ "${FOUND[kolmogorov2d]}" = 1 ] || [ "${FOUND[cylinder2d]}" = 1 ]; then
  sub gallery_dumps_posttransfer --time=02:00:00 dump_kolm_gallery.sh   # skips existing npz
else
  echo "  gallery dumps: skipped (no learned run dirs yet)"
fi

echo
echo "=== queue ==="
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.24E" | head -30
