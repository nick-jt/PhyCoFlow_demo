#!/bin/bash
#SBATCH --job-name=guard_smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=100G
#SBATCH --output=guard_smoke_%j.log
# Integration smoke for artifact_guard through the REAL evaluator, on symlinked
# copies of two run dirs with their own empty Evaluation/ -- nothing canonical
# is touched. The dmf copy is pre-seeded with the real 50-frame
# sensor_sweep_dmfgen_n65.json under its hardcoded legacy name (incident #4).
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
B=/work/hdd/bilr/ntricard/guard_smoke_20260910
G="--model geofno --run-dir $B/geo --split val --K 1 --n-frames 3 --cond-fields 0 --n-obs-list 655 --stratify-blocks 1 --out-prefix gsm --no-figs"
run () { python eval_kolm_ensemble.py "$@" > $B/last.log 2>&1; echo $?; }
pass=0; fail=0
check () { if [ "$1" = "$2" ]; then echo "PASS  $3"; pass=$((pass+1)); else echo "FAIL  $3 (got '$1', want '$2')"; fail=$((fail+1)); fi; }

rc=$(run $G);                     check "$rc" 0 "S1 fresh clean write exits 0"
check "$(grep -c 'REFUSED' $B/last.log)" 0 "S1 no refusals"
check "$(python -c "import json;print(json.load(open('$B/geo/Evaluation/crps_n655_snap0.json')).get('protocol'))")" \
      kolm2d_matched_v1 "S1 cache record now carries its protocol"

rc=$(run $G --resume);            check "$rc" 0 "S2 same measurement re-run (resume) exits 0"
check "$(grep -c '\[resume\] snap=' $B/last.log)" 3 "S2 all 3 snapshots resumed from cache"
check "$(grep -c 'REJECT' $B/last.log)" 0 "S2 protocol-aware resume rejects nothing"

python - <<PY
import json; p="$B/geo/Evaluation/gsm_geofno_K1.json"
d=json.load(open(p)); d["n_obs"]=[65]; json.dump(d,open(p,"w"))   # a different prior measurement
PY
rc=$(run $G --resume);            check "$rc" 3 "S3 collision on aggregate -> exit 3"
check "$(grep -c 'REFUSED to overwrite gsm_geofno_K1.json' $B/last.log)" 1 "S3 guard names the refused file"
check "$(python -c "import json;print(json.load(open('$B/geo/Evaluation/gsm_geofno_K1.json'))['n_obs'])")" "[65]" "S3 existing file left untouched"
check "$(ls $B/geo/Evaluation/gsm_geofno_K1.CONFLICT_*.json 2>/dev/null | wc -l)" 1 "S3 new result preserved in a sidecar"

rc=$(run --model dmfgen --run-dir $B/dmf --ckpt best --K 8 --nfe 4 --cond-fields 0 --cond-source points \
         --n-obs-list 65 164 --stratify-blocks 1 --out-prefix gsm --n-frames 3 --seed 0 --op-seed 1000 --no-figs)
# Two densities, not one: a single-density dmfgen run takes the prefixed
# main-JSON branch (line ~1052) and never reaches the hardcoded legacy names.
# Only the multi-density sweep branch writes them -- the branch the fleet
# launcher always uses for dmfgen, and the one incident #4 went through.
check "$rc" 3 "S4 3-frame dmfgen vs 50-frame canonical (legacy name) -> exit 3"
check "$(grep -c 'REFUSED' $B/last.log)" 1 "S4 exactly one refusal (only the genuine collision)"
check "$(python -c "import json;print(json.load(open('$B/dmf/Evaluation/sensor_sweep_dmfgen_n164.json'))['n_frames'])" 2>/dev/null)" 3 "S4 non-colliding n164 legacy file written normally"
check "$(md5sum -c $B/seed.md5 >/dev/null 2>&1 && echo intact)" intact "S4 canonical 50-frame file byte-identical"
check "$(ls $B/dmf/Evaluation/sensor_sweep_dmfgen_n65.CONFLICT_*.json 2>/dev/null | wc -l)" 1 "S4 3-frame result preserved in a sidecar"

echo; echo "SMOKE: $pass passed, $fail failed"; [ $fail -eq 0 ] && echo "GATE PASS" || echo "GATE FAIL"
