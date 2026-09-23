#!/bin/bash
#SBATCH --job-name=dmfgen_sweep_clean
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=100G
#SBATCH --output=dmfgen_sweep_clean_%j.log
# Rebuild the CLEAN DMF-Gen density sweep, destroyed on 2026-09-10 when the
# operator fleet runs wrote over it: the dmfgen branch hardcoded
# "sensor_sweep_dmfgen_n<N>.json" and ignored the operator suffix. DMF-Gen keeps
# no per-snapshot cache (resume is disabled for that model), so unlike the S3GM
# incident this cannot be re-aggregated and has to be recomputed.
#
# Expected on success (the clean values read off these files before they were
# overwritten -- if the run disagrees, the run is right and this comment is the
# thing to distrust):
#   65:0.8871  164:0.7403  655:0.4874  1965:0.3645  6554:0.2582
# and every file must come back stamped protocol=kolm2d_matched_v1 with
# sensors == n_obs (not 0.75*n_obs, which is the occlusion signature).
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RD=$(ls -d ../Save_TrainedModel/kolmogorov2d/pointcloud_ffm/bench_kolm_v1_DemoN101_* | tail -1)
python eval_kolm_ensemble.py --model dmfgen --run-dir "$RD" \
    --ckpt best --K 8 --nfe 4 --cond-fields 0 --cond-source points \
    --n-obs-list 65 164 655 1965 6554 \
    --stratify-blocks 1 --out-prefix kolm_fleet --expect-val-len 640 \
    --n-frames 50 --seed 0 --op-seed 1000 --no-figs
RC=$?
echo "=== verification ==="
python - "$RD" <<'PY'
import json,sys,glob,os
ok=True
for p in sorted(glob.glob(sys.argv[1]+"/Evaluation/sensor_sweep_dmfgen_n*.json")):
    d=json.load(open(p)); s=d["snapshots"][0]
    n=d["n_obs"][0]; good = d["protocol"]=="kolm2d_matched_v1" and s["sensors"]==n
    ok &= good
    print(f"{os.path.basename(p):<34} n={n:<6} sensors={s['sensors']:<6} "
          f"relL2={d['summary']['aggregate']['rel_l2_mean']:.4f} "
          f"{'OK' if good else 'BAD'}")
print("GATE", "PASS" if ok else "FAIL")
PY
exit $RC
