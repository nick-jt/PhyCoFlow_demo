#!/bin/bash
#SBATCH --job-name=cyl_mesh_gh200
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=100G
#SBATCH --output=cyl_mesh_gh200_%j.log
# Re-score the three mesh-native cylinder rows on GH200 so every cylinder mesh
# row -- learned and classical -- shares ONE sensor draw (DELTA_STATUS sec.26:
# the 23,800-cell mesh draw is not reproducible H100 -> GH200). Runs on
# symlinked copies with EMPTY Evaluation/ dirs under cylinder2d_gh200rescore/,
# and without --resume: the H100 per-snapshot caches and canonical rows are
# untouched, and fleet_select (canonical key only) never sees these files.
# Expected draw at frame 300: idx_sum 5594792 (= the GH200 classical draw).
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
N=../Save_TrainedModel/cylinder2d_gh200rescore
COMMON="--ckpt best --K 8 --cond-fields 0 1 --expect-val-len 600 --cond-source points --stratify-blocks 2 --out-prefix cyl_fleet --n-frames 50 --seed 0 --op-seed 1000 --no-figs"
RC=0
python eval_kolm_ensemble.py --model dmfgen    --run-dir $N/pointcloud_ffm/bench_cyl_v1_DemoN102_20260906_082022 --nfe 4 --n-obs-list 238 $COMMON || RC=1
python eval_kolm_ensemble.py --model senseiver --run-dir $N/baseline_det/Baseline_senseiver_Stage1_DemoN71_20260906_082017 --n-obs-list 238 $COMMON || RC=1
python eval_kolm_ensemble.py --model mlp_rbf   --run-dir $N/baseline_mlp_rbf/Baseline_mlp_rbf_Stage1_DemoN72_20260906_082016 --n-obs-list 238 $COMMON || RC=1
echo "=== GH200 re-score vs canonical H100 rows ==="
python - <<'PY'
import json, glob
canon = {"dmfgen": 0.221, "senseiver": 0.145, "mlp_rbf": 0.388}
for p in sorted(glob.glob("../Save_TrainedModel/cylinder2d_gh200rescore/*/*/Evaluation/cyl_fleet_*.json")):
    d = json.load(open(p)); a = d["summary"]["aggregate"]
    s = {x["snapshot"]: x["idx_sum"] for x in d["snapshots"]}
    print(f"{d['model']:<10} relL2 {a['rel_l2_mean']:.4f} (H100 {canon.get(d['model'])})  "
          f"frame-300 idx_sum {s.get(300)} {'= GH200 draw' if s.get(300)==5594792 else '!! unexpected'}  gpu={d.get('gpu')}")
PY
exit $RC
