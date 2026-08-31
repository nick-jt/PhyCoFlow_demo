#!/bin/bash
#SBATCH --job-name=ckptvar_eval
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=200G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
DEMO=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
SHARED=/projects/ammoniacomb/generative_reconstruction/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5
L=calib_ckptvar_eval_${SLURM_JOB_ID}.log
# Run dir is resolved HERE, at run time -- the trainer creates it with a
# timestamp that does not exist at submission.
RUN=${CKPTVAR_RUN:-$(ls -dt $DEMO/Save_TrainedModel/JHU/pointcloud_ffm/ckptvar_n29_DemoN29_* 2>/dev/null | head -1)}
MINCK=${CKPTVAR_MIN:-8}
NSNAP=${CKPTVAR_NSNAP:-12}
QSUB=${CKPTVAR_QSUB:-400000}
echo "host=$(hostname) run=$RUN start=$(date)" > $L
[ -n "$RUN" ] && [ -d "$RUN" ] || { echo "FATAL: no ckptvar run dir found" >> $L; exit 3; }
N=$(ls $RUN/archive/ckpt_epoch*.pt 2>/dev/null | wc -l)
echo "archived checkpoints: $N (require >= $MINCK)" >> $L
[ "$N" -ge "$MINCK" ] || { echo "FATAL: only $N checkpoints; refusing to average a partial window" >> $L; exit 3; }
# The trainer records the STAGED (node-local) data path in args.json; that path
# is gone by the time this job runs. Point it back at the shared file.
python - "$RUN" "$SHARED" >> $L 2>&1 <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], "args.json")
a = json.load(open(p))
if not os.path.exists(a.get("data", "")):
    print(f"[fixup] data path {a.get('data')} missing -> {sys.argv[2]}")
    a["data"] = sys.argv[2]
    json.dump(a, open(p, "w"), indent=2)
else:
    print(f"[fixup] data path OK: {a.get('data')}")
PY
mkdir -p $RUN/Evaluation
FAIL=0
for CK in $RUN/archive/ckpt_epoch*.pt; do
  EP=$(basename $CK .pt | sed 's/ckpt_epoch//')
  echo "##### checkpoint epoch $EP" >> $L
  python ensemble_eval.py --run-dir "$RUN" --ckpt "archive/$(basename $CK)" \
      --K 8 --n-steps 4 --n-snapshots $NSNAP --no-clamp --no-figs \
      --cond-fields 0 2 --n-obs 19531 19531 \
      --query-subset $QSUB --chunk 262144 \
      --out "$RUN/Evaluation/ckptvar_ep${EP}.json" >> $L 2>&1 || FAIL=$((FAIL+1))
done
M=$(ls $RUN/Evaluation/ckptvar_ep*.json 2>/dev/null | wc -l)
echo "evaluated $M / $N checkpoints ($FAIL failures)" >> $L
[ "$M" -ge "$MINCK" ] || { echo "FATAL: only $M metric files" >> $L; exit 4; }
python -u calib_ckptvar_analyze.py "$RUN/Evaluation" >> $L 2>&1
RC=$?
echo "end=$(date) rc=$RC" >> $L
exit $RC
