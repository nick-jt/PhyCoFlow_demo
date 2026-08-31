#!/bin/bash
#SBATCH --job-name=ckptvar_n29
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=200G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
DEMO=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion
L=calib_ckptvar_train_${SLURM_JOB_ID}.log
T0=$(date +%s)
echo "host=$(hostname) start=$(date) job=$SLURM_JOB_ID" > $L

# ---- stage the H5 to node-local NVMe, size-checked, with fallback ----------
SRC=/projects/ammoniacomb/generative_reconstruction/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5
STAGE=/tmp/$USER/jhu_${SLURM_JOB_ID}
DATA=$SRC
mkdir -p $STAGE
SZ=$(stat -c %s $SRC)
AVAIL=$(df -B1 --output=avail /tmp | tail -1)
echo "stage: src_bytes=$SZ avail_bytes=$AVAIL" >> $L
if [ "$AVAIL" -gt $((SZ + 20000000000)) ]; then
  TS=$(date +%s)
  cp $SRC $STAGE/ && DSZ=$(stat -c %s $STAGE/$(basename $SRC)) || DSZ=0
  if [ "$DSZ" = "$SZ" ]; then
    DATA=$STAGE/$(basename $SRC)
    echo "stage: OK $(( $(date +%s) - TS ))s -> $DATA" >> $L
  else
    echo "stage: size mismatch ($DSZ vs $SZ); using shared path" >> $L
    rm -f $STAGE/$(basename $SRC)
  fi
else
  echo "stage: not enough local space; using shared path" >> $L
fi

# ---- job-specific config: N29 verbatim, new save_dir + staged data ---------
CFG=Save_config/calib_ckptvar_${SLURM_JOB_ID}.yaml
sed -e "s|^data:.*|data: \"$DATA\"|" \
    -e "s|^save_dir:.*|save_dir: \"Save_TrainedModel/JHU/pointcloud_ffm/ckptvar_n29\"|" \
    $DEMO/Save_config/config_iclr_jhu_xcube_spec02.yaml > $DEMO/$CFG
grep -E "^(data|save_dir|epochs|batch_size|spectral_weight|eval_every|compile_model):" $DEMO/$CFG >> $L

# ---- checkpoint-window archiver (epochs 5700..6000 every 30) ---------------
python -u calib_archive_window.py \
  --glob "$DEMO/Save_TrainedModel/JHU/pointcloud_ffm/ckptvar_n29_DemoN29_*" \
  --start 5700 --stop 6000 --every 30 >> calib_ckptvar_arch_${SLURM_JOB_ID}.log 2>&1 &
ARCH=$!

# ---- train (fresh, no RELOAD) ---------------------------------------------
CUDA_VISIBLE_DEVICES=0 python -u train_pointcloud_ffm.py \
    --config $CFG --Demo-Num 29 >> $L 2>&1
RC=$?
TRAIN_WALL=$(( $(date +%s) - T0 ))
wait $ARCH || true

# ---- instrumentation: steps, compute time, duty cycle ---------------------
RUN=$(ls -dt $DEMO/Save_TrainedModel/JHU/pointcloud_ffm/ckptvar_n29_DemoN29_* | head -1)
python - "$RUN" "$TRAIN_WALL" >> $L 2>&1 <<'PY'
import csv, glob, sys, json
import numpy as np
run, wall = sys.argv[1], float(sys.argv[2])
c = sorted(glob.glob(run + "/Loss_*/losses.csv"))[-1]
r = list(csv.DictReader(open(c)))
et = np.array([float(x["epoch_time_s"]) for x in r])
n_ep = len(r); steps = n_ep * 8
compute = et.sum(); warm = et[0]
print(f"[instr] epochs={n_ep} optimizer_steps={steps}")
print(f"[instr] summed_loop_time={compute/3600:.3f} h (first epoch {warm:.0f}s = compile warmup)")
print(f"[instr] steady median epoch {np.median(et):.3f}s -> {np.median(et)/8:.4f} s/step")
print(f"[instr] process wall={wall/3600:.3f} h  duty(loop/wall)={compute/wall:.3f}")
json.dump({"epochs": n_ep, "steps": steps, "loop_h": compute/3600,
           "median_s_per_step": float(np.median(et))/8, "wall_h": wall/3600,
           "duty_loop_over_wall": compute/wall},
          open(run + "/instrumentation.json", "w"), indent=2)
PY
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State,MaxRSS -X >> $L 2>&1
rm -rf $STAGE
echo "end=$(date) rc=$RC" >> $L
exit $RC
