#!/bin/bash
#SBATCH --job-name=kolm_geofno_dump
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=100G
#SBATCH --output=kolm_geofno_dump_%j.log
# The Kolmogorov Geo-FNO row is now the budget-matched run (decided 2026-09-10).
# The spectra figure and the gallery read kolm_geofno.npz, which came from the
# 4x-budget checkpoint, so it is re-dumped from the matched one on the same
# frame / sensors / seed. The old dump is moved aside, not deleted.
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
OUT=../Save_TrainedModel_pof/field_dumps
mkdir -p $OUT/superseded_4xbudget
[ -e $OUT/kolm_geofno.npz ] && mv $OUT/kolm_geofno.npz $OUT/superseded_4xbudget/
python eval_kolm_ensemble.py --model geofno \
  --run-dir ../Save_TrainedModel/kolmogorov2d_fullbudget/baseline_geofno/Baseline_geofno_Stage1_DemoN265_20260910_063432 \
  --ckpt best --dump-frame 256 --n-obs-list 655 --seed 0 --dump-npz $OUT/kolm_geofno.npz
python -c "import numpy as np,json;d=np.load('$OUT/kolm_geofno.npz');m=json.loads(str(d['meta']));print('idx_sum',int(d['sensor_indices'].sum()),'(want 22128955)  run',m['run_dir'].split('/')[-1])"
