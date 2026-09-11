#!/bin/bash
#SBATCH --job-name=jhu_gallery
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=240G
#SBATCH --output=jhu_gallery_%j.log
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RC=0
python dump_jhu_gallery.py --methods fno3d || RC=1   # dmfgen + classical already written
# latent FM through its own driver; it draws sensors itself, so its idx_sum is
# checked against the recorded one afterwards rather than assumed
FD=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_TrainedModel_pof/field_dumps
[ -s $FD/jhu_latent_fm.npz ] || python dump_fields_baseline.py --method latent_fm \
    --run-dir /work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN24_20260828_164541 \
    --frame 153 --sensor-seed 103 --cond-fields 0 2 --n-obs 19531 --K 1 \
    --field-names Ux Uy Uz p --data-path /work/hdd/bilr/ntricard/datasets/JHU_4cubes_stride100.h5 --out $FD/jhu_latent_fm.npz || RC=1
python - <<'PY'
import numpy as np
fd="../Save_TrainedModel_pof/field_dumps/"
ref=int(np.load(fd+"jhu_sit.npz")["sensor_indices"].sum())
for f in ("jhu_dmfgen","jhu_fno3d","jhu_classical","jhu_latent_fm"):
    try: s=int(np.load(fd+f+".npz")["sensor_indices"].sum()); print(f, s, "MATCH" if s==ref else "DIFFERENT DRAW")
    except Exception as e: print(f, "missing:", e)
PY
exit $RC
