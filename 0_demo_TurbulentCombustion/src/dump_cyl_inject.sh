#!/bin/bash
#SBATCH --job-name=cyl_inject
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=120G
#SBATCH --output=cyl_inject_%j.log
# Cylinder mesh panels on the draw the FLEET scored. The canonical cylinder fleet
# ran on H100; the 23,800-point mesh draw is not reproducible on GH200 (CUDA
# randperm is SKU-dependent at this size; grid and Kolmogorov draws are not
# affected). The H100 draw for frame 300 survives in cyl_dmfgen.npz
# (idx_sum 5581141), so it is INJECTED here. The GH200-draw versions are moved
# to superseded_gh200draw/, not deleted.
set -uo pipefail
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
WT=/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion
cd $WT/src
OUT=$WT/Save_TrainedModel_pof/field_dumps; REF=$OUT/cyl_dmfgen.npz
mkdir -p $OUT/superseded_gh200draw
for f in cyl_classical.npz cyl_mlprbf.npz; do [ -e $OUT/$f ] && mv $OUT/$f $OUT/superseded_gh200draw/; done
RC=0
python dump_classical_gallery.py --dataset cylinder2d --sensor-indices-npz $REF || RC=1
python eval_kolm_ensemble.py --model mlp_rbf \
  --run-dir $WT/Save_TrainedModel/cylinder2d/baseline_mlp_rbf/Baseline_mlp_rbf_Stage1_DemoN72_20260906_082016 \
  --ckpt best --dump-frame 300 --n-obs-list 238 --cond-fields 0 1 --expect-val-len 600 \
  --K 8 --seed 0 --sensor-indices-npz $REF --dump-npz $OUT/cyl_mlprbf.npz || RC=1
python - <<'PY'
import numpy as np
fd="/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/Save_TrainedModel_pof/field_dumps/"
r=np.load(fd+"cyl_dmfgen.npz"); ri=np.asarray(r["sensor_indices"]); rf=np.asarray(r["sensor_field_ids"])
for f in ("cyl_classical","cyl_mlprbf","cyl_senseiver"):
    d=np.load(fd+f+".npz"); i=np.asarray(d["sensor_indices"]); fi=np.asarray(d["sensor_field_ids"])
    same=all(set(i[fi==k])==set(ri[rf==k]) for k in (0,1))
    print(f"{f:<15} idx_sum={int(i.sum())}  identical sensor set to fleet H100 draw: {same}")
PY
exit $RC
