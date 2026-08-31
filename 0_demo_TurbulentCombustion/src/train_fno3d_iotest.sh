#!/bin/bash
#SBATCH --job-name=fno3d_iotest
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=128G

# A/B the I/O share: 4 epochs reading the shared Lustre path, then 4 epochs
# reading a node-local NVMe copy, same node, same config, back to back.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L="train_fno3d_iotest_${SLURM_JOB_ID}.log"
CFG=Save_config/config_baseline_fno3d_smoke.yaml
SHARED=$(grep -oP '^data\s*:\s*"\K[^"]+' ../${CFG})
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST" > "$L"

# --- A: shared Lustre ---
echo "=== A: shared Lustre ($SHARED) ===" >> "$L"
python -u train_pointcloud_ffm.py --config ${CFG} --Demo-Num 92 >> "$L" 2>&1

# --- B: node-local NVMe ---
LOCAL=$(bash stage_h5.sh "$SHARED" "$SLURM_JOB_ID" 2>>"$L")
echo "=== B: node-local ($LOCAL) ===" >> "$L"
JC="/tmp/${USER}/iotest_${SLURM_JOB_ID}.yaml"
python - "$SHARED" "$LOCAL" "../${CFG}" "$JC" <<'PY'
import sys
sh, lo, src, dst = sys.argv[1:5]
s = open(src).read().replace(f'"{sh}"', f'"{lo}"')
s = s.replace('Demo_Num: 91', 'Demo_Num: 93')
open(dst, 'w').write(s)
PY
python -u train_pointcloud_ffm.py --config "$JC" --Demo-Num 93 >> "$L" 2>&1

echo "=== epoch times ===" >> "$L"
for D in 92 93; do
  F=$(ls ../Save_TrainedModel/JHU/baseline_fno/fno3d_smoke_DemoN${D}_*/Loss_*/losses.csv 2>/dev/null | head -1)
  [ -n "$F" ] && { echo "DemoN${D}:" >> "$L"; cat "$F" >> "$L"; }
done
rm -rf "/tmp/${USER}/jhu_${SLURM_JOB_ID}" "$JC" 2>/dev/null
