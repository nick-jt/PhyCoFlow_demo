#!/bin/bash
#SBATCH --job-name=op_smoke
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=100G
#SBATCH --output=op_smoke_%j.log
# Operator-axis smoke gate, 3 snapshots on the cheapest deterministic row.
# Checks, before any fleet compute is spent:
#   1. clean and noisy runs write DIFFERENT per-snapshot cache files
#   2. the noisy run carries its own protocol stamp, so the density figure
#      (which filters on the clean stamp) cannot pick it up
#   3. noise actually changes the score, and slab occlusion actually drops
#      the sensor count by the requested fraction
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
RD=$(ls -d ../Save_TrainedModel/kolmogorov2d/baseline_geofno/Baseline_geofno_Stage1_DemoN65_* | tail -1)
for OP in "" "--sensor-noise 0.1" "--sensor-occlusion 0.25"; do
  echo "===== operator: ${OP:-clean} ====="
  python eval_kolm_ensemble.py --model geofno --run-dir "$RD" \
      --dataset kolmogorov2d --split val --K 1 --n-snapshots 3 \
      --cond-fields 0 --n-obs-list 655 --stratify-blocks 1 \
      --out-prefix opsmoke --resume $OP 2>&1 | \
      grep -E "\[operator\]|\[seedcheck\]|\[RESULT\]|rel_l2_mean|\[out\]|Error|Traceback" | head -12
done
echo "===== cache files written ====="
ls "$RD"/Evaluation/ | grep -E "^crps_n655" | head -20
echo "===== protocol stamps ====="
for f in "$RD"/Evaluation/opsmoke_*.json; do
  python -c "import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[1].split('/')[-1],'->',d['protocol'],d.get('operator',{}).get('tag'),'relL2=%.5f'%d['summary']['aggregate']['rel_l2_mean'])" "$f"
done
