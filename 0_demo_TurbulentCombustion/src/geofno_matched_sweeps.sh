#!/bin/bash
#SBATCH --job-name=geofno_matched
#SBATCH --time=01:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=100G
#SBATCH --output=geofno_matched_%j.log
# Budget-matched Geo-FNO (kolmogorov2d_fullbudget, 625 epochs = 50k steps,
# canonical eval 0.4165): the density sweep and the operator matrix the paper
# currently reports only for the 4x-budget checkpoint. Everything lands in the
# matched run's own Evaluation/ with density- or operator-qualified names, so
# neither canonical row is touched. Input for the Geo-FNO canonical-row decision.
set -u
cd $SLURM_SUBMIT_DIR
grep -q 'kolmogorov2d_fullbudget)' eval_kolm_fleet.sh || { echo "launcher has no fullbudget case"; exit 1; }
RC=0
for N in 65 164 1965 6554; do
  NOBS_OVERRIDE=$N DATASET=kolmogorov2d_fullbudget MODELS=geofno bash eval_kolm_fleet.sh || RC=1
done
for OP in "--sensor-noise 0.1" "--sensor-noise 0.3" "--sensor-occlusion 0.25"; do
  OPERATOR="$OP" DATASET=kolmogorov2d_fullbudget MODELS=geofno bash eval_kolm_fleet.sh || RC=1
done
source ~/envs/jhtdb
python - <<'PY'
import json, glob
print("=== budget-matched Geo-FNO: every eval written ===")
for p in sorted(glob.glob("../Save_TrainedModel/kolmogorov2d_fullbudget/baseline_geofno/*/Evaluation/kolm_fleet_full*.json")):
    d = json.load(open(p)); a = d["summary"]["aggregate"]
    print(f"  {p.split('/')[-1]:<44} n_obs={d['n_obs']} protocol={d['protocol']:<32} relL2={a['rel_l2_mean']:.4f}")
PY
exit $RC
