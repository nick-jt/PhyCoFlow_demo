#!/bin/bash
#SBATCH --job-name=cyl2d_gen
#SBATCH --partition=short
#SBATCH --account=ammoniacomb
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-5
#SBATCH --output=cyl2d_gen_%A_%a.log

set -euo pipefail
. /etc/profile.d/modules.sh 2>/dev/null || true
module load openfoam/9-craympich
# The module alone does not put binaries on PATH; source the OF environment.
set +u
source /nopt/nrel/apps/gpu_stack/software/openfoam/ofoam9/OpenFOAM-9/etc/bashrc || true
set -u

RES=(60 80 100 150 200 250)
Re=${RES[$SLURM_ARRAY_TASK_ID]}
DEST=${CYL2D_DEST:-/projects/ammoniacomb/generative_reconstruction/cylinder2d/runs}
mkdir -p "$DEST"
bash "$(dirname "$0")/run_case.sh" "$Re" "$DEST"
