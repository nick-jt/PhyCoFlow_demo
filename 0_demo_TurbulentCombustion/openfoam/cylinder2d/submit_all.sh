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
# Module tree is only visible on GPU nodes; sourcing the OF bashrc directly
# is what puts binaries on PATH and works on any node (shared filesystem).
module load openfoam/9-craympich 2>/dev/null || true
set +u
source /nopt/nrel/apps/gpu_stack/software/openfoam/ofoam9/OpenFOAM-9/etc/bashrc || true
set -u
command -v blockMesh >/dev/null || { echo "blockMesh not on PATH"; exit 1; }

RES=(60 80 100 150 200 250)
Re=${RES[$SLURM_ARRAY_TASK_ID]}
DEST=${CYL2D_DEST:-/projects/ammoniacomb/generative_reconstruction/cylinder2d/runs}
mkdir -p "$DEST"
bash "$(dirname "$0")/run_case.sh" "$Re" "$DEST"
