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
# The OF9 build links the nvhpc runtime (libnvf.so); its module only exists
# on GPU nodes but the libraries live on the shared filesystem.
export LD_LIBRARY_PATH="/nopt/nrel/apps/gpu_stack/compilers/linux-rhel8-zen4/gcc-12.3.0/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib:${LD_LIBRARY_PATH:-}"
command -v blockMesh >/dev/null || { echo "blockMesh not on PATH"; exit 1; }
blockMesh -help >/dev/null 2>&1 || { echo "blockMesh cannot execute (missing runtime libs?)"; exit 1; }

RES=(60 80 100 150 200 250)
Re=${RES[$SLURM_ARRAY_TASK_ID]}
DEST=${CYL2D_DEST:-/projects/ammoniacomb/generative_reconstruction/cylinder2d/runs}
mkdir -p "$DEST"
# dirname $0 points at slurmd's spool copy; the real case dir is where we submitted from
bash "$SLURM_SUBMIT_DIR/run_case.sh" "$Re" "$DEST"
