#!/bin/bash
# Replicates submit_all.sh's environment EXACTLY (no module load — as on
# shared CPU nodes) and runs every binary the pipeline touches on a tiny case.
set -u
set +u
source /nopt/nrel/apps/gpu_stack/software/openfoam/ofoam9/OpenFOAM-9/etc/bashrc || true
set -u
export LD_LIBRARY_PATH="/nopt/nrel/apps/gpu_stack/compilers/linux-rhel8-zen4/gcc-12.3.0/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib:${LD_LIBRARY_PATH:-}"

TEMPLATE=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion/openfoam/cylinder2d/case_template
CASE=$HOME/.claude/jobs/3ac3fd02/tmp/cyl_env_smoke
rm -rf "$CASE"; mkdir -p "$CASE"; cp -r "$TEMPLATE/." "$CASE/"
sed -i "s/NU_VALUE/0.01/" "$CASE/constant/transportProperties"
cd "$CASE"

fail=0
step() {
    local name=$1; shift
    if "$@" > "log.$name" 2>&1; then
        echo "PASS  $name"
    else
        echo "FAIL  $name  (tail below)"
        tail -4 "log.$name"
        fail=1
    fi
}

step ldd_blockMesh bash -c 'ldd $(command -v blockMesh) | grep -q "not found" && exit 1 || exit 0'
step blockMesh blockMesh
step checkMesh checkMesh
step cellCentres postProcess -func writeCellCentres -time 0
sed -i 's/^endTime .*/endTime 0.02;/' system/controlDict
step pimpleFoam pimpleFoam
# forceCoeffs functionObject ran inside pimpleFoam; verify its output exists
if ls postProcessing/forceCoeffs1/*/* >/dev/null 2>&1; then
    echo "PASS  forceCoeffs output"
else
    echo "FAIL  forceCoeffs output missing"; fail=1
fi
# phase-2 controlDict sed (as run_case.sh does) + restart from latestTime
sed -i 's/^endTime .*/endTime         0.04;/' system/controlDict
sed -i 's/^writeInterval .*/writeInterval   0.01;/' system/controlDict
step pimpleFoam_phase2 pimpleFoam
n_dirs=$(ls -d 0.0* 2>/dev/null | wc -l)
echo "time dirs written in phase 2: $n_dirs"
[ "$n_dirs" -ge 2 ] || { echo "FAIL  phase-2 writes"; fail=1; }
echo "=== ENV TEST $( [ $fail -eq 0 ] && echo ALL PASS || echo HAD FAILURES ) ==="
exit $fail
