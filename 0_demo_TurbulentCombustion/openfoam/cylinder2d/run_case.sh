#!/bin/bash
# Usage: run_case.sh <Re> <dest_root>
# Two-phase run: 0->120 transient (single write), 120->270 sampling
# (write every 0.5 => 300 snapshots over ~25 shedding cycles at Re=100).
set -euo pipefail

Re="$1"
DEST_ROOT="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE="$DEST_ROOT/Re${Re}"

mkdir -p "$CASE"
cp -r "$SCRIPT_DIR/case_template/." "$CASE/"
NU=$(python3 -c "print(1.0/${Re})")
sed -i "s/NU_VALUE/${NU}/" "$CASE/constant/transportProperties"

cd "$CASE"
echo "[run_case] Re=$Re nu=$NU case=$CASE"
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1 || echo "[run_case] checkMesh reported issues (see log.checkMesh)"
postProcess -func writeCellCentres -time 0 > log.cellCentres 2>&1 || true

echo "[run_case] phase 1 (transient 0->120)"
pimpleFoam > log.phase1 2>&1

sed -i 's/^endTime .*/endTime         270;/' system/controlDict
sed -i 's/^writeInterval .*/writeInterval   0.5;/' system/controlDict
echo "[run_case] phase 2 (sampling 120->270, write every 0.5)"
pimpleFoam > log.phase2 2>&1

n_dirs=$(ls -d [0-9]* 2>/dev/null | wc -l)
echo "[run_case] DONE Re=$Re; $n_dirs time directories"
