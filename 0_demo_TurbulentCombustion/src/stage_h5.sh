#!/bin/bash
# Stage the JHU HDF5 to node-local NVMe and echo the path to use.
# Echoes the SHARED path (and warns) if anything about the copy is wrong, so a
# staging failure degrades to the previous behaviour instead of killing the run.
# Usage:  DATA_LOCAL=$(bash stage_h5.sh "$SRC" "$SLURM_JOB_ID")
set -u
SRC="$1"
JOBID="${2:-$$}"
DEST_DIR="/tmp/${USER}/jhu_${JOBID}"
DEST="${DEST_DIR}/$(basename "$SRC")"

if [ ! -f "$SRC" ]; then
  echo "$SRC"; echo "[stage] WARN: source missing, using shared path" >&2; exit 0
fi
SRC_SZ=$(stat -c %s "$SRC" 2>/dev/null || echo 0)
AVAIL=$(df -B1 --output=avail /tmp 2>/dev/null | tail -1 | tr -d ' ')
if [ -z "$AVAIL" ] || [ "$AVAIL" -lt $((SRC_SZ * 2)) ]; then
  echo "$SRC"; echo "[stage] WARN: /tmp has ${AVAIL:-?} B free, need $((SRC_SZ*2)); using shared path" >&2; exit 0
fi

mkdir -p "$DEST_DIR" 2>/dev/null || { echo "$SRC"; echo "[stage] WARN: mkdir failed; using shared path" >&2; exit 0; }
T0=$(date +%s)
if ! cp "$SRC" "$DEST" 2>/dev/null; then
  rm -rf "$DEST_DIR"; echo "$SRC"; echo "[stage] WARN: copy failed; using shared path" >&2; exit 0
fi
DST_SZ=$(stat -c %s "$DEST" 2>/dev/null || echo 0)
if [ "$DST_SZ" != "$SRC_SZ" ]; then
  rm -rf "$DEST_DIR"; echo "$SRC"
  echo "[stage] WARN: size mismatch src=$SRC_SZ dst=$DST_SZ; using shared path" >&2; exit 0
fi
echo "$DEST"
echo "[stage] OK: staged ${SRC_SZ} B to $DEST in $(( $(date +%s) - T0 )) s" >&2
