#!/bin/bash
# Submit N dependent segments of a resumable sbatch script (afterany chain).
# Engaging's mit_normal_gpu caps at 6h; every trainer here resumes from
# last.pt (--RELOAD / --reload), so a chain of K segments gives K*6h of
# training with no lost work.
#
# Usage: [FIRST_DEP=afterok:<jobid>] submit_chain.sh <script.sh> <n_segments> [extra sbatch args...]
# FIRST_DEP puts a dependency on the first segment only (e.g. the merge job).
set -eu
SCRIPT=$1; N=$2; shift 2
PREV=""
for i in $(seq 1 "$N"); do
  if [ -z "$PREV" ]; then
    if [ -n "${FIRST_DEP:-}" ]; then
      PREV=$(sbatch --parsable --dependency=$FIRST_DEP "$@" "$SCRIPT")
    else
      PREV=$(sbatch --parsable "$@" "$SCRIPT")
    fi
  else
    PREV=$(sbatch --parsable --dependency=afterany:$PREV "$@" "$SCRIPT")
  fi
  echo "segment $i: job $PREV"
done
