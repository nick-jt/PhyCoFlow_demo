#!/bin/bash
# Transient helper: watch SLURM job $1 until it leaves the queue, emitting
# state changes; then print the final sacct state and the dump dir listing.
JOB=${1:-18035182}
prev=""
while true; do
  st=$(squeue -j "$JOB" -h -o %T 2>/dev/null)
  if [ -z "$st" ]; then
    echo "JOB_ENDED $(sacct -j "$JOB" -n -o State%20,Elapsed 2>/dev/null | head -1)"
    ls /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion/Save_TrainedModel_pof/field_dumps/
    break
  fi
  if [ "$st" != "$prev" ]; then echo "state=$st"; prev=$st; fi
  sleep 60
done
