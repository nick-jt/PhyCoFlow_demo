#!/bin/bash
# Send the five benchmark datasets (~26 GB) to NCSA Delta.
# Run from the Kestrel login node inside tmux/screen. Resumable: just re-run.
set -u
DEST=${DEST:-ntricard@dtai-login.delta.ncsa.illinois.edu:/work/hdd/bilr/ntricard/datasets}
P=/projects/ammoniacomb/generative_reconstruction

rsync -avhP --partial \
  "$P/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5" \
  "$P/firebench3d/FireBench_u10u12_merged.h5" \
  "$P/kolmogorov2d" \
  "$P/cylinder2d" \
  "$DEST/"

rsync -avhP --partial "$P/shift_wing/processed_v3" "$DEST/shift_wing_processed_v3"

echo "done — verify with: ssh <delta> 'du -sh /work/hdd/bilr/ntricard/datasets/*'"
