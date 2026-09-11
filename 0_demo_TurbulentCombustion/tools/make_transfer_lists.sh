#!/bin/bash
# Emit rsync/globus-compatible absolute-path lists for the transfer tiers.
PROJ=/projects/ammoniacomb/generative_reconstruction
REPO=/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe
WT=$REPO/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion
MAIN=$REPO/0_demo_TurbulentCombustion
OUT=$WT

{
  echo "$PROJ/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5"
  echo "$PROJ/firebench3d"
  echo "$PROJ/shift_wing/processed_v3"
  echo "$PROJ/cylinder2d"
  echo "$PROJ/kolmogorov2d"
  for d in "$MAIN"/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*; do
      [ -d "$d" ] && echo "$d"
  done
  for b in latent_fm senseiver sit fno s3gm deeponetpp classical; do
      [ -d "$MAIN/Save_TrainedModel/JHU/baseline_$b" ] && echo "$MAIN/Save_TrainedModel/JHU/baseline_$b"
  done
  echo "$MAIN/Save_TrainedModel/firebench"
  echo "$MAIN/Save_TrainedModel/wing"
  echo "$WT/Save_TrainedModel"
  echo "$WT/Save_TrainedModel_pof"
} > "$OUT/transfer_tier1.txt"

{
  echo "$PROJ/baselines/Gen4Turbulence"
  echo "$PROJ/baselines/CoNFiLD"
  echo "$PROJ/baselines/sparse-reconstruction"
  echo "$PROJ/baselines/sparse-reconstruction-3d"
  echo "$PROJ/baselines/Senseiver_OrchardLANL"
  echo "$MAIN/Save_TrainedModel/JHU/baseline_confild"
  echo "$MAIN/Save_TrainedModel/JHU/pointcloud_ffm"
} > "$OUT/transfer_tier2.txt"

echo "--- tier1 ($(wc -l < "$OUT/transfer_tier1.txt") entries) ---"
while read -r p; do
    [ -e "$p" ] && printf "%8s  %s\n" "$(du -sh "$p" 2>/dev/null | cut -f1)" "$p" \
                || printf "%8s  %s\n" "MISSING" "$p"
done < "$OUT/transfer_tier1.txt"
echo "--- tier1 total ---"
du -shc $(cat "$OUT/transfer_tier1.txt") 2>/dev/null | tail -1
echo "--- tier2 total ---"
du -shc $(cat "$OUT/transfer_tier2.txt") 2>/dev/null | tail -1
