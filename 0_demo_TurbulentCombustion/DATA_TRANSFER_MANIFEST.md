# Direct Data Transfer Manifest — PoF Benchmark → new HPC
**Measured 2026-09-08.** Source host: Kestrel. Totals: 884 GB under
`/projects/ammoniacomb/generative_reconstruction/` + 86 GB of checkpoints under the repo.
**Recommended transfer: 83 GB (Tier 1) or 172 GB (Tier 1+2). The other ~800 GB should not move** —
it is raw source material that was already processed into the files in Tier 1.
(Tier totals are `du`-measured, not estimated; regenerate with
`bash tools/make_transfer_lists.sh` if paths change.)

Shorthand below: `$PROJ = /projects/ammoniacomb/generative_reconstruction`,
`$REPO = /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe`.

---

## Tier 1 — send these (83 GB measured). Nothing in the paper reproduces without them.

### 1a. Datasets (37 GB)
| Path | Size | Why |
|---|---|---|
| `$PROJ/jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5` | 5.9 G | **The** 3D benchmark file (cross-cube protocol; frames 0–149 train cubes 0–2, 150–199 val cube 3). Re-downloading from JHTDB took ~5 days under politeness limits. |
| `$PROJ/firebench3d/` | 19 G | Wildfire LES, merged u10+u12. Regenerable from GCS zarr only via a fiddly crop recipe. |
| `$PROJ/shift_wing/processed_v3/` | 3.9 G | Wing training set (673 cases, 600/73 split). **This is the processed output of the 554 GB raw tree — send this, not the raw.** |
| `$PROJ/cylinder2d/` | 7.3 G | Both H5s (mesh + grid) + manifest. Regenerable in ~2 h via OpenFOAM, but sending avoids standing up OpenFOAM at all. |
| `$PROJ/kolmogorov2d/` | 802 M | 2D Kolmogorov H5 + manifest. |

### 1b. Frozen paper checkpoints (25 GB)
Everything the current tables and figures were produced from:
| Path | Size |
|---|---|
| `$REPO/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*` | 1.2 G |
| `$REPO/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/baseline_{latent_fm,senseiver,sit,fno,s3gm,deeponetpp,classical}/` | ~15 G |
| `$REPO/0_demo_TurbulentCombustion/Save_TrainedModel/firebench/` | 7.0 G |
| `$REPO/0_demo_TurbulentCombustion/Save_TrainedModel/wing/` | 773 M |
| `$REPO/.claude/worktrees/pof2026-benchmark/0_demo_TurbulentCombustion/Save_TrainedModel/` (both 2D fleets + all eval JSONs) | 18 G |

> The 2D worktree tree is where every Kolmogorov/cylinder result lives, including the eval
> JSONs the paper tables read. Do not skip it.

---

## Tier 2 — send if bandwidth is cheap (≈50 GB)
| Path | Size | Why you might want it |
|---|---|---|
| `$PROJ/baselines/Gen4Turbulence/` | 44 G | Shipped checkpoints + its 3D `u.npy`; needed only to *re-run* the Gen4Turb row (already evaluated and frozen). |
| `$REPO/.../Save_TrainedModel/JHU/baseline_confild/` | 22 G | CoNFiLD latent-capacity sweep history; the reported row is frozen, this is provenance. |
| `$REPO/.../Save_TrainedModel/JHU/pointcloud_ffm/` (remaining ablation runs) | ~13 G | N15/N22/N33/kprior/K-sweep arms — the spectral-weight and backbone ablations. Reported numbers are in JSONs already in Tier 1; these are the weights behind them. |
| `$PROJ/baselines/{CoNFiLD,sparse-reconstruction,sparse-reconstruction-3d,Senseiver_OrchardLANL}/` | ~10 G | Reference implementations; all re-clonable from GitHub except the Kolmogorov npy (3.2 G, refigshare-able). |

---

## Tier 3 — do NOT transfer (≈830 GB)
| Path | Size | Why not |
|---|---|---|
| `$PROJ/shift_wing/data/` | **554 G** | Raw HuggingFace downloads. Already distilled into `processed_v3` (Tier 1). Only needed to re-preprocess with a *different* case selection. |
| `$PROJ/jhu_homogeneous_turbulence/outputfiles_scale/` | 108 G | The 1200-frame scale campaign — fed a 500³ capability demo that was never completed (P2 item). Re-downloadable if that experiment is revived. |
| `$PROJ/jhu_homogeneous_turbulence/outputfiles/` | 54 G | Includes `JHU_20cubes_plus_cube3.h5` (37 G) from the more-snapshots campaign. **No current result uses it.** |
| `$PROJ/kagglehub/` | 64 G | FireBench 2D slices — explicitly evaluated and rejected as the wrong modality. |
| `$PROJ/jhu.../outputfiles_diverse/JHU_cube{0,1,2,3}_stride100.h5` | 6 G | Per-cube components; redundant with the merged 4-cube file. |
| `$REPO/.../Save_TrainedModel/_legacy/`, `_failed_attempts/`, `baseline_deeponet{,_wide}/` | ~5 G | Superseded/retired arms. |

---

## Transfer mechanics

**Preferred — Globus** (both NREL and most university HPCs have endpoints; handles the 60 GB
unattended with checksums and restart):
1. Activate both endpoints in the Globus web UI.
2. Transfer the Tier-1 paths above. Enable *"sync — checksum"* and *"preserve timestamps"*.
3. Verify with the counts in §"Post-transfer verification" below.

**Fallback — rsync over SSH** (use the generated file lists, which encode exactly the tiers above):
```bash
# from the SOURCE host
rsync -avhP --files-from=0_demo_TurbulentCombustion/transfer_tier1.txt / \
      user@newhpc:/path/to/destination/
# Tier 2 likewise with transfer_tier2.txt
```
`transfer_tier1.txt` / `transfer_tier2.txt` are committed next to this file (absolute paths,
one per line, suitable for `rsync --files-from=` or `globus transfer --batch`).

**tar-and-ship** only if the link is flaky:
`tar -I 'pigz -p 8' -cf jhu_bench_tier1.tgz -T transfer_tier1.txt` (expect ~55 GB compressed;
the H5s are float32 and compress poorly, ~10%).

---

## Post-transfer verification
```bash
# 1. sizes/counts match
du -sh <dest>/projects/.../{firebench3d,cylinder2d,kolmogorov2d,shift_wing/processed_v3}

# 2. the three H5s open and carry the expected shapes/attrs
python - <<'EOF'
import h5py
for p, shape in [("JHU_4cubes_stride100.h5", (1,200,1953125,1,1,4)),
                 ("Cylinder2D_mesh.h5",      (1,1800,23800,1,1,3)),
                 ("Kolmogorov2D_shu_stride4.h5", (1,3200,65536,1,1,1))]:
    with h5py.File(p) as f:
        print(p, f["fields"].shape, "OK" if f["fields"].shape == shape else "MISMATCH")
EOF

# 3. split boundaries still exact (catches truncated transfers)
JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 python -c "
from helpers_baseline import TurbulentCombustionH5Dataset as D
d=D('Kolmogorov2D_shu_stride4.h5',split='train',train_ratio=0.8,seed=42,time_stride=1,
    field_names=('vorticity',),stats_path='/tmp/s.json'); print(len(d))"   # must print 2560

# 4. one checkpoint loads and reproduces a known number
#    DMF-Gen N29 on JHU cube 3 must give rel-L2 0.5926 / CRPS 0.2908 (n=50, K=8, NFE=4)
```

**If the new HPC has different GPU SKUs than H100:** sensor draws are SKU-dependent, so
re-run the canonical evals for *all* methods rather than mixing new numbers with the
transferred JSONs. Budget ~1 day of GPU for the full re-baseline (2D is minutes; 3D fleet
is the cost). The transferred JSONs stay valid as the reference to diff against.
