# Tier 2 Transfer — Provenance & Re-run Capability
**Measured 2026-09-08 · 89 GB total (7 paths) · companion to `DATA_TRANSFER_MANIFEST.md`**

Tier 1 (83 GB) is what the paper *needs*. Tier 2 is what you need to **re-run or defend**
results rather than merely reproduce the tables: competitor checkpoints and data, the
ablation weights behind reported ablation rows, and reference implementations.

Every number Tier 2 supports is **already frozen in the JSONs shipped with Tier 1** — so
nothing here blocks writing or submitting. It matters if a reviewer says *"re-run X"* or
*"show me the ablation"*, and for the artifact release.

Shorthand: `$PROJ = /projects/ammoniacomb/generative_reconstruction`,
`$MAIN = /home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion`.

---

## Tier 2a — recommended (36 GB). Highest value per GB.

| # | Path | Size | What it buys you |
|---|---|---|---|
| 1 | `$MAIN/Save_TrainedModel/JHU/baseline_confild` | 22 G | CoNFiLD's latent-capacity sweep (arms C/F/P, 384→4096-dim). The paper's CoNFiLD row is a *capacity-bound* claim — the decoder ceiling caps reconstruction before conditioning enters. These weights are the evidence for that claim, and it is the single most likely row for a reviewer to challenge, since we report a competitor performing poorly. |
| 2 | `$PROJ/baselines/sparse-reconstruction` | 3.2 G | Contains `data/kolmogorov_shu.npy` — **the raw source of the entire 2D Kolmogorov benchmark** (40 traj × 320 frames × 256²). Re-downloadable from figshare, but that host serves 0-byte 202s from the `figshare.com` domain (must use `ndownloader.figshare.com/files/<id>`); having the file removes the risk. Also the Thuerey 2D reference implementation. |
| 3 | `$PROJ/baselines/sparse-reconstruction-3d` | 51 M | Thuerey 3D code (the published-protocol comparison target). Data was never public; code only. Negligible size. |
| 4 | `$PROJ/baselines/Senseiver_OrchardLANL` | 2.6 M | Upstream Senseiver reference used for the fidelity audit diffs. Negligible size. |
| 5 | `$PROJ/baselines/CoNFiLD` | 7.3 G | Upstream CoNFiLD repo + its pretrained Zenodo archives (`model.zip` 6.9 G, `cond_input.zip` 443 M). Pairs with item 1; re-downloading is ~7 GB from Zenodo. |
| 6 | `$MAIN/Save_TrainedModel/JHU/pointcloud_ffm` **minus** the `iclr_jhu_xcube_spec02_DemoN29_*` run already in Tier 1 | ~13 G | Every DMF-Gen ablation arm: spectral-weight {0, 0.02, 0.05} (N15/N29/N22), augmentation variants (N17/N20/N21), supervision arm (N26–28), regularization seeds, K-ablation (N37/N38), Kolmogorov prior (N34), CQ backbone (N33), and the checkpoint-variance study. These back the appendix ablation tables. |

**Tier 2a rsync** (note the exclude — avoids re-sending 1.2 GB already in Tier 1):
```bash
rsync -avhP --exclude 'iclr_jhu_xcube_spec02_DemoN29_*' \
      $MAIN/Save_TrainedModel/JHU/pointcloud_ffm  user@newhpc:/dest/.../JHU/
rsync -avhP $MAIN/Save_TrainedModel/JHU/baseline_confild \
            $PROJ/baselines/{CoNFiLD,sparse-reconstruction,sparse-reconstruction-3d,Senseiver_OrchardLANL} \
            user@newhpc:/dest/...
```

---

## Tier 2b — optional (44 GB). One path, and it is the whole cost of Tier 2.

| # | Path | Size | Verdict |
|---|---|---|---|
| 7 | `$PROJ/baselines/Gen4Turbulence` | **44 G** | The voxel-diffusion competitor: shipped `best_model.pt` + `models_uxuz/` (our protocol retrain, 501 checkpoints) + `3_flow_reconstruction/data/u.npy` (5.5 G, the 120³ 4-channel cube) + Zenodo artifacts. |

**Recommendation: skip unless you expect to re-run Gen4Turb.** Its row is settled and
adversarially fair already — we retrained it *under our own observation model* and it did
not improve (0.706 → 0.713 rel-L2), which is the strongest form the comparison can take.
The repo is public (`vivekoommen/Gen4Turbulence`, MIT) and its data is Zenodo 17088765
(38.94 GB, CC-BY), so it is fully recoverable on demand. If you do send it, the
**`models_uxuz/` directory is the irreplaceable part** — that retrain cost 24 GPU-hours and
exists nowhere else:
```bash
rsync -avhP $PROJ/baselines/Gen4Turbulence/dm/models_uxuz user@newhpc:/dest/...   # ~8 G
```

---

## Not in any tier — deliberately
`$PROJ/baselines/data/thuerey_figshare_39181919.dat` is a **0-byte failed-download stub**;
`$PROJ/kagglehub` (64 G) is FireBench 2D slices, evaluated and rejected as the wrong
modality. Neither should move.

---

## Verification after Tier 2 lands
```bash
# CoNFiLD sweep arms present (expect latsweep_2048 / 4096 / strict2048 / improve)
ls $MAIN/Save_TrainedModel/JHU/baseline_confild/

# Kolmogorov source npy intact — shape must be (40, 320, 256, 256) float32
python -c "import numpy as np; a=np.load('$PROJ/baselines/sparse-reconstruction/data/kolmogorov_shu.npy', mmap_mode='r'); print(a.shape, a.dtype)"

# DMF-Gen ablation arms present (expect N15/N17/N20/N21/N22/N26-28/N33/N34/N37/N38)
ls $MAIN/Save_TrainedModel/JHU/pointcloud_ffm/ | wc -l

# Gen4Turb protocol retrain, if sent — the wall-clock-matched checkpoint must exist
ls $PROJ/baselines/Gen4Turbulence/dm/models_uxuz/model_4930.pt
```

## Size summary
| Bundle | Size | When to send |
|---|---|---|
| Tier 2a | 36 G | Recommended — provenance for the rows most likely to be challenged |
| Tier 2b (Gen4Turb full) | 44 G | Only if re-running Gen4Turb |
| Tier 2b (`models_uxuz` only) | ~8 G | Good compromise — keeps the irreplaceable retrain |
| **Tier 2 complete** | **89 G** | |
