# PoF Benchmark — Migration Handoff & Remaining Work
**Written 2026-09-08 · Target: Physics of Fluids special issue, deadline 2026-10-30 (~7 weeks)**
**Repo state: `origin/main` @ `cdd6689` (branch `worktree-pof2026-benchmark` merged, fast-forward)**

---

## 1. Where the paper stands

`Paper/pof2026/main.tex` compiles clean (tectonic, ~16 pp two-column). Five-regime benchmark:
cylinder → Kolmogorov → JHU → FireBench → wing. Complete and written:

| Component | State |
|---|---|
| JHU 3D fleet (16 rows) + cost table | **done**, canonical n=50 |
| FireBench operator matrix | **done**, frame-matched re-eval (audit trail in repo) |
| Kolmogorov: DMF-Gen / SiT / latent-FM / Senseiver / MLP-RBF / S3GM / classical | **done** (GeoFNO row pending eval) |
| Cylinder: 6 learned + classical | **done** (latent-FM pending stage-2) |
| Protocol ablation ("worse than literature" section) | **measured**, one cell pending |
| Reconstruction galleries (4 datasets) | **done** (2 pending panels) |
| Scaling / Pareto / capability / spectra / uncertainty / sensors figures | **done** |
| Wing baseline table | **NOT STARTED** — the largest gap |

**Headline findings already banked** (all measured, all in the draft):
1. Method ranking is regime-dependent — gappy POD 0.056 wins the cylinder outright; GeoFNO 0.385 leads 2D chaos; latent-FM/SiT/IDW split the 3D columns. No method wins twice.
2. **Protocol, not architecture**, explains the gap to literature numbers: uniform-vs-scattered sensor layout alone is 25% rel-L2 (0.487→0.367); leaky splits pay only memorizing models (Senseiver 0.17 train / 0.91 val), hence *reorder* rankings.
3. Unobserved-channel identifiability wall in 3D (fleet-wide ≥0.9) vs full recovery in 2D cylinder (POD 0.051) — the benchmark's clearest open problem.
4. Ensemble size corrects *coverage estimates*, not accuracy (K 8→64: cov90 0.59→0.73, rel-L2 −2.5%, CRPS flat).
5. Latent compression truncates the dissipation band ~5×; ambient methods starve the inertial band.

---

## 2. Remaining work, prioritized

### P0 — blocks submission
| # | Item | Effort | Notes |
|---|---|---|---|
| 1 | **Wing baseline table** | 3–5 d | Machinery exists (`dataset_wing_baseline.py`, `config_baseline_SiT_wing.yaml`, `baseline_gappy_pod_wing.py`). **Open item: conditioning must come from the surface pool** (`build_sparse_condition_from_pool`), not random volume points — otherwise baselines get information our model never receives. Point-native subset only + honest "N/A (grid-locked)" rows. |
| 2 | Finish in-flight evals | <1 d compute | Job 18146832 (Senseiver/SiT protocol-B, combined arm); GeoFNO Kolmogorov eval; cylinder latent-FM stage-2 + eval. Commands in §5. |
| 3 | Fill paper `\todo`s from landed JSONs | 1–2 d | ~10 markers; every one has its data source named inline. |
| 4 | Abstract headline numbers + author list/affiliations | 0.5 d | Needs Nick's input on authorship. |

### P1 — reviewer-critical
| # | Item | Effort | Notes |
|---|---|---|---|
| 5 | **Post-hoc recalibration TEST numbers** (P3 from the old handoff) | 1–2 d | `recalibrate_spread.py`, `conformal_recalib.py` written and smoke-tested; fit on TUNE (cube-3 odd), freeze, report on TEST (even). Must also be offered to latent-FM for fairness. Upgrades calibration story to "characterized *and* repaired". |
| 6 | Training-seed replicates | 2–3 d compute | Seed variance is the one un-quantified error bar. 2 replicates × the 2D fleet is cheap now that 2D trains in 2–4 h. |
| 7 | S3GM cylinder audit (rel-L2 1.019, worse than train-mean) | 0.5 d | Known sampler fragility. Either fix guidance params or report as a documented divergence with its figures — do **not** silently drop the row. |
| 8 | Leakage mini-table (3 methods, shuffled vs honest) | 0.5 d | Numbers exist in archived pre-fix evaluations; needs assembly. |

### P2 — strengthens, not required
9. SiT patch-4 arm on 2D (quantifies how much of its error is patch granularity — its blockiness is visible in the galleries). ~4 h compute.
10. Cylinder protocol variants never run: **surface-tap-only** sensing and **observe-u-only** cross-channel test (both configured for, neither executed). These are the cylinder's analogue of the wing/3D identifiability probes.
11. 2D spectra panels (Kolmogorov energy spectra by method) — the 3D spectral story has no 2D counterpart yet.
12. Verify `picsb2026taxonomy` citation against arXiv:2603.22319 (placeholder in `refs_extra.bib`).
13. Artifact packaging + reproduction dry-run from the artifact alone (Zenodo/HF). **Check JHTDB redistribution terms** — fallback is coords + regeneration script.

---

## 3. Migration checklist

### 3.1 Code — trivial
```bash
git clone git@github.com:nick-jt/PhyCoFlow_demo.git       # everything is in origin/main @ cdd6689
```
All configs, launchers, converters, eval drivers, figure scripts, and audit docs are tracked.
Note `.gitignore` has a global `*.yaml` rule — **new configs need `git add -f`**.

### 3.2 Data — 250 GB total, but most is regenerable
| Path | Size | Migration strategy |
|---|---|---|
| `jhu_homogeneous_turbulence/JHU_4cubes_stride100.h5` | 6.3 G | **COPY — highest priority.** Re-downloading from JHTDB took ~5 days under politeness limits. The other 173 G in that dir (20-cube file, per-cube files) is optional. |
| `firebench3d/` | 19 G | **COPY** (regenerable from GCS zarr but the crop recipe is fiddly — see `extract_firebench.py`). |
| `shift_wing/processed_v3/` | 3.9 G | **COPY** processed only; the 554 G of raw HF downloads is *not* worth moving (re-download if ever needed). |
| `cylinder2d/` | 7.3 G | **Regenerate** — `openfoam/cylinder2d/submit_all.sh`, ~15 min/case × 6 on any CPU partition, then `convert_cylinder.sh`. Faster than copying. |
| `kolmogorov2d/` | 802 M | **Regenerate** from `kolmogorov_shu.npy` (3.2 G, re-downloadable from the Thuerey figshare) via `convert_kolmogorov.sh`. |
| `baselines/` repos | ~50 G | Re-clone; only Gen4Turb checkpoints (`3_flow_reconstruction/`) are worth copying. |
| `Save_TrainedModel/` (main + worktree) | 86 G | **COPY the frozen paper checkpoints**: JHU `iclr_jhu_xcube_spec02_DemoN29_*`, all `JHU/baseline_*` best.pt, `firebench/*`, `wing/*`, and both 2D fleets. Retraining is ~200 GPU-h. Prune `_legacy/` and `_failed_attempts/` first. |

### 3.3 Environment
```bash
python -m venv ~/venvs/jhtdb_env && source ~/venvs/jhtdb_env/bin/activate
pip install torch h5py numpy scipy matplotlib pyyaml tqdm einops ema_pytorch \
            accelerate torchinfo torchprofile pykeops gcsfs zarr
# ~/envs/jhtdb is a 4-line activation shim: venv + JHTDB token + `module load cuda/12.9`
```
Also needed: **KeOps** (compiles a cache on first run — do a warm-up run before submitting a fleet),
`~/bin/tectonic` for the paper, OpenFOAM 9 only if regenerating the cylinder.

### 3.4 Cluster-portability gotchas (hard-won — re-check each on the new HPC)
1. **Partition names/QOS**: launchers hardcode `--partition=gpu-h100 --account=f2pde` (and `shared`/`short` for CPU). One `sed` across `src/*.sh` + `openfoam/*/submit_all.sh`.
2. **Standby starvation**: on Kestrel, standby QOS left jobs pending for *days*; the workaround was rotating jobs through `debug-gpu-stdby` (2-job cap, 4 h, idle nodes). If the new HPC has real allocation, this whole machinery is unnecessary — delete the rotation scripts.
3. **OpenFOAM on CPU nodes**: the OF9 build links `libnvf.so` from nvhpc, whose module only exists on GPU nodes. `submit_all.sh` hardwires `LD_LIBRARY_PATH` to the shared-filesystem path. **Run `openfoam/cylinder2d/env_test.sh` first** — 8-stage preflight, seconds, catches exactly this class of failure.
4. **`$SLURM_SUBMIT_DIR` not `dirname $0`**: SLURM copies batch scripts to a spool dir; relative paths break.
5. **Login-node policy**: network-bound work (downloads) fine; all compute via SLURM.
6. **Evals must run on a compute node**: `require_compute_node` guards exist because CUDA sensor draws are SKU-dependent — sensor layouts are only bit-identical on matching GPU SKUs. **If the new HPC has different GPUs, re-run the canonical evals for every method rather than mixing old and new numbers.**

---

## 4. Protocol invariants — do not break these

These are the benchmark's contract; violating any invalidates cross-method comparability:
- `JHU_SPLIT_MODE=block JHU_SPLIT_GAP=<n>` with **`round()`** boundary arithmetic (an `int()` off-by-one leaked one frame at train_ratio 0.8 — fixed in both `helpers.py` and `helpers_baseline.py`).
- Sensors drawn via **canonical `helpers.build_sparse_condition`** under `torch.manual_seed(seed*777+snap)` for *every* method (`helpers_baseline`'s CUDA-randint variant is NOT equivalent — ~2% index overlap).
- Ensemble noise: `base = seed*131 + si`, then `manual_seed(base*10_000 + k)`.
- Deterministic methods: two-member tiling so **CRPS ≡ MAE exactly**; dispersion fields `null`, never 0.
- Report **per-channel**, observed vs unobserved flagged separately.
- Spectra: **windowed (Hann) only** — non-periodic cutouts leak a broadband floor that invalidated an entire earlier analysis.
- Single samples for spectra/galleries, never ensemble means.
- Cost fields (train s/step, GPU-h, peak GB, infer s/field, peak GB) required on every row.
- Model selection on TUNE (odd indices), reporting on TEST (even) — never the same set.

---

## 5. Restart commands

```bash
# 2D data (regenerate)
sbatch src/convert_kolmogorov.sh
bash   openfoam/cylinder2d/env_test.sh && sbatch openfoam/cylinder2d/submit_all.sh
sbatch src/convert_cylinder.sh && python src/check_strouhal.py   # St must match 0.14-0.21

# smoke gates BEFORE any fleet (they use `set -euo pipefail` and really do gate)
sbatch src/smoke_kolm2d.sh ; sbatch src/smoke_cyl2d.sh

# fleets
sbatch src/train_bench_kolm_ffm.sh
CONFIG=Save_config/kolmogorov2d/config_baseline_Det_kolm.yaml TRAINER=train_Det_Baseline.py \
  sbatch src/train_kolm_baseline.sh          # repeat per config; LFM_STAGE=1 then 2
sbatch src/train_bench_cyl_ffm.sh ; (same pattern via src/train_cyl_baseline.sh)

# evals (dataset-generic driver)
DATASET=kolmogorov2d MODELS="dmfgen sit latent_fm senseiver mlp_rbf geofno s3gm" sbatch src/eval_kolm_fleet.sh
DATASET=cylinder2d   MODELS="...same..."                                        sbatch src/eval_kolm_fleet.sh
ARMS="b_senseiver b_sit uniform_b_dmfgen" sbatch src/eval_kolm_litprotocol.sh   # protocol ablation
sbatch src/eval_kolm_ksweep.sh                                                   # K sensitivity

# figures + paper
python src/figs_pof/<fig>.py     # each is standalone, CPU-only, re-runnable as data lands
cd Paper/pof2026 && ~/bin/tectonic main.tex
```

---

## 6. Reading order for a new session
1. This file.
2. `HEADLINE_COMPARISON_2026-09-04.md` — 9-axis method comparison + "what 3D exposes that 2D cannot".
3. `HANDOFF.md` — pivot banner, priorities, owed corrections.
4. `FLEET_SUMMARY_TABLE_2026-08-30.md`, `FLEET_AUDIT_2026-08-29.md`, `BASELINE_AUDIT_2026-08-28.md` — the 3D fleet's provenance.
5. `FIREBENCH_FRAME_AUDIT_2026-09-05.md` — worked example of the audit standard this benchmark holds itself to.
6. `Paper/pof2026/main.tex` — every `\todo` names its data source.
