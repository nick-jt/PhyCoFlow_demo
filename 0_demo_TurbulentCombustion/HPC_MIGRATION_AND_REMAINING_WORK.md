# PoF Benchmark — Migration Handoff & Remaining Work
**Written 2026-09-08 · Target: Physics of Fluids special issue, deadline 2026-10-30 (~7 weeks)**
**Repo state: `origin/main` (branch `worktree-pof2026-benchmark` merged, fast-forward)**

> ## ⚠ TWO STANDING DECISIONS (2026-09-08, Nick)
>
> **1. New compute host: NCSA Delta.** Datasets live at
> **`/work/hdd/bilr/ntricard/datasets/`** on `dtai-login.delta.ncsa.illinois.edu`
> (transferred directly; see §3.2). All config data paths must be repointed there.
>
> **2. Paper scoped to THREE regimes: cylinder wake, Kolmogorov, JHU 3D turbulence.**
> FireBench (multiphysics LES) and SHIFT-WING (unstructured geometry) are **deferred to a
> follow-up paper**, not abandoned — their data, checkpoints, results, figures, audit docs,
> and eval machinery all remain in the repo and in this handoff. This removes the largest
> P0 item (the wing baseline table) and leaves three regimes whose fleets are *fully
> populated*, i.e. a submittable paper with no pending rows.

---

## 1. Where the paper stands

`Paper/pof2026/main.tex` compiles clean (tectonic). **Three-regime benchmark: cylinder →
Kolmogorov → JHU** — a clean dimensional and dynamical ladder (2D laminar → 2D chaotic →
3D chaotic). In scope and complete:

| Component | State |
|---|---|
| JHU 3D fleet (16 rows) + cost table | **done**, canonical n=50 |
| Kolmogorov: DMF-Gen / SiT / latent-FM / Senseiver / MLP-RBF / S3GM / classical | **done** (GeoFNO row pending eval) |
| Cylinder: 6 learned + classical | **done** (latent-FM pending stage-2) |
| Protocol ablation ("worse than literature") | **measured**, one cell pending |
| Reconstruction galleries (3 in-scope datasets) | **done** (2 pending panels) |
| Scaling / Pareto / capability / spectra / uncertainty / sensors figures | **done** |

**Deferred to the follow-up paper** (all artifacts retained, nothing to redo):
FireBench operator matrix (done, frame-matched, audit trail in
`FIREBENCH_FRAME_AUDIT_2026-09-05.md`); SHIFT-WING (data + trained model + galleries exist;
its baseline table was never built — that work moves to the follow-up, where the
surface-pool conditioning caveat in §2 P0-1 still applies).

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
| 1 | Repoint config data paths to Delta | 0.5 d | `/projects/ammoniacomb/generative_reconstruction/…` → `/work/hdd/bilr/ntricard/datasets/…`. Note the reshaped paths: JHU and FireBench are now flat filenames, wing is `shift_wing_processed_v3/`. Then re-run a smoke gate per dataset. |
| 2 | Finish in-flight evals | <1 d compute | Job 18146832 (Senseiver/SiT protocol-B, combined arm); GeoFNO Kolmogorov eval; cylinder latent-FM stage-2 + eval. Commands in §5. **If Delta GPUs are not H100, re-run the full canonical eval set instead** (see §3.4 item 6). |
| 3 | Fill paper `\todo`s from landed JSONs | 1–2 d | Every marker names its data source inline. |
| 4 | Abstract headline numbers + author list/affiliations | 0.5 d | Needs Nick's input on authorship. |

*(The wing baseline table — previously the largest P0 item, 3–5 d — left this list with the
scope decision. Its open caveat is preserved in §6 for the follow-up.)*

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

### 3.2 Data — direct transfer to Delta (~26 GB, sent)

**Destination: `/work/hdd/bilr/ntricard/datasets/` on `dtai-login.delta.ncsa.illinois.edu`.**
Transfer script: `tools/send_datasets_to_delta.sh` (resumable; re-run to top up).

| File / dir at the destination | Size | Regime | In paper scope? |
|---|---|---|---|
| `JHU_4cubes_stride100.h5` | 5.9 G | 3D isotropic turbulence | **yes** |
| `cylinder2d/` (mesh + grid H5 + manifest) | 7.3 G | 2D laminar vortex shedding | **yes** |
| `kolmogorov2d/` (H5 + manifest) | 802 M | 2D chaotic turbulence | **yes** |
| `FireBench_u10u12_merged.h5` | 8.3 G | 3D multiphysics LES | deferred |
| `shift_wing_processed_v3/` | 3.9 G | 3D unstructured geometry | deferred |

Notes: only the *merged* FireBench file is used by any config (the two 4.2 G `_dense` files
are its inputs and were not sent). Paths are flatter than on Kestrel — configs must be
repointed accordingly (P0-1). Not transferred and not needed: the 554 G raw wing tree
(already distilled into `processed_v3`), the 108 G JHU scale campaign, the 54 G 20-cube
file, 64 G of kagglehub 2D slices.

**Checkpoints (86 GB) were NOT part of this transfer.** Models must either be retrained on
Delta (2D fleets 2–4 h each; the 3D fleet is the real cost, ~200 GPU-h for all methods) or
sent separately — see `DATA_TRANSFER_MANIFEST.md` §Tier 1b for the exact frozen-checkpoint
paths, and `DATA_TRANSFER_TIER2.md` for provenance/ablation weights. Retraining is in fact
the *safer* option if Delta's GPUs are not H100 (§3.4 item 6).

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
# 0. repoint configs at the Delta dataset dir (P0-1), then verify one dataset opens
grep -rl '/projects/ammoniacomb/generative_reconstruction' Save_config/ src/ \
  | xargs sed -i 's#/projects/ammoniacomb/generative_reconstruction#/work/hdd/bilr/ntricard/datasets#g'
# then hand-fix the three reshaped paths: JHU + FireBench are now flat filenames
# (…/datasets/JHU_4cubes_stride100.h5), wing is …/datasets/shift_wing_processed_v3/

# 2D data — ONLY if regenerating rather than using the transferred copies
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

## 6. Deferred to the follow-up paper (FireBench + wing)

Nothing here needs redoing; it is parked, not lost. Restarting it is a matter of
re-including the material, not re-running it.

**FireBench (multiphysics LES, realistic measurement operators) — COMPLETE.**
5-operator × 3-method matrix (clean / noise 0.1σ / noise 0.3σ / 25% slab occlusion /
channel dropout), frame-matched after the index audit; robust-vs-clean ablation (N18 vs the
N31 clean control); cross-variable result (fuel density recovered from wind sensors alone at
0.057 rel-L2, std concentrating on the fire front at 7.6–25×). Artifacts: `Save_TrainedModel/
firebench/`, `figures/qual_firebench.*`, `recon_gallery_firebench.*`,
`FIREBENCH_FRAME_AUDIT_2026-09-05.md`, and the removed §4.4 + `tab:ops` recoverable from git
history (pre-scope-cut commits on `worktree-pof2026-benchmark`).

**SHIFT-WING (unstructured mesh, surface-only sensing) — MODEL DONE, BASELINES NOT.**
Our model trains and evaluates (673 cases, 600/73 split; Cp 0.126 rel-L2, velocities
0.53–0.69, corr(std,|err|) 0.77). The baseline table was never built; when it is:
- Conditioning **must** come from the surface pool (`build_sparse_condition_from_pool`), not
  random volume points, or baselines receive information our model never gets.
- Only point-native methods can participate (Senseiver, SiT-point, MLP-RBF, gappy-POD-wing,
  IDW); grid-locked methods get honest "N/A — cannot represent an unstructured mesh" rows,
  which is itself a benchmark result.
- Machinery ready: `dataset_wing_baseline.py`, `config_baseline_SiT_wing.yaml`,
  `baseline_gappy_pod_wing.py`, `evaluate_wing.py --plots`.

**Why this is a good follow-up rather than a loss:** both regimes test the axis the
three-regime paper cannot — measurement realism and geometry — and the capability-matrix
argument ("7 of 10 generative methods are grid-locked") becomes a headline there instead of
a side note here.

---

## 7. Reading order for a new session
1. This file.
2. `HEADLINE_COMPARISON_2026-09-04.md` — 9-axis method comparison + "what 3D exposes that 2D cannot".
3. `HANDOFF.md` — pivot banner, priorities, owed corrections.
4. `FLEET_SUMMARY_TABLE_2026-08-30.md`, `FLEET_AUDIT_2026-08-29.md`, `BASELINE_AUDIT_2026-08-28.md` — the 3D fleet's provenance.
5. `FIREBENCH_FRAME_AUDIT_2026-09-05.md` — worked example of the audit standard this benchmark holds itself to.
6. `Paper/pof2026/main.tex` — every `\todo` names its data source.
