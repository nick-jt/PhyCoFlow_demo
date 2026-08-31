# LOCUS / ICLR 2027 — session handoff

Written 2026-08-28. Repo root for all paths below:
`/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion`

## 1. The job

ICLR 2027 paper. **LOCUS** (placeholder name): generative reconstruction of 3D
fields from sparse point sensors. Thesis: *ambient pointwise conditional
generation removes the 3D discretization bottleneck of generative field
reconstruction, without latent compression.*

Non-negotiable attributes: 3D, ambient (non-latent) space, end-to-end (no
autoencoder), function space, generative. Paper is at `Paper/iclr2027/main.tex`
(9-page limit excluding references; **currently overruns to page 11** — unresolved,
see §7). Tone: professional publication style, not narrative or explanatory.

## 2. Datasets and protocols

| dataset | shape | observed | inferred |
|---|---|---|---|
| JHU isotropic (`isotropic1024coarse`) | 125³, 4 ch (Ux,Uy,Uz,p) | sensors on **Ux, Uz only**, ~1% | Uy, p fully unobserved |
| SHIFT-WING | 673 RANS cases, 4e5 nodes | **surface only** (Cp, wall shear) | volumetric field |
| FireBench LES | 152×126×192, 5 ch, 3.7M pts | sensors + measurement operators | full field |

JHU protocol is **cross-cube**: train on cubes 0–2 (150 snapshots), evaluate on
spatially disjoint cube 3 (50 snapshots). Adopted after showing a random-in-time
split reduces to near-duplicate interpolation. `JHU_SPLIT_MODE=block`,
`JHU_SPLIT_GAP=0`, `JHU_AUGMENT=octahedral`.

New 20-cube corpus (1,200 snapshots) merged at
`/projects/ammoniacomb/.../outputfiles_scale/JHU_20cubes_plus_cube3.h5`,
laid out so frames [0,1200) train and [1200,1250) are the *same* cube 3 block —
validation losses are directly comparable to the 150-snapshot runs.

## 3. Architecture

**Ours** (`src/Model.py`, `ConditionalPointHybridLocalGlobalRBF`, 6.51M params):
conditional rectified flow on field values at scattered points. Gaussian-process
(random-Fourier-feature) source prior. Conditioning is **locality-factored**:
a cross-attention encoder compresses sensors into a global summary (O(SM)), and
each query gathers a top-K importance-warped RBF neighbourhood of sensor tokens
(O(NK), KeOps). Training subsamples ~2% of points per step, decoupling memory
from resolution. Binned spectral loss on the rectified-flow endpoint
x̂₁ = x_t + (1−t)v. Backbone `GL_rbf_ENH`, `gather_topk=16`, λ_sp=0.02, NFE 4.

`GL_rbf_CQ` (vendored upstream, `src/phycoflow_pointcloud/`) is an alternative
query side. **CQ + our valid-key flash attention + K=16 wins every cost axis**
(0.430 s/step train, 10.3× faster NFE-4 inference, 2.7× less inference memory)
but was **not adopted as default** — it ties our λ_sp=0 variant on accuracy.

**Baselines**: latent flow matching (ConvAE3D + latent RF), Senseiver, Gen4Turb
(`oommen2026turbulent`), SiT-point, CoNFiLD. See §6 for which are valid.

## 4. Design decisions that matter

- **Windowed spectra everywhere.** Cutouts are non-periodic sub-blocks; a raw FFT
  creates a broadband leakage floor that drives every method's band ratio toward
  unity. All spectral numbers use `spectral_utils.shell_spectrum(..., window=True)`.
- **Metrics**: fair CRPS, spread/error, central coverage, rank histograms, paired
  per-snapshot CIs, TOST equivalence. Aggregate relL2 is an unweighted mean over
  4 channels — **report per-channel instead**, the aggregate hides the result.
- **Observation clamping** (`clamp_hard`) writes true sensor values into our
  prediction each ODE step. Measured effect on relL2: **+0.0002** (negligible),
  but it makes our sensor-consistency 0 by construction while baselines get
  0.36–0.71. Report unclamped; disclose.
- **Budget runs in optimizer steps, not epochs.** 8× more data must not silently
  multiply cost.

## 5. Results that currently stand

**JHU per-channel relL2** (ensemble mean, unclamped, NFE 4, 50 snapshots):

| | Ux [obs] | Uy [unobs] | Uz [obs] | p [unobs] | agg |
|---|---|---|---|---|---|
| ours N29 | **0.176** | 1.048 | **0.182** | 0.964 | 0.593 |
| latent FM | 0.342 | **0.809** | 0.323 | **0.780** | **0.564** |
| constant predictor | 0.762 | 0.833 | — | — | 0.798 |

Calibration: ours leads decisively (spread/err 0.63 vs 0.53, cov90 0.54 vs 0.45,
t≈9–28 over 50 paired snapshots). CRPS ties or wins.

**FireBench under measurement operators** (relL2) — the strongest result:

| | clean | noise .1σ | noise .3σ | slab occl | drop v,w |
|---|---|---|---|---|---|
| LOCUS N31 | **0.455** | **0.456** | **0.458** | **0.513** | 0.743 |
| LOCUS N18 | 0.456 | 0.458 | 0.461 | 0.511 | **0.603** |
| Latent FM | 0.724 | 0.722 | 0.724 | 0.720 | 0.719 |
| Senseiver | 0.779 | 0.779 | 0.779 | 0.780 | 0.908 |

Our worst operator beats every baseline's clean case. **Decide N18 vs N31** as the
FireBench paper model (N31 wins 4 operators, N18 wins channel dropout).

**SHIFT-WING** (K=4, 8 cases, no paired stats, no generative baseline):
N13 relL2 0.436 / CRPS 0.161; N19 (paper model) 0.473 / 0.177 but better
calibrated. We report the less accurate one — say so in the text.

**Scaling (N36)**: 150 → 1,200 snapshots improved val loss 0.7467 → **0.7256**,
only 2.8%. **This contradicts the appendix claim that snapshot count is the
binding constraint** — rewrite that paragraph.

**Spectral arms**: N34 (k^−5/3 prior) **withdrawn as an improvement** — prior is
3.7×/36× over-energized in inertial/dissipation bands, so its calibration gain
was noise injection. N35 (windowed loss) is a null on standard metrics but gives
**5.2× more inertial energy on the unobserved channel** (0.025 → 0.130).

## 6. Known issues

**The central scientific limitation.** Our model recovers **nothing** of the
unobserved channels: fluctuation correlation with truth is 0.087 (Uy) and 0.019
(p), vs latent FM's 0.640 and 0.664. Three independent measurements agree:
(a) 50× more sensors moves Uy by <0.002; (b) divergence/gradient RMS is 1.554 vs
DNS 0.172 and latent FM's 0.148; (c) near-zero fluctuation correlation.
**Caveat on (b) and (c)**: both come from `Paper/iclr2027/figures/spectra_fields.npz`,
which stores latent FM at NFE 16 and ours at NFE 4 (`spectra_paper.py:104` vs `:63`)
— off the matched protocol. The 0.64-vs-0.09 gap is far too large to be an NFE
artifact, but recompute at matched NFE before the numbers go in the paper.
**Cause**: incompressibility is a *differential* constraint, and our queries are
mutually independent given the conditioning — the network never sees its own
output at neighbouring points, so it cannot form a derivative. Literature
confirms soft divergence penalties *hurt* in turbulence (Oommen Table 2: 46%
worse spectrum); **hard projection is the thing to test**, and no ML paper has
run that ablation on a genuinely unobserved component. That is a claimable gap.

**Withdrawn baselines** (do not report these numbers):
- **SiT** — `sit_conditional_sample_points_chunked` chunked *contiguous* raster
  indices, giving planar token sets (x-extent exactly 0) vs randperm training
  tokens; plus `huber_beta=0.1` ran in its linear/L1 regime instead of SiT's MSE.
  Both fixed; retraining as job 16908439.
- **Senseiver** — Fourier `max_freq` hard-coded 64 instead of derived from the
  grid (adjacent-cell encodings have cosine similarity 0.599 vs −0.024); encoder
  and decoder both drop the original's residual MLP. **A KD-tree beats it 2.6×**
  and its sensor-consistency is 0.72 *on training data*. Not yet fixed. Real
  upstream now vendored at `baselines/Senseiver_OrchardLANL/`.
- **CoNFiLD** — four arms failed because our loop stepped decoder and latents
  every minibatch; upstream steps latents every batch but the **decoder once per
  epoch**. Fixed via `--nf-accum` (50 is best). Faithful 384-d latent saturates
  at bound 0.584 against an oracle ceiling of 0.535; capacity-matched 1024-d
  reaches **0.512**. NOTE: the TTA bound is a NOISY diagnostic — sorting the log
  surfaces 0.353 first, which is a single outlier from a 2-snapshot estimate
  (the adjacent checkpoint reads 0.627). `n_val` has since been raised 2 -> 4
  and per-snapshot values are now printed with their standard deviation.
  Quote 0.512, not 0.353. Note upstream's largest 3D case is 39× smaller in DOF than
  ours and their stage 1 has **no held-out split at all**.

**Operational traps that produced silent successes today** — check runtime
plausibility on every "COMPLETED":

1. `DEMO_NUM` in the launcher vs `Demo_Num:` in the YAML — the **YAML wins**.
2. `--RELOAD` / `shared.reload: true` resumes an existing run dir; if the new
   epoch target is below the old one, training is skipped and exit code is 0.
3. `save_dir` copied from a template points a new run at an old model's dir.
   This overwrote N29's `args.json` (restored; checkpoints were untouched).
4. Eval scripts piping stderr into `grep "[ensemble]"` swallow tracebacks — a
   50/50 failure looked like success. Fixed in `eval_sit_xcube.sh`.
5. `evaluate_Gen_Baseline.py` resolves a relative `--run-dir` against the **repo
   root**, not the working dir. Use `readlink -f`.
6. Slurm has `ConstrainDevices=yes`, but a leaked process from a prior job can
   still occupy your assigned GPU (seen: 69 GiB squatter). Use `--exclude` and
   consider a free-memory check at job start.

## 7. Open, unresolved

- **Paper runs to page 11 vs a 9-page limit.** Three ~1-page trim candidates were
  offered (compress §4.1 uncertainty/small-scale paragraphs; move spectral-leakage
  methodology to appendix; tighten the intro's two detour paragraphs). User has
  not chosen. Removing all `\todo` markers changes nothing (measured).
- No separate test set: cube 3 is both `best.pt` selection and reporting set for
  every model. The 20-cube corpus makes a clean fix possible.
- Single seed per arm; no training-seed replicates anywhere.
- `sample_ensemble` bypasses `model.sample()`, so the ODE cache never activates
  in evaluation (pure speedup available, no number changes).
- Latent FM's `field_names: null` mislabels its output JSONs with combustion
  channel names (CH4/CO/T/U_1 → Ux/Uy/Uz/p positionally).

## 8. In flight (as of handoff)

| job | what it settles |
|---|---|
| 16908439 `sit_xcube` | SiT retrain, MSE + permuted chunking |
| 16969696 `sens_hi` | sensor density 10–60%, crossing the ~40% gradient-resolution threshold where ∇·u becomes usable |
| 16969891 `iclr_jhu_k32` | does top-K RBF K=32 beat K=16 (accuracy *and* uncertainty) |
| 16972401 `jhu_panels` | K=25 midplane + uncertainty figures. **The first attempt (16971506) reported COMPLETED but silently dropped latent FM** — a wrong `build_obs_grid_mask3d` signature — so `paper_jhu_panels_s{0,12}.png` currently contain ONLY our model. Rewritten to consume an `ENSEMBLE_NPZ` dump from the baseline's own visualizer rather than reimplementing its sampler. |
| — | K=64 (16969892) **OOM'd** in `torch.compile`; retry by lowering `gather_query_chunk_size`, which keeps K the only varying factor |

## 8b. Repo state — reorganization is overdue

`src/` currently holds **75 Python files, 108 shell scripts and 306 logs**, and the
structure has drifted badly under experimental pressure. Measured:

- `model_baseline.py` is **7,965 lines** carrying 8 baseline adapters and **13
  visualizer functions**. Every baseline defect found in this session lived here.
- `_save_single_field_plot` is **duplicated** in `helpers.py` and
  `helpers_baseline.py`; both had to be patched separately, twice, for the
  slicing fix and again for percentile colour limits.
- **9 evaluation entry points** (`ensemble_eval.py`, `evaluate_ffm.py`,
  `evaluate_Gen_Baseline.py`, `evaluate_Det_Baseline.py`, `evaluate_wing.py`,
  `evaluate_confild_*.py`, `evaluate_coherence.py`, `evaluate_full_dataset.py`)
  with overlapping and inconsistent metric conventions — the deterministic branch
  computes a flat relL2 over channels while `ensemble_eval` averages per-field,
  a 4% discrepancy on the same prediction.
- ~18 one-off analysis scripts (`compare_spectra*.py`, `diagnose_*.py`,
  `spectra_*.py`, `qualitative_*.py`, `replot_*.py`, `k_ablation.py`,
  `bench_*.py`) that re-derive sampling and metric logic independently.
- **8 of 75 `src/*.py` and 4 of 48 configs are untracked by git.** The repo's
  `.gitignore` has a global `*.yaml` rule, so configs need `git add -f`.

**Target structure** (not yet started — this is a real task, not cleanup):

- `pointcloud_ffm.py` — our model, consolidating `Model.py` and the
  `model_cq.py` adapter behind one backbone switch.
- `train_Gen_Baseline.py` — all generative baselines (latent FM, SiT, Gen4Turb,
  CoNFiLD) through one adapter registry.
- `train_Det_Baseline.py` — deterministic baselines (Senseiver, gappy POD, NN
  interpolation).
- Shared helpers: **one** plotting module (kill the duplicate
  `_save_single_field_plot`), **one** metric module (`ensemble_eval` conventions
  everywhere), one dataset module, one measurement-operator module.
- Move logs out of `src/`; archive one-off scripts under `src/analysis/`.

Do this **before** the next round of experiments. Three of four baselines had
implementation defects, and the duplication above is why each fix had to be
applied in several places and why two of them were missed the first time.

## 9. Next steps, in priority order

**Do first: reorganize `src/` per §8b.** Everything below is slower and more
error-prone until that is done.

1. **Test hard solenoidal projection** in the sampler (not a soft penalty) and
   measure whether Uy fluctuation correlation moves off 0.09. This is the paper's
   biggest open scientific question and an unclaimed gap in the literature.
2. Rewrite the appendix paragraph on snapshot count (N36 refutes it).
3. Fix Senseiver (PE `max_freq` from grid shape; restore residual MLPs), retrain,
   and **gate acceptance on sensor-consistency < 0.1 and monotone improvement with
   sensor count**, not on the loss curve.
4. Finish CoNFiLD stages 2–3 from the capacity-matched 1024-d stage 1.
5. Add a nearest-neighbour interpolation row to the paper — it currently beats
   one of our baselines and a reviewer will ask.
6. Report JHU per-channel throughout; add the constant-predictor floor row.
7. Decide FireBench N18 vs N31; decide wing N13 vs N19.
8. Resolve the page overrun.

## 10. Standing user constraints

- Submit SLURM jobs without asking (see memory `slurm-launch-authorization`).
  Account `f2pde` only — `ammoniacomb` is overdrawn.
- JHTDB queries strictly sequential; network-bound downloads may run on the login
  node, compute must go through SLURM.
- Baselines must be a fair representation — no cheating. Check every adaptation
  against the upstream source before trusting a number.
- **Visual inspection outranks aggregate statistics.** Every failure found in this
  project was visible in a figure and invisible in the metrics.
