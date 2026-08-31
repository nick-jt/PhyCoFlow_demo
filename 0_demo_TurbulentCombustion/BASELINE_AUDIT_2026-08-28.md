# Baseline fidelity audit + JHU rebuild — 2026-08-28

Six agents, one per baseline. Each audited its adapter against genuine upstream,
classified every difference, matched capacity, instrumented cost, and relaunched.

## 1. Paper corrections forced by the audit

| # | Location | Problem |
|---|---|---|
| 1 | `main.tex:123`, `:244` | **Importance-warped retrieval does not exist.** Top-K selection is plain Euclidean (`Model.py:1218-1224`, `:1246-1250`); `_get_topk_neighbors` never receives the bias. Beta enters only the softmax logits *after* selection (`:1409-1411`), scaled by a learned constant init 1e-2. The wing claim that warped retrieval enables surface-to-volume inference is unsupported — that result was obtained with Euclidean retrieval. |
| 2 | `main.tex:59` (intro) | **"Gen4Turb trains on 32^3 crops" is false.** Upstream `Par.pkl` records nx=256, ny=256, nz=32 = 2,097,152 voxels. Our 120^3 = 1,728,000 is 18% *smaller*. Replace with our measured memory wall. |
| 2b | `tab:jhu` caption | Caption says "1.95M points" but the Gen4Turb row is scored on the 120^3 crop = **1,728,000** points. |
| 3 | `main.tex:404` | Gen4Turb "57.5 GB at 120^3" is the **128^3** measurement. Correct 120^3 value = **47.35 GB**. |
| 4 | `app:baselines` | "wall-clock-matched epoch 4,930" is **23.61 h**, not 20 h. Labelling error only — see §3. |
| 5 | `tab:jhu` caption | Gen4Turb shared-mask **0.354/0.204 is wrong** -- validation-contaminated (checkpoint selected on frames 150-167; 5 of 8 eval snapshots inside) and computed on an 8-snapshot subset. Honest value, budget-matched checkpoint, all 50 snapshots, canonical seeding: **0.470 / 0.256**. Optimistic by ~0.12 relL2 and ~0.05 CRPS. The qualification must stay: shared-mask observes all four channels at identical locations, ~2x the information any other method gets, so it must never share a column with strict-protocol numbers. |
| 5b | `main.tex:176` | "severely underdispersed (coverage 0.25)" is true under the **strict** protocol but is a consequence of **conditioning starvation, not the sampler**: the same model under shared-mask reaches spread-error **0.644** and cov90 **0.484**, versus 0.284/0.245 under strict. State the cause. |
| 6 | `app:training` | Sensor counts are drawn **uniformly** (`torch.randint`), not log-uniformly. Same in `helpers.py:536` and `helpers_baseline.py:1457`, so no unfairness — but the text is wrong. Uniform over [1953, 19531] centres mass near 0.55% coverage. |
| 7 | `app:baselines` (latent FM) | **"with a point-encoder conditioning branch" is false.** Runs used `cond_mode: "image"`; `model_baseline.py:2480` explicitly forbids the PointNet path in 3D. |
| 8 | `app:baselines` | Latent FM parameter count: claimed 6.53M, measured **6,679,268**. |
| 9 | `main.tex:49`, `:59` | **"Autoencoder bounds what can be recovered" must be narrowed to the dissipation range.** Measured AE ceiling on held-out cube = 0.0296 aggregate relL2, 19.4x below the pipeline's 0.574. AE retains ~1.0 of inertial energy but only **0.17-0.24** of dissipation energy. Claim is strong for dissipation, false for accuracy and for the inertial range. |
| 10 | `tab:jhu` (Senseiver row) | 0.722 came from a model that was **+28.7% over parameter budget AND structurally broken** (halved PE spectrum, no feed-forward after either cross-attention, untied encoder weights, non-upstream optimizer). Invalid. |
| 11 | `main.tex:140`, `tab:ops` caption | **"Identical seeded sensor draws" is false.** See §2. Invalidates all paired CIs and TOST tests as currently computed. |
| 12 | project-wide | `helpers.py:245-251` makes **cube 3 the validation split for every method**, so `best.pt` is selected on the reporting set. Handoff §7 confirmed concrete. |

## 2. The sensor-pairing defect (most consequential)

Two failure modes, both now fixed for the rebuild:

1. **Two `build_sparse_condition` implementations diverge.** `helpers.py:536` draws the count with
   CPU `torch.randint`; `helpers_baseline.py:1457` uses `device=device` (CUDA), advancing CUDA RNG
   state before `torch.randperm`. Measured index overlap at seed 0: **787/39062 = 2.0%** — exactly
   chance. Our model and every baseline saw statistically independent layouts.
2. **`torch.randperm` on CUDA is not portable across GPU SKU.** Login nodes (H100 PCIe) give
   `idx_sum=38044045759`; compute nodes (H100 80GB HBM3 SXM) give `37987162596`. No PCIe node
   exists in `gpu-h100`, so this is strictly a login-vs-compute effect.

**Protocol now enforced fleet-wide:**
- Import `build_sparse_condition` from **`helpers.py`** only.
- Run every evaluation through SLURM on a **compute node**. Never interactively.
- Canonical eval: `--seed 0 --op-seed 1000 --n-snapshots 50 --K 8 --cond-fields 0 2 --n-obs 19531 19531`
  (evaluation count is FIXED at 1%; log-uniform 0.1-1% is the *training* distribution only).
- Verification fingerprint: `snap=29 sensors=39062 idx_sum=37987162596`.
  Independently confirmed by Senseiver, CoNFiLD, and latent FM on compute nodes.

## 3. Per-baseline outcome

**Senseiver** — upstream `e443eb0`. Restored: per-axis `max_freq` from grid shape (was scalar 64,
halving the spectrum), residual feed-forwards after both cross-attentions, encoder weight tying,
width-preserving MLP, bare-Linear readout, plain Adam constant LR. Gradient clip at 1.0 measured
**binding on 100% of steps** (min norm 1.27) = normalized-gradient Adam; **removed** to match
upstream. New config 64x320 = 6,419,396 params (-1.3%), bottleneck 20,480 scalars (deliberately
below ours; equalizing is free in parameters and therefore meaningless as a control).
Old checkpoint **failed the acceptance gate** (SenConsis 0.71-0.99; nearest-sensor floor beats it).

**Latent FM** — the strongest baseline, and **we crippled it**. `model_baseline.py:2543-2544`
average-pools a zero-filled sparse grid over 8^3=512 voxels, so one sensor of value v presents as
v/512: measured conditioning std **0.0014-0.0085** against field std 0.841, while the co-input z_t
has std 1-3.6. Also: AE latents fed raw to an N(0,I) flow (std 3.62, 8x per-channel anisotropy;
LDM uses `scale_factor` for exactly this), and a pure L1+MSE autoencoder with no KL/VQ/LPIPS/adversarial
— **which biases our own spectral-floor result in our favour** and must be disclosed.
Canonical-layout score is **0.574**, not the published 0.564; lead over our 0.593 narrows to 0.019.

**Gen4Turb** — upstream `8af0a7b`; model code byte-identical, deviation surface is 4 places.
`dim=16` gives 6,157,076 params (-5.4%), already inside the band. Measured wall (H100 80GB, batch 6):
120^3 = 47.4 GB, 128^3 = 57.4 GB, 160^3 = OOM; at batch 1, 256^3 = 75.5 GB.
Score **0.71 +/- 0.068** (true checkpoint sigma, complete 11-checkpoint window, after removing
checkpoint x snapshot interaction; the estimate converged 0.2 -> 0.100 -> 0.076 -> 0.068 as
checkpoints accumulated and has stabilised).
20 h window mean 0.747 vs 23.6 h 0.700 = 0.047 = **0.65 sigma, not significant** — the budget mislabel is a
labelling error, not a scoring error. Crop bias b ~ 0.010, retention 0.884270, **pairable** with a footnote.
**Both windows complete — the noise never settles, and the published number is a lucky draw.**
20 h window (11 ckpt) = **0.760 +/- 0.068** all-50 equivalent; 23.4 h window (8 ckpt, complete) =
**0.765 +/- 0.085**. Difference -0.0045, t = -0.11: the later window is nominally *worse*, so the
model is definitively **not** still improving and no fairness allowance is owed. Window B sigma
(0.085) is *larger* than window A's (0.068) -- noise persists to the end of training.
Consequence: the published **0.7133 is epoch 4,930, sitting -0.61 sigma below its own window mean**;
the expected value at that budget is **~0.765**. The current table therefore slightly *flatters*
Gen4Turb. RECOMMENDATION: report **0.713 with sigma = 0.09 attached and labelled a single
checkpoint**, noting the window expectation of 0.765 -- i.e. keep the number least favourable to us
while disclosing the full picture, so the correction cannot read as cherry-picking against a baseline.
NOTE for whoever collects the annealing arm: a single post-annealing checkpoint is **not
interpretable** against 0.76 +/- 0.09; it needs the same window scan.

**Checkpoint selection is now justified on principle, not generosity.** Validation-selected
all-50 canonical = **0.9351** vs budget-matched **0.7133** — a 0.22 gap, **3.2x the checkpoint
sigma of 0.068**, so decisively real rather than sampling noise. Mechanism: upstream selects on
*spectral error*, and its log records `best model: 3161` = only **15.1 h**, an under-trained
checkpoint whose spectral error is low but whose reconstruction accuracy is much worse. Compounding
it, upstream's validation frames 150-167 sit **inside evaluation cube 3**. So the validation-selected
number is compromised twice: wrong criterion and contaminated split. Reporting the budget-matched
checkpoint is therefore the only defensible choice -- rewrite the appendix sentence that currently
calls it "the choice most generous to the baseline".

**Upstream never calls `scheduler.step()`** — the LR is constant 1e-4 for the entire run and the
scheduler exists only to be read by a log line. An annealing arm runs as a *labelled deviation*.

**SiT-point** — upstream `cbde832`. Both known bugs genuinely fixed in the trained weights
(randperm sampling `:6457`; `huber_beta=0.0` giving upstream MSE). No fidelity bugs remain.
Old run was +14.8% over budget -> superseded; matched run at 6,571,924 (+1.0%).
**Assembles a field from 239 chunks sharing the sensor set but with independent source noise**, and
self-attention couples tokens within each chunk — so pointwise metrics are valid but spatial
coherence is destroyed above the chunk density scale. **Exclude from the spectra figure.**
Class-(iii), unchanged: `qk_norm=True`, the AdamW/warmup/cosine package, grad clip + spike-skip,
32-step Euler vs upstream 250-step dopri5.

**CoNFiLD** — upstream `449835e`, unchanged since our port; vendored copy byte-identical.
Found and eliminated a **protocol violation**: `--nf-accum` had been selected using the TTA bound
**on cube 3**. Latent LR restored to upstream 1e-5. Three arms: C (1024-d code, 1.4M prior),
F (384-d, upstream verbatim), **P (published 108M prior, wall-clock matched)** — the +/-10% rule was
waived for P because it forces a **75x reduction** in exactly the component under evaluation, while
the codec eats 70-75% of the budget. Benchmarked: published prior costs 44.5 vs 25.3 ms/step,
i.e. ~2 extra hours to recover 75x of generative capacity. Quote **0.512**, never 0.353.

**Classical anchors** — built from scratch; none existed.
Constant predictor = **exactly 1.000** on every channel (train mean is 0 in z-score units) — this
replaces the unsourced 0.762/0.833/0.798. KD-tree 0.648 / IDW(k=8) **0.620** aggregate.
**Both beat the published Senseiver 0.722**, confirming the handoff's claim directionally
(not by 2.6x). Gappy POD (rank 80, chosen on train cube 2) = 0.955.
**Our model scores 1.050 on Uy; a constant scores 1.000.** Re-running on compute nodes.

## 4. Standing recommendation

Report **per-channel throughout**; demote the aggregate. The aggregate averages two observed
channels where we lead against two unobserved ones where everything sits near 1.0, which makes us
look only marginally better than an inverse-distance interpolator (0.593 vs 0.620). Per channel we
are ~25% better than IDW on Ux (0.177 vs 0.235) and honestly worse than a constant on Uy. The second
story is both more accurate and more defensible.

## 5. Decisions made without the user, all reversible

1. Removed Senseiver's gradient clip (measured binding on 100% of steps; upstream has none).
2. Added CoNFiLD arm P outside the +/-10% parameter rule, alongside the two strict arms.
3. Made `helpers.py` the canonical sensor-draw path fleet-wide.

## 6. Working-copy safety

No worktree was used: the default branches from `origin/main` and would have discarded the 369
uncommitted files that *are* the code under audit. A tar snapshot of `src/*.py`, `src/*.sh` and all
configs was taken before any agent ran, in `$CLAUDE_JOB_DIR/tmp/`.
`helpers.py`, `helpers_baseline.py`, `ensemble_eval.py`, `train_Gen_Baseline.py` — **untouched**.
`model_baseline.py`, `train_Det_Baseline.py` — modified only in the Senseiver region.
Note `.gitignore` has a global `*.yaml` rule with per-file whitelists; new configs need a `!` line
or `git add -f` at commit time.

## 7. OPEN CRITICAL QUESTION — is the headline comparison inside checkpoint noise?

Gen4Turb's measured **true checkpoint sigma is 0.068-0.085** (many checkpoints in a tight window,
snapshots/sensor draws/per-sample seeds all fixed; sigma computed as Var(row means) - Var(e)/S, not
the naive std of row means -- the naive figure runs high, and a 4-snapshot design inflated the first
estimate to 0.2 against a true 0.068).

The paper's headline is **ours 0.593 vs latent FM 0.574 = a 0.019 gap**, and CRPS -0.014 with
CI [-0.026, -0.002]. **If our model or latent FM carries checkpoint noise of even a third of
Gen4Turb's, both claims are uninterpretable from single checkpoints** -- independently of the
sensor-pairing fix.

Measurement commissioned: window of >=8-11 checkpoints per model near the budget, >=12 snapshots,
everything else held fixed, reporting naive and corrected sigma for both models, and an explicit
verdict on whether the 0.019 and 0.014 gaps survive. If checkpoints were not saved densely enough
near the budget, that is itself a finding: the comparison could not then be defended from existing
artifacts and would need a rerun with periodic checkpointing.

Note this is DISTINCT from the known "single training seed per arm, no seed replicates" gap
(handoff s7). That concerns variation across training runs; this concerns variation across
checkpoints *within* one run, and is cheaper to measure.

### 7a. Infrastructure gap found while answering it

**No trainer archives checkpoints.** `train_pointcloud_ffm.py:1241-1244` and
`train_Gen_Baseline.py:244-247` both save only `best.pt` plus an **overwritten** `last.pt`.
`save_every` controls visualization/NFE benchmarking, NOT checkpoint archiving
(`train_pointcloud_ffm.py:1246`); `Evaluation/epoch_XXXX/` holds figures and JSON, no weights.
Across 30+ `pointcloud_ffm` runs the maximum is 5 `.pt` files, and that run is the random-split
one with ad-hoc manual backups. **So no checkpoint-noise window can be formed from existing
artifacts for any model in the paper.** Add periodic archiving before the next training round.

### 7b. Preliminary signal — probably fine

Latent FM's two checkpoints, **1,410 epochs = 14.1% of budget apart**, under fully fixed conditions
on 50 snapshots and canonical layouts, differ by **0.0002** in aggregate relL2 (0.57427 vs 0.57446)
and 0.0011 in CRPS. That is ~100x smaller than the 0.019 gap under test.
This is N=2 at the wrong spacing, NOT a sigma -- but it is consistent with theory: Gen4Turb samples
with 32 **stochastic** denoising steps at `S_churn=80`, whereas our model and latent FM integrate a
near-straight rectified-flow ODE with a **deterministic** Euler solver at NFE 4 and average K=8.
Gen4Turb's 0.068-0.085 sigma should therefore NOT be expected to transfer to the flow models.
Job 16996667 runs the same 2-point check on our DemoN15/DemoN29 to decide cheaply whether a 20 h
rerun with periodic checkpointing is warranted. The rerun is deliberately NOT launched until it reports.

Latent FM window run: job 16996542, 11 checkpoints at epochs 10500-11000 step 50 (-4.5%..0% of budget).
Estimator validated synthetically (true 0.000/0.020/0.080 recovered from naive 0.007/0.026/0.084).

### 7c. NFE sweep, latent FM, canonical layouts, 50 snapshots

relL2 is flat in NFE (0.5755 -> 0.5711 across NFE 2->32, a spread of 0.004) while CRPS improves
monotonically by 8% (0.3072 -> 0.2824). The two metrics measure different things; report both.

## 8. Gen4Turb — FINAL, all five configs, all 50 snapshots, canonical seeding

| checkpoint | protocol | relL2 mean | sample | CRPS | spread/err | cov90 |
|---|---|---|---|---|---|---|
| ep 4,170 (20.0 h) | strict | 0.8636 | 0.8879 | 0.5094 | 0.311 | 0.270 |
| **ep 4,930 (23.6 h)** | **strict** | **0.7133** | 0.7350 | 0.4418 | 0.284 | 0.245 |
| best (val-sel, 15.1 h) | strict | 0.9351 | 0.9570 | 0.6058 | 0.235 | 0.144 |
| **ep 4,930** | **shared** | **0.4697** | 0.5381 | **0.2563** | 0.644 | 0.484 |
| best (val-sel) | shared | 0.3924 | 0.4562 | 0.2050 | 0.657 | 0.523 |

Windows (all-50 equiv): 20 h = **0.760 +/- 0.068**, 23.4 h = **0.765 +/- 0.085**, difference t = -0.11.

**The paper's shared-mask 0.354 decomposes exactly:**
0.3540 (paper) -> 0.3924 (same ckpt, all 50) = **+0.038 favourable 8-snapshot subset**
0.3924 -> 0.4697 (budget-matched, all 50) = **+0.077 cube-3 contamination**
**Total overstatement +0.116 relL2, +0.052 CRPS**, ~1/3 subset and ~2/3 contamination.

**Recommended table entries:** strict **0.76 +/- 0.09** (or 0.713 explicitly marked one draw,
sigma 0.085); shared-mask **0.470 / 0.256** in a separate column labelled ~2x information;
validation-selected 0.9351 if shown at all, noting upstream selects on *spectral error* at 15.1 h.

## 9. Periodic figures + loss plots (user requirement, 2026-08-28)

Standard to match is `pointcloud_ffm`: `train_pointcloud_ffm.py:1247` runs
`run_reconstruction_benchmark` every `save_every` epochs under EMA weights; `:1267` calls
`logger.plot_history()`. Baselines going through `train_Gen_Baseline.py` get the equivalent from
`:256-257` (`_tracker.log` + `_tracker.plot` **every epoch** -> `loss_history.{csv,json,png}`) and
`:272` (`adapter.visualize` inside `evaluation_weights` at each `save_every`).

**Target: ~20-40 reconstruction figures per run (`save_every ~ epochs/30`), same held-out snapshot
each time, truth / prediction / |error| per channel + ensemble std where available.**

Audit of current cadence:

| baseline | epochs / save_every | figures | verdict |
|---|---|---|---|
| **SiT** | 6,000 / 3,000 | **2** | **BROKEN — restart with save_every 200** |
| latent FM | 10,000 / 500 or 200 | 20-50 | ok |
| Senseiver | ~11,000 / 500 | ~22 | ok |
| CoNFiLD | custom trainer | **ZERO — none at all, both stages** | **FIXED, 3 arms restarted (~5% cost)** |
| Gen4Turb | external trainer | **none** | must be added to the annealing arm |
| S3GM | via train_Gen_Baseline | inherits | set save_every deliberately, do not inherit |
| FNO | via train_pointcloud_ffm | inherits | free, if it runs in 3-D at all |
| classical | no training | n/a | still needs reconstruction figures |

Rationale: the user's standing rule is that visual inspection outranks aggregate statistics -- every
failure found in this project was visible in a figure and invisible in the metrics. Concretely:
Senseiver's pre-audit checkpoint failed its acceptance gate (SenConsis 0.71-0.99, beaten by a
Voronoi floor) and CoNFiLD's four earlier arms died of latent-independent collapse; both are obvious
in a reconstruction image and subtle in a loss curve.

## 10. CORRECTION — the latent-FM attenuation does NOT blind the model

I initially framed the 100-600x conditioning attenuation as leaving the baseline "effectively blind
to its own sensors". **That is an overstatement and must not go in the paper.** Rendered figures
(same snapshot, same sensors, NFE 4) show the shipped model recovering large-scale topology
correctly at epoch 8,590 (L2 0.435) -- the network learned to amplify the attenuated conditioning,
which a `Conv3d` with unbounded weights and 10,000 epochs can do.

**Defensible claim:** the attenuation is a real defect that costs **small-scale fidelity and
training efficiency**, not one that blinds the model. Evidence: the fixed model at epoch **155**
(1.4% of training) already reproduces the same large-scale topology and carries visibly *more*
fine-scale texture than the fully-trained shipped model, with error concentrated on filaments
rather than through the bulk. Not yet a win claim -- 0.572 @ ep155 vs 0.435 @ ep8590 is not a
comparison; the 11,000-epoch run settles it.

**This is the clearest vindication of the "visual inspection outranks aggregate statistics" rule in
the whole exercise**: within an hour of figures being requested, a figure corrected a claim that had
already been written down on the strength of a measured attenuation factor alone.

Figure infrastructure confirmed working for latent FM (924 PNGs in DemoN23; truth / reconstruction /
|error| panels with sensor positions as green markers, per channel per NFE) and for Senseiver
(3-panel column per channel, sensor overlay, separate perceptual colormap for |error|). Senseiver's
epoch-20 smoke figure independently demonstrates the point: reconstruction is flat blue noise while
truth shows turbulent structure -- unmistakable in the image, invisible in the loss value.

Decision: **no ensemble-std panel in the training-time visualizer** (K draws per figure multiplies
monitoring cost for marginal value); spread belongs in the final evaluation figures, which use K=8.

## 11. MAJOR — our own predictive spread has a floor and does not track information

Triggered by the latent-FM agent noticing that its spread on *observed* channels is a near-uniform
speckle that does not collapse near sensors. Testing the same question against our own sensor sweep
(N29, K=8, all snapshots, observed channels Ux/Uz averaged):

| sensors % | spread | rmse | spread/err | cov90 |
|---|---|---|---|---|
| 0.10 | 0.1551 | 0.3403 | 0.46 | 0.419 |
| 0.25 | 0.1175 | 0.2518 | 0.47 | 0.477 |
| 0.50 | 0.1087 | 0.2113 | 0.51 | 0.525 |
| **1.00** | **0.1052** | **0.1803** | **0.58** | **0.575** |
| 2.00 | 0.1040 | 0.1535 | 0.68 | 0.633 |
| 5.00 | 0.1029 | 0.1214 | 0.85 | 0.716 |
| 10.00 | 0.1028 | 0.1064 | 0.97 | 0.769 |
| 20.00 | 0.1021 | 0.0908 | **1.12** | 0.817 |

**Across a 200x increase in sensors, RMSE falls 73% while spread falls only 34% and saturates at a
floor of ~0.102.** Consequently spread/error sweeps **0.46 -> 1.12**, crossing 1.0 near 10% sensors:
the model is **underdispersed when data is sparse and OVERdispersed when data is dense**.

**What this means.** The predictive spread is close to a constant set by the model/prior rather than
a quantity tracking the information content of the observations. The good calibration reported at
the 1% operating point (spread/err 0.58-0.63) is a *point on this sweep*, not evidence that the
model knows what it does not know. Candidate mechanism worth checking: the RFF-GP source has fixed
correlation length 0.15 and fixed amplitude, and the flow may not fully contract it by NFE 4.

**Consequences for the paper.**
1. "Best-calibrated of any method evaluated" **survives** -- it is a relative claim and the baselines
   are worse.
2. The *mechanism* narrative at `main.tex:190` must be rewritten. It currently explains the flat
   spread-vs-sensor-distance profile as "governed by which channels are observed and by local flow
   structure ... consistent with a conditioning design in which every query also reads a global
   summary." That presents a **spread floor** as a design consequence. The flat profile and the
   flat density response are the same phenomenon and should be reported as one.
3. **This can be turned into a contribution rather than a concession.** No prior work reports the
   sensor density at which a generative reconstructor's predictive distribution crosses from under-
   to over-dispersed. The sweep above is a novel, honest, and directly useful figure.

Caveats before publishing: K=8 makes each spread estimate coarse (the paper notes K=32 raises
coverage); the trend is monotone and clean but should be reconfirmed at K=32 at two or three
densities. Test the prior-floor hypothesis by sweeping the GP correlation length/amplitude, or by
measuring spread at NFE 16-64 where the flow has more opportunity to contract.

### 9a. CoNFiLD had no figures at all — worse than the cadence risk

`confult_upstream_training.py` bypasses `train_Gen_Baseline.py:256-272` entirely: no `savefig`,
no `matplotlib`, no tracker. Both stages ran blind on `history.jsonl` alone. (`save_every: 5000`
governs *checkpoints*, not figures, and was harmless -- there were simply no figures to schedule.)

Now emits, on **progress-fraction** scheduling rather than a fixed interval (arms run 35-58 s/epoch
and every stage is wall-clock-truncated, so a constant N gives a different count per arm):
~30 figures in stage 1 and ~24 in stage 2 for **every** arm.

- Stage 1 (2.0 s/event): a *trained* item decoded from its own learned latent -- the direct picture
  of latent-independent collapse; plus the **same held-out snapshot every time** via a frozen-decoder
  latent fit, which is the codec bound and the panel comparable **across the three arms at matched
  epochs**. New diagnostics `latent_dependence` and `heldout_codec_rel_l2` in the loss plot.
- Stage 2 (9.7 s/event): codec sanity; an **unconditional prior sample** decoded to a field -- if the
  prior has not learned the latent manifold this is visibly not turbulence, which no diffusion loss
  reveals; and sampled-vs-real latent marginals.

Correctness trap handled: stage 1 accumulates decoder gradients across a whole epoch and steps at the
epoch boundary, so a naive TTA backward would silently corrupt the pending decoder update.

**All three arms restarted** (16997995 / 16997998 / 16998001), ~1 h lost each of 19.5 h (~5%):
the live processes had already imported the old module, so only stage 2 -- 13.5 h later -- would have
picked up figures, leaving stage 1 blind in exactly the stage where the collapse failure mode lives.
Old dirs renamed `ABORTED_nofigures_*` so stage 2 cannot select a stale stage-1 checkpoint.
CoNFiLD results therefore land ~6 h later than the rest of the fleet.

## 12. CoNFiLD normalization bug after augmentation (user-reported, CONFIRMED + measured)

**The claim.** `FieldStatistics.minimum/maximum` are **per-point** `[point, channel]` arrays reduced
over time only (`confild_upstream_core.py:157-166`). `octahedral_gather` returns at row k a value
gathered from source index `flat[k]` but representing the transformed field at **target**
`point_ids[k]`. The training loop then normalizes with the **target** ids:

```
values = octahedral_gather(physical, dataset.grid_shape, group_index, ids)   # :465
batch_truth.append(stats_device.normalize(values.to(device), ids_device))    # :471  <- TARGET ids
```
and `normalize` indexes `self.minimum[point_indices]`. For **47 of 48** group elements this uses
another location's statistics. **CONFIRMED.** Same defect at `:139` in `_diagnostic`, so
`train_rel_l2` -- the metric used to detect latent collapse -- is itself computed in distorted units.

**Worse than reported:** `octahedral_gather` also permutes velocity components and applies **sign
flips**. A correct fix must gather stats at the source index, apply the same channel permutation,
AND handle sign flips, under which min/max **swap and negate** (`new_min = -old_max`,
`new_max = -old_min`). Missing that inverts the range on flipped channels.

**Measured magnitude** (strided 19,532 points over the full cube, 150 train snapshots).
Per-point range CV: Ux **9.7%**, Uy **10.2%**, Uz **12.9%**, p **19.1%** (p99/p1 up to 2.38).
Ratio of two independently drawn per-point ranges -- i.e. the scale error the bug applies:

| ch | median | 5-95% band | mean abs log ratio |
|---|---|---|---|
| Ux | 1.000 | [0.799, 1.247] | **10.8%** |
| Uy | 0.999 | [0.788, 1.272] | **11.7%** |
| Uz | 0.999 | [0.743, 1.344] | **14.5%** |
| p  | 1.000 | [0.645, 1.555] | **21.2%** |

**"Modest" is the wrong word.** Unbiased but random multiplicative rescaling of the regression
target, 11-15% on velocity and 21% on pressure, and the factor **changes with the group index** --
so the same physical structure is presented at a different scale depending on which transform was
drawn, directly opposing what the augmentation is for. Pressure is worst because it is strongly
intermittent (deep minima in vortex cores), making its per-point sample extrema most variable.

**Fix adopted: spatially-uniform (global per-channel) min/max -- but NOT for the reason I first gave.**

I argued the spread was *pure estimation noise* because HIT is homogeneous. **That is measurably
false.** Split-half within a single cube (even vs odd snapshots, with a positive control on the
per-point time mean at r = 0.80-0.96) shows the per-point range reproduces at **r = 0.42-0.59** --
it carries real, reproducible spatial structure, roughly half the spatial variance. Statistical
homogeneity is an *ensemble* property; a finite record of one realisation with long-lived
large-scale structure is not spatially homogeneous. (An earlier split-half across snapshots 0-74 vs
75-149 gave incoherent results because those halves straddle cube boundaries -- different flow
realisations, not exchangeable samples. The positive control caught it: time-mean correlation of
-0.36/-0.49 is impossible for a real quantity.)

**The decisive test is transfer across cubes, and there the structure vanishes:**

| pairing | Ux | Uy | Uz | p |
|---|---|---|---|---|
| cube0 vs cube1 | 0.043 | -0.236 | 0.264 | -0.116 |
| cube0 vs cube2 | -0.027 | -0.185 | -0.021 | 0.360 |
| cube1 vs cube2 | 0.014 | 0.067 | 0.035 | -0.116 |
| cube0 vs **held-out 3** | 0.002 | -0.256 | 0.270 | -0.031 |
| cube1 vs **held-out 3** | -0.131 | 0.083 | 0.386 | -0.031 |
| cube2 vs **held-out 3** | 0.010 | 0.032 | -0.102 | -0.104 |

Within a realisation r ~ 0.5; **across realisations r ~ 0**, scattered either side of zero in all six
pairings. So per-point statistics fitted on cubes 0-2 encode structure that is real but
**realisation-specific**, carrying essentially no information about cube 3, and inject a spatially
structured 11-21% multiplicative error into held-out reconstructions that is uncorrelated with
anything real there. Upstream's scheme is right for *their* wall-bounded cases, where per-point
structure is a property of the geometry and therefore does transfer across held-out times; here
there is no geometry to pin it.

**Broader implication worth carrying:** any per-point statistic fitted on cubes 0-2 fails to transfer
to cube 3. This is independent evidence that the cross-cube protocol is genuinely hard, and it should
be checked wherever else the project fits per-point quantities on the training cubes. Upstream's pointwise scheme is right for *their*
inhomogeneous cases (walls, geometry) and degenerates to noise on HIT, besides being incompatible
with the mandated octahedral augmentation. Declare as class-(ii) with that justification.
Arms to be restarted; they were only ~20 min in when this was found.

Also adopted, belt-and-braces: **normalise-then-augment**, using the identity
`normalize(g.f ; g.stats) == g.normalize(f ; stats)` -- normalise once up front, then apply the group
action to *normalised* values. Exact (verified for all 48 elements including the sign-flip min/max
swap) and needs no statistics transform at all.

**Attribution correction:** the bug predates the `octahedral_gather` optimisation. The original path
did `octahedral_transform(physical, ...)` then `normalize(physical[ids], ids)` -- identical
semantics. The gather is bit-exact with it, so it faithfully reproduced the defect. The audit gap was
verifying the optimisation *against the existing implementation* rather than against the semantics.
The legacy `confild_baseline.py` never had this bug: `field_minmax` (`:84`) does `amin(dim=0)` over
*points*, giving global per-channel stats, and normalises before augmenting.

Jobs: 16999582 (C), 16999584 (F), 16999585 (P), all gated `--dependency=afterok:16999526` on a
normalisation+figure smoke, so they launch only if it passes. Superseded dirs renamed
`ABORTED_normbug_*` / `ABORTED_nofigures_*`. Total discarded compute ~1.5 h against 19.5 h per arm.

### 12a. Second-order trap: relative L2 is not comparable across normalisation schemes

Switching CoNFiLD to global extremes changed the scale of the normalised field, and that silently
corrupted a diagnostic. Measured:

| | Ux | Uy | Uz | p |
|---|---|---|---|---|
| global range / mean per-point range | 1.73 | 1.89 | 1.89 | 2.47 |
| std of normalised field (global) | 0.161 | 0.179 | 0.126 | 0.117 |
| std of normalised field (per-point) | 0.284 | 0.300 | 0.332 | 0.323 |
| 1st-99th pct (global) | -0.39..0.32 | -0.68..0.12 | -0.30..0.26 | **-0.02..0.56** |

The field now sits in a narrow, **DC-offset** band of [-1,1] -- pressure never straddles zero. Harmless
for the SIREN (linear output layer, Adam is scale-invariant) and for reported metrics (evaluation
denormalises exactly, so the scale cancels). But `rel_l2 = ||pred-truth|| / ||truth||` computed in
[-1,1] units has its **denominator inflated by the offset**, making the codec bound look better than
it is: the epoch-0 held-out bound read **0.726 under global stats vs 0.955 under per-point** -- the two
were never comparable.

Fixed by adding `_rel_l2_zscore`, which denormalises to physical and re-scores in the **benchmark
z-scored units every baseline is measured in** (matching `ensemble_eval`). Unit-tested against a
construction with a known 10% error: recovered 0.100093. Both are logged; the loss plot and
checkpoint selection use the z-scored one. Unchanged: checkpoint selection (the [-1,1] denominator is
constant, so the argmin was already identical) and the final reported metrics (from
`ensemble_metrics`).

**Generalisable rule: never compare a relative-L2 quoted in one normalisation scheme against one
quoted in another.** Report bounds in the benchmark z-scored units used by `ensemble_eval`, or state
the scheme explicitly. Happily this also restores comparability with history: the legacy
`confild_baseline.py` used global per-channel stats all along, so the previously quoted **0.512 /
0.535** bounds are in the same family as the new runs -- the unified trainer's per-point statistics
were the outlier, not the legacy numbers.

Launch chain: `16999526 (smoke) -> 16999958 (smoke on edited code) -> arms 16999582 / 16999584 /
16999585`, wired with `scontrol update` so the arms keep queue position and start only on `afterok`.

## 13. FNO — the existing backbone was 2-D AND dead code; a faithful 3-D one now runs

**Phase 0 verdict: `Model.FNO` is strictly 2-D and could never have run on JHU.** Two independent
hard blockers: `Model.py:1727` passes a 2-tuple `n_modes` so upstream sets `order=2` and calls
`rfft2`; `_get_grid_permutation` does `coords[0,:,:2]`, dropping z; and
`validate_regular_grid_compatibility` (`helpers.py:35`) **rejects our data outright** -- JHU has 125
unique values on each of three axes, so `Num_x*Num_y <= 15,625` and every candidate fails, with
`train_pointcloud_ffm.py:1086-1092` raising `SystemExit(1)` before step 1. Independently,
`FNOFFM.training_loss` (`Model.py:2282`) does not accept the four spectral-loss kwargs the trainer
passes at `:653`, so the branch would `TypeError` on the first step **even on 2-D data**. It is dead
code and has never run.

**A faithful 3-D extension is feasible and was built** (`src/fno3d_backbone.py`, `--backbone fno3d`).
42-config memory sweep at full 125^3: **nothing OOM'd**; peak memory is linear in width and
essentially independent of modes (spectral weights are tiny beside `width x 1.95M` activations).
width 128 / modes 16^3 = 151M params trains at 30.8 GB.

**UPSTREAM BUG FOUND (neuralop 2.0.0).** `SpectralConv` applies `fftshift` on **both** the forward and
inverse paths. Self-inverse on even axes; **not** on odd axes. Measured on a full-band identity
convolution: max relative error **2.0 at N=125** versus 2.5e-07 at N=124. Fixed upstream at HEAD
(`00b7d86`) with `ifftshift`. Restored locally via `SpectralConvOddSafe` (3.5e-07 at N=125); the
shared venv was not touched. **Implication: any published neuralop-2.0.0 FNO result on an odd-sized
grid with >=2 spatial dims is wrong.** Worth a footnote and worth reporting upstream.

Capacity: **6,729,135 by `numel` (+3.43%)**, but real DOF **13,447,599** -- spectral weights are
complex, so `numel` undercounts 2x. Matched on the standard convention, both reported.
Bottleneck = **mode truncation: 2,304 of 984,375 retained rfftn coefficients = 0.234% of the
half-spectrum**, resolving |k| <= 8 against Nyquist 62 = 12.9% of the wavenumber range.

**Headline cost result: inference 0.097 s for a full 1.95M-point field at NFE 4, ~60x faster than our
5.73 s.** Consistent with the paper's own concession that grid computation wins below the ~240^3
crossover; what the FNO pays is a fixed discretization, a 0.23% spectral bottleneck, and the
forfeited point-cloud interface.

Class (iii), flagged and NOT changed (both plausibly handicap the FNO, both inherited from the 2-D
class): time conditioning is a spatially-constant broadcast channel so it reaches the spectral
operator only at k=0 (upstream offers `norm="ada_in"`; diffusion-FNO practice is FiLM); and
`domain_padding=None` although a 125^3 cutout of the 1024^3 box is **not periodic**, so the FNO
imposes wrap-around across faces.

Fingerprint verified exactly. Jobs: 16997834 (memory wall), 16998935 (fingerprint), 16999367
(training smoke, passed, ~31 figures), 16999861 (eval smoke), **16999778 (main run, queued)**.
Instrumentation: train 0.7623 s/step (B=8, full grid, excl. val), 48.44 GB peak; inference 0.0967 s
and 6.29 GB per field.

## 14. PAPER CORRECTION — our own stated training budget is wrong

`app:training` claims our model trains for "6,000 epochs (~20 h wall-clock; the latent baseline's two
stages total ~21 h, so total budgets are matched)". Measured for N29, the paper model (job 16567553):

- **SLURM elapsed: 16:35:49 (16.6 h)** including evaluation and visualization overhead
- **Pure training: 13.22 h** (sum of 6,000 logged epoch times, median 7.30 s/epoch)

So **~20 h overstates our own budget by ~20-50%** depending which figure is meant, and the
"budgets are matched" sentence against the latent baseline's ~21 h is not supported.

Consequence for this audit: every baseline has been matched at ~19.5 h, i.e. **more compute than our
model actually received**. That errs in the baselines' favour, which is the safe direction -- but the
paper must state the real numbers. Cleanest fix: report our actual budget and say plainly that every
baseline was given at least our wall-clock.

## 15. S3GM — ten class-(i) fidelity bugs; the port was a strawman

Upstream `github.com/lzy12301/S3GM` @ `2343293` (Li et al., Nat. Mach. Intell. 6:1566-1579).
Model code is a **verbatim port** (diff shows only `th`->`torch`, docstrings, reflow). The damage was
all in configuration and the sampling/训练 wrapper:

| # | ours | upstream | class |
|---|---|---|---|
| 1 | `sigma_max: 7.0` | 20 (`train.py:46`) | (i) restored |
| 2 | **frame axis collapsed (T=1, whole field)**, window machinery discarded, `frame_indices=zeros` | `arange(T)`, windows (`datasets.py:82-98`) | **(i) restored, T=10 z-slabs** |
| 3 | `clip_grad_norm_(0.5)` + skip if >50 | no clipping | (i) removed |
| 4 | 100-epoch warmup + cosine LR | no scheduler | (i) removed |
| 5 | lr 1e-4 / wd 1e-6 | **2e-4 / 0** | (i) restored |
| 6 | **`alpha_obs: 1.0` fixed** | `alpha = alpha_case/sqrt(1-sparsity)` -> **5.0 at 1% observed** | **(i) guidance was 5x too weak** |
| 7 | 5 Langevin corrector steps | `NoneCorrector` | (i) -> 0 |
| 8 | `clamp(step_size, max=1.0)` in corrector | absent | (i) (moot, corrector off) |
| 9 | step size from continuous schedule while `sde.N=1000` | `VESDE(N=outer_loop)`, discrete sigma ladder matches step count | (i) SDE rebuilt |
| 10 | window-overlap consistency loss (beta=0.4) **absent** | `utils.py:709` | (i) restored, 14 windows |

Matching exactly: the training objective, the mask conventions, and the DPS rule itself.

**On the sqrt(M) guidance question:** upstream's residual is an **unnormalised sum of squares**, so the
per-sensor gradient is independent of M -- the 1/sqrt(M) scaling belongs to the zeta/||r|| DPS variant
upstream does not use. Upstream instead *deliberately* reintroduces an M dependence through
`alpha = alpha_case/sqrt(observed fraction)`. At our fixed 19,531/channel this is a constant
alpha = 5.0, so it is resolved; the evaluator recomputes alpha from the actual observed fraction so a
density sweep stays correct, **but any sweep must report alpha alongside M**.

**Capacity:** upstream default is **18,856,612 (2.9x ABOVE target)** -- this scales *down*, not up. The
shipped 2-D config would have been **300,812,932 (46x)** and was never run. Chosen nf=32,
ch_mult (1,2,3,4) -> **6,348,228 (-2.43%)**, moving only the multipliers; width, depth, attention
placement and head count are upstream's. No compressive bottleneck (UNet skips bypass it).
Memory wall measured: uncheckpointed batch 32 peaks **74,183 MB**, 48 OOMs; with upstream's
`use_checkpoint`, 32 -> 26,224 MB at +24% step time. Inference: one draw 32,035 MB, four OOM.

**Class (iii), flagged NOT changed:** (1) **z mapped onto the frame axis** -- 2-D convs in (x,y), z
coupled only by relative-position attention at the coarsest level, so **the model is anisotropic in a
physically isotropic flow** (octahedral augmentation mitigates but does not prove isotropy);
(2) **our single-snapshot protocol removes the time correlation S3GM exists to exploit**;
(3) 200 vs upstream's 1000 denoising steps (cost: 3.85 h vs 19.2 h evaluation); (4) `alpha_case=0.5`.

**Framing that must go in the paper:** a weak S3GM result should be read as *"S3GM's mechanism does not
transfer to single-snapshot 3-D"*, not *"S3GM is weak"*. Its RPE and temporal attention exist for a
time axis this protocol does not provide.

Fingerprint verified exactly on a compute node (and `helpers_baseline`'s variant reproduced the
38144789850 / 2.00% trap). Instrumentation: train **0.634-0.656 s/step** excl. val, **26,332 MB**;
inference **34.6 s** per full 125^3 field at N=200, **13,266 MB** (corrected -- the earlier 32,035 MB was probed with `use_checkpoint=False` while the shipped config sets `true`; the same error inflated the inference memory wall, so "2 draws 63.9 GB, 4 OOM" must NOT be quoted). Figures: truth/prediction/|error|
per channel with sensors overlaid **plus an ensemble-std panel**, ~35 across the run.
Jobs: 17000065 (smoke 2) -> **17000129** (training, `afterok`, 1400 ep ~ 19.4 h) -> 17000204 (archiver).

### 15a. S3GM OPEN RISK — the guided sampler may not converge

At 2 epochs and N=25 the smoke gave relL2 ~ **4e6 on the OBSERVED channels** (Ux, Uz) but only
~80/49 on the unobserved ones -- the blow-up comes **entirely from the DPS term**, which is why it
appears only where guidance acts. Intrinsic to upstream's rule: the gradient is subtracted with **no
step size**, and it is worst at coarse N. Expected to subside as the score improves (an untrained
score gives a garbage x_hat_0 and therefore an enormous gradient), but unproven.

**No step size was invented** -- that would be an unjustified deviation. Instead a `[sampler]`
diagnostic prints `obs_rmse_z`, `max_abs`, `dx_max`, `clamped_elems` and a `DIVERGED` flag at every
monitor figure (first at epoch 40, ~35 min in). Job 17000581 measures N = 25/100/200 sensitivity to
bound it. An explicit abort criterion (threshold + epoch) has been requested so this is caught early
rather than after 19.4 h.

**If it does not converge at upstream-faithful settings, that IS the result** and belongs in the
paper as *"S3GM's guided sampler does not converge on single-snapshot 3-D reconstruction at its
published settings"*, with the N-sensitivity as evidence -- not quietly stabilised.

**Correction to my own brief, recorded:** I told this agent the unnormalised residual makes guidance
scale as sqrt(M). It does not -- the per-sensor gradient is independent of M; the 1/sqrt(M) behaviour
belongs to the zeta/||r|| DPS variant upstream does not use. Upstream reintroduces M-dependence
deliberately via `alpha = alpha_case/sqrt(observed fraction)`. The same claim was made to the CoNFiLD
agent and should be re-checked there against its own guidance rule rather than assumed.

### 15b. RESOLVED — the two DPS baselines use different residual forms

I claimed to both agents that an unnormalised DPS residual makes guidance scale as sqrt(M). Checked
against both upstreams; the claim is **half right**, and the difference is real:

| baseline | upstream residual | per-sensor gradient | M-dependence |
|---|---|---|---|
| **S3GM** | `sum(residual**2)` (`utils.py:706-728`) | `2*alpha*(x0_j - y_j) * dx0_j/dx` | **independent of M** |
| **CoNFiLD** | `torch.linalg.norm(difference)` (`condition_methods.py:31`) | `d||r||/dr_j = r_j/||r||`, and `||r|| ~ sqrt(M)*sigma_r` | **scales as 1/sqrt(M)** |

So my sqrt(M) claim was **correct for CoNFiLD and wrong for S3GM**. S3GM instead reintroduces
M-dependence *deliberately* through `alpha = alpha_case/sqrt(observed fraction)`.

**Both are resolved by our fixed 19,531-sensor evaluation, but for different reasons** -- CoNFiLD
because constant M gives constant guidance strength, S3GM because constant observed fraction gives
constant alpha.

**Consequence for anyone running a sensor-density sweep on either:** they behave *oppositely*.
CoNFiLD's guidance automatically weakens as 1/sqrt(M) with more sensors, while S3GM's alpha rule
*strengthens* it as 1/sqrt(fraction). Any density sweep must report the effective guidance strength
alongside M for both, or the two curves are not comparable to each other or to themselves.

### 15c. S3GM divergence is NOT a step-count artifact — measured, gated, enforced

N-sensitivity on the 2-epoch smoke checkpoint (alpha=5.0, 4 draws/setting):

| N | median `obs_rmse_z` | `dx_max` | `clamped_elems` | agg relL2 |
|---|---|---|---|---|
| 25 | 6.25e7 | 1.000e8 (saturated) | 6.19e5 | 2.60e6 |
| 100 | 6.14e7 | 1.000e8 (saturated) | 3.56e6 | 2.56e6 |
| 200 | 6.00e7 | 1.000e8 (saturated) | 7.47e6 | -- |

**8x more denoising steps buys 4.4%.** `dx_max` sits exactly on upstream's +/-1e8 clamp at every
setting and `clamped_elems` scales *linearly* with N (~3.1e3 elements/step) -- **every step saturates
the clamp**. This is a hard runaway whose fixed point is set by the clamp, not a discretisation
artifact. **Step count is not the lever; score quality is.** (The agent's own earlier hypothesis that
this was an N=25 artifact is refuted.)

**Reusable calibration idea.** Fields are per-field z-scored, so `obs_rmse_z` -- RMSE between the
returned sample and its own sensor values, at the sensors -- has a principled reference:
**1.41 (= sqrt 2) means the sample is statistically independent of its observations**, i.e. guidance
doing literally nothing. Below that, guidance helps; far above, guidance is actively pushing the
sample away from its own data. Worth using as a sanity check on any guided sampler in this project.

**Committed abort criterion, enforced by a watchdog job rather than by anyone watching:**
> At **epoch 200** (~2.8 h, 14% of budget), if median `obs_rmse_z` > **10**, abort.
> Re-checked at **epoch 600** (~43%). Immediate abort on any non-finite value at epoch >= 40.
> **1.41 < obs_rmse_z <= 10 is "poor but converging": the run continues and the honest number is
> reported.**

Threshold 10 is ~7x the do-nothing baseline, and the measured value is 6e6 *above* it, so it cannot
fire ambiguously in either direction. `s3gm_watchdog.py` (job **17000895**, `after:17000129`) polls
every 120 s, and on a fired gate calls `scancel` and writes `s3gm_verdict_*.json` with the full
trajectory, alpha and clamp statistics. Both branches unit-tested against synthetic logs.

If it aborts, **that is the result**: "S3GM's guided sampler does not converge on single-snapshot 3-D
reconstruction at its published settings", evidenced by the verdict JSON plus the table above. Any
stabilised variant must be a clearly-labelled second row, never a substitute for the faithful one.

## 16. SYSTEMIC — shared-filesystem I/O contention invalidates the wall-clock budget control

`TurbulentCombustionH5Dataset.__getitem__` (`helpers.py`) reads
`h5["fields"][0, t_idx, :, 0, 0, :]` on **every** access with **zero caching**. Verified by grepping
the class for `cache`, `preload`, `in_memory`, `lru`, `self.fields =` -- **all counts are 0**. The file
is **5.9 GB**; one snapshot is 31.2 MB.

- one train epoch = 150 x 31.2 MB = **4.7 GB**
- a validation pass = **1.6 GB**, every 5 epochs
- at ~6 s/epoch that is **~780 MB/s sustained** off shared Lustre, per job

With ~10 sibling jobs on the same filesystem, the Senseiver run measured **13.5 s/epoch wall against
~6 s of compute -- a 45% duty cycle** -- and **stalled outright for 13 minutes** with no traceback
(GPU work healthy throughout).

**Why this matters more than a slow run: it breaks the control variable.** The protocol fixes
wall-clock at ~19.5 h. If half of that is I/O wait whose magnitude depends on how many siblings
happen to be running, then "matched budget" is not controlled -- it measures queue occupancy. And it
is *unequally* distributed, since the fleet drains at different times. Our own N29 reference shows
the same signature: **16.6 h SLURM elapsed against 13.2 h of summed epoch time.**

**Fix, launcher-only, no shared-code change:** stage the H5 to node-local NVMe at job start
(`/tmp/$USER/jhu_$SLURM_JOB_ID/`, ~3.2 TB available), rewrite `data_path` into a job-local config, and
**fall back to the shared path if the copy or a size check fails** -- so it can only help. ~1-2 min
once, removes the coupling for the whole run.

**Also mandated fleet-wide:** log **optimizer steps alongside wall-clock**, and report the **duty
cycle** (summed compute time vs SLURM elapsed). If contention survives staging, step counts keep the
comparison auditable in a way wall-clock alone cannot, and the paper can state honestly what fraction
of each budget was real compute.

**Second-order consequences flagged to the owning agents:**
- **FNO is worst exposed** -- `requires_full_grid` means it reads the entire snapshot per item where
  every other method subsamples ~39k queries. Its reported **0.7623 s train step** may be contaminated
  and must be re-measured on local disk before it goes in the cost table against our 0.44 s.
- **CoNFiLD's three arms start together and read the same file simultaneously**, so they contend with
  each other; an I/O-driven difference between them would masquerade as a bottleneck/capacity effect.
- **S3GM's abort gate** is specified at epoch 200 = "~2.8 h, 14% of budget". That mapping assumed a
  ~100% duty cycle; at 45% the gate costs twice its budgeted compute and the epoch-600 re-check may
  never be reached.
- **Checkpoint-archive windows** specified as epoch ranges near the end of training (SiT
  `archive_from: 5700`; latent FM 10,500-11,000) are **never reached** if contention prevents the run
  from finishing its epoch count -- yielding zero archives and, for latent FM, losing the
  checkpoint-noise measurement entirely.

### 16a. The staging fix creates a NEW asymmetry unless the budget unit changes

Flagged by the Senseiver agent and confirmed: **our reference model N29 was never staged either.**
Its launcher `train_iclr_jhu_xcube_spec02.sh` contains no node-local copy. Measured:

| | N29 (ours, reference) | baselines as configured |
|---|---|---|
| optimizer steps | **48,000** (8 steps/epoch x 6,000) | varies by method |
| summed compute | **13.22 h** | ~18 h at 19.5 h wall, if staged |
| SLURM elapsed | 16.60 h | 19.5 h budget |
| duty cycle | **80%** (unstaged, quiet cluster) | 45% unstaged / ~100% staged |

So giving baselines node-local staging while the reference ran unstaged **hands them more effective
compute per wall-clock hour** -- a new asymmetry, in their favour this time, but still a broken
control. N29's 80% duty cycle is much better than the 45% measured today simply because it ran on a
quieter cluster (22 Aug, before this fleet existed), which is itself the point: **wall-clock is
contaminated by whatever else was running at the time.**

**RESOLUTION -- change the budget unit to compute-hours (summed step time), and report all three.**

Wall-clock is contaminated by contention; optimizer steps ignore that different architectures do
different work per step. **Summed compute time is I/O-independent AND accounts for per-step cost**,
so it is the fairest single control. Every run now logs `total_optimizer_steps`,
`duty_cycle_compute_over_wall`, and summed compute, so all three can be reported and audited.

Reference budget: **13.22 compute-hours / 48,000 optimizer steps**. Baselines running at 19.5 h wall
will exceed that, which is the safe direction -- state plainly in the paper that **every baseline
received at least our compute budget**, and give the table of all three numbers per method.

If strict symmetry is wanted instead, re-running N29 with staging costs ~19.5 h and is available;
I did not launch it speculatively.

### 16b. CORRECTION — I over-generalised. Duty cycle varies 45%-99.9% BY METHOD, and that is the real problem

I broadcast "the wall-clock budget is invalid fleet-wide" from a single stalled job. Three
measurements later that is too strong, and the accurate statement is more useful:

| method | duty cycle | mechanism |
|---|---|---|
| **Senseiver** | **45%** (13.5 s/ep wall vs ~6 s compute), 13-min stall | `TurbulentCombustionH5Dataset.__getitem__` re-reads 31.2 MB per access, no cache |
| **Latent FM** | **86.6%**, degradation as the fleet filled only **+0.6%** | 6.27 GB file fits page cache under `--mem=96G`; effectively node-local after epoch 1 |
| **CoNFiLD** | **99.91%** (5113.8 s compute / 5118.5 s elapsed) *under full fleet contention* | `cache_snapshots: true` streams all 150 snapshots to RAM once at startup; the hot loop touches HDF5 **zero** times |

All jobs request 96-128 GB, so memory allocation does **not** explain the difference -- implementation
does. Latent FM also correctly noted that its `epoch_time_s` is measured around the whole batch loop
and therefore *already includes* dataloader stalls, so its 13% gap is validation, checkpointing and
plotting -- not input I/O. The robust diagnostic is **degradation as the fleet filled**, which for
latent FM is +0.6%.

**The finding that survives, and it is cleaner:** wall-clock is not a valid budget control **not
because everything is I/O-bound, but because methods differ in I/O exposure by more than 2x.** The
same 19.5 h buys 45% of a card for one method and 99.9% for another. That is exactly what a control
variable must not do.

**So the resolution in 16a stands unchanged: budget in compute-hours, report all three numbers.**
Staging remains worth doing everywhere -- harmless, with fallback, and good fleet citizenship since it
removes ~4.7 GB/epoch of shared-Lustre reads that hurt I/O-bound siblings -- but it is a fix for
*some* methods, not a fleet-wide emergency.

**Remaining exposure to watch:** FNO (`requires_full_grid`, so it reads an entire snapshot per item --
the largest I/O volume in the fleet) and every **evaluator**, including CoNFiLD's, which use
`TurbulentCombustionH5Dataset` and re-read ~4.7 GB per run even when the trainer does not.

**Process note for myself:** this is the third generalisation today walked back by a second
measurement (the others: Gen4Turb "+-0.2 checkpoint noise" from 4 snapshots, and latent FM
"blind to its own sensors" from an attenuation factor). The pattern is inferring a mechanism from one
measurement and stating it as general. Measure the second case before broadcasting.

### 16c. Our own 0.44 s/step does not match what the run experienced -- and my 80% duty cycle was wrong

Prompted by the FNO agent asking, correctly, whether our quoted step time is measured the same
data-resident way as its own. Checked:

| quantity | value |
|---|---|
| paper's quoted step time (`main.tex`, cost paragraph) | **0.44 s/step** |
| N29 measured: 7.30 s/epoch median / 8 steps | **0.912 s/step WALL** |
| N29 `compile_model` | **True** -- so the compiled 0.41-0.44 s figure should apply |
| implied pure compute: 0.44 x 48,000 steps | **5.87 h** |
| N29 training-loop wall | 13.22 h |
| N29 SLURM elapsed | 16.60 h |
| **implied duty cycle** | **35% vs elapsed, 44% vs training-loop wall** |

**Correction to 16a:** I reported N29's duty cycle as 80%. That used *summed epoch wall time* as if it
were compute. It is not -- epoch time includes the dataloader wait. On the pure-compute basis our own
model sits at **~35-44%**, i.e. **at or below Senseiver's 45%**, not comfortably above it. Our model
is among the *more* I/O-exposed in the fleet, which is consistent with the mechanism the FNO agent
identified: exposure tracks *cheap compute per item*, and our model does 39k-query subsampled steps.

**Caveat before this is used:** the 0.44 s may have been benchmarked without the binned spectral loss
that production includes -- exactly the flaw the FNO agent just found and patched in its own
benchmark (`bench_fno3d.py` was omitting it). Ours must be re-benchmarked the same way before either
number goes in the cost table.

**Consequences:**
1. The cost table compares numbers that may be measured differently across methods. Every entry
   should be **pure GPU compute, data-resident, including all loss terms**, and said to be so.
2. The budget comparison shifts again: if our model truly received ~5.9 compute-hours while baselines
   receive ~18 at 19.5 h wall, "matched budget" is off by ~3x, not the ~1.5x implied in 16a.
   Direction is still generous to baselines, but the magnitude must be stated honestly.
3. **Do not quote any duty cycle computed from epoch wall time.** Only compute-time instrumentation
   counts. Every baseline now logs this; our reference does not, and needs the same treatment.

### 16d. Measured duty cycles after staging, and a self-correction on S3GM memory

S3GM after staging: **epoch_duty 0.983-0.989** (loader wait 0.5-0.8 s per 75-step epoch), against the
45% Senseiver measured unstaged. Its `job_duty=0.621` from a 153 s validation run must **not** be
quoted -- ~58 s of that was fixed startup amortised over 1400 epochs in the real run; projected ~0.95,
and the run reports its own figure at exit.

FNO A/B, same node back to back: **16.180 s/epoch shared Lustre vs 16.127 s local NVMe (0.3%, noise)**,
duty cycle 0.895 vs 0.898. Compute-bound, so staging recovers nothing -- kept only for contention
tail-risk. Caveat stated by the agent: arm A warmed the page cache before arm B, so B understates a
genuinely cold read; the duty cycle is unaffected since 14.48 s of the 16.18 s epoch is measured GPU
compute regardless, leaving <=1.7 s for all non-compute combined.

**Mechanism now established across four methods: I/O exposure tracks CHEAP COMPUTE PER ITEM, not I/O
volume.** FNO has the fleet's largest read per step (~250 MB vs ~5 MB) and is immune, because 0.76 s
of full-grid FFT lets four workers hide a 31 MB read. Senseiver is exposed because its per-item
compute is cheap. This also predicts our own model is exposed (39k-query subsampled steps), which
16c confirms.

### 16e. CAUTION — the duty-cycle instrumentation we added today is itself new code, and it was wrong

CoNFiLD's agent read its smoke *output* rather than checking the exit code, and found **two bugs in
the duty-cycle reporting mandated in 16a** -- both of which would have put fabricated numbers in the
paper:

1. **Wrong denominator span.** The job start was stamped once per *job*, so stage 2's elapsed included
   all of stage 1: it reported `duty_cycle_vs_slurm=0.0238` for a stage whose true figure was ~0.17.
   **A completely fabricated stall**, and precisely the kind of number that would have been quoted as
   evidence of the I/O pathology in 16a.
2. **Cross-stage total double-counted itself** -- the aggregation looped over every record including
   the previously written `total`. Masked in production because each stage runs in its own run dir and
   the two never met, meaning the "totalled" figure was silently just one stage.

Both fixed and verified offline against hand-computed expectations (stage2 0.800 not 0.12; total
120.0 not 200; peak memory combined as **max** across stages, not sum).

**Generalisable caution: every baseline added similar instrumentation today under time pressure. It
can be wrong in ways that look like findings** -- a fabricated 2.4% duty cycle reads as a dramatic I/O
pathology, not as a bug. Before any duty cycle, step count or compute-hour figure goes in the paper,
verify it against a hand-computed case. Check specifically: does each stage divide by its own elapsed
time; does a combined figure actually sum the stages; is peak memory a max rather than a sum; and does
re-running the aggregation double-count.

This is the same lesson as the `--nf-accum` finding and the four CoNFiLD smoke cycles: **the failures
were visible in the output, never in the exit code.**

### 16f. RULING — latent FM receives ~2x our compute; let it run and disclose

The latent-FM agent asked which clock our "~20 h" refers to, since it lands on budget by one measure
and 15% over by another. **The answer is neither** -- the paper's figure matches no measured quantity.

| clock | N29 (paper model) | latent FM | excess |
|---|---|---|---|
| train-loop (summed epoch times) | **13.22 h** | 20.10 h | **+52%** |
| job wall-clock (SLURM elapsed) | **16.60 h** | ~23.0 h | **+39%** |
| pure compute (est., see 16c) | **~5.9 h** | ~12 h | **~2x** |
| paper's claim | ~20 h | -- | matches neither |

**Decision: do not cut stage 2.** (a) Cutting mid-run discards elapsed work and risks the
10,500-11,000 checkpoint window that the sigma measurement depends on. (b) Generous-to-the-baseline is
defensible, stingy is not: *"every baseline received at least our compute budget, and latent FM
received approximately twice it"* pre-empts the obvious reviewer attack. (c) It cuts in our favour on
CRPS -- a tie achieved with 2x our compute is a stronger result for us than a matched tie.

**The honest caveat: it cuts AGAINST us on relL2**, where they currently lead 0.574 to 0.593 with that
advantage. If the final numbers land close, a **compute-matched arm** (~9,400 stage-2 epochs) becomes
worth running. Logged as a contingency, not launched.

Latent FM step counts for the cost table: 19 steps/epoch; stage 1 = **95,000** optimizer steps,
stage 2 = **209,000** at 11,000 epochs.

**Root cause of why this was unanswerable:** `pointcloud_ffm` writes **no per-epoch timing record at
all** -- no `loss_history.csv`, nothing. Every baseline now logs per-epoch time, steps, and duty cycle;
our own reference logs none of it, which is why its cost numbers had to be reconstructed from `sacct`
plus log scraping. **The rerun of our model must fix this.**

## 17. MOST SIGNIFICANT FINDING — the paper's calibration claim is checkpoint-sensitive

DemoN15, two checkpoints, canonical layouts, 12 snapshots, NFE 4, K=8:

| | epoch 790 (`best`) | epoch 6000 (`last`) | delta |
|---|---|---|---|
| relL2 | 0.61482 | 0.60067 | 0.0142 |
| CRPS | 0.31315 | 0.33619 | 0.0230 |
| **spread/error** | **0.8163** | **0.5416** | **0.275** |
| **coverage_90** | **0.6509** | **0.4550** | **0.196** |

The paper claims ours at **0.63 spread/error and 0.54 coverage** against latent FM's **0.53 and 0.45**,
calls the separation **"decisive" at t ~ 9-28**, and it is the strongest surviving claim in the paper.

**At `last.pt` our model reads 0.542 and 0.455 -- statistically indistinguishable from latent FM.**
The published values sit *between* our two checkpoints. The checkpoint that wins on calibration is the
one that **loses** on relL2, so checkpoint selection is not metric-neutral.

Caveats, load-bearing: epochs 790 and 6000 are 87% of the run apart, so this is dominated by training
progress rather than checkpoint noise, and N=2. The archived-window rerun resolves it.

**This unifies with the spread-floor finding (section 11), and the direction matters.** spread/error
fell 0.816 -> 0.542 while relL2 barely moved (0.615 -> 0.601): the predictive **spread shrank
substantially over training while the error did not**. Combined with the density sweep -- where spread
saturates at ~0.102 and does not respond to a 200x sensor increase -- the coherent account is that our
predictive spread is a **learned, information-insensitive quantity that also drifts with training
duration**. If that holds, our reported calibration is an artifact of *where training stopped* and
*which sensor density we evaluate at*, not a property of the model.

**Consequences if the window confirms it:**
1. "Best-calibrated of any method evaluated" may survive as a *relative* statement at a stated
   operating point, but the **"decisive, t~9-28"** framing cannot -- a paired t-statistic computed at
   one checkpoint says nothing if the quantity drifts by 0.275 across training.
2. The mechanism narrative at `main.tex:190` needs rewriting regardless (already flagged in 11).
3. The honest and more interesting contribution is the **sensor-density calibration crossing** plus the
   **training-duration drift** -- both novel, both measurable, neither previously reported.

Commissioned on the archived window: spread/error and coverage_90 per checkpoint with the same
variance decomposition as relL2/CRPS, **plus the trajectory** -- if spread is monotonically shrinking
rather than fluctuating, that is a trend and must be stated as one, not as sigma.

### 17a. `TrainingHistoryLogger` reuse, with two bugs to fix while porting

Reuse `model_baseline.TrainingHistoryLogger` (`model_baseline.py:4676-4740`) for our own model rather
than writing a new logger -- proven across every baseline run; columns `epoch, train_loss, val_loss,
epoch_time_s, peak_gpu_mem_mb, cumul_train_time_s`. Two defects to fix:
1. **On resume it opens the CSV with mode `"w"` and resets the cumulative clock**, so any `--reload`
   run truncates its own history and under-reports total time. Append and seed from the last row.
2. **Duty cycle as `cumul_train_time_s / job_elapsed` is not an input-I/O diagnostic**, because
   `epoch_time_s` already contains dataloader waits -- the same conflation that produced CoNFiLD's
   fabricated 2.4%. Time loader wait separately.

### 16g. Staging measured: ~2x effective compute, and the disparity is now quantified

Senseiver, same model and config, staged vs unstaged:

| | s/epoch wall | duty cycle | compute delivered in 19.5 h |
|---|---|---|---|
| unstaged (DemoN42) | 13.49 | **45.0%** | ~8.8 h |
| **staged (DemoN43)** | **7.02** | **88.6%** | **~17.3 h** |

**1.92x wall speedup, ~2x actual compute inside the same budget.** Per-epoch compute unchanged at
~6.0-6.3 s and peak memory identical at 22,396 MB, confirming staging touches only I/O. Cost ~1 min
once. Projections: ~10,000 epochs, ~80,000 optimizer steps, ~40 figures.

**The fairness point cuts both ways and the pre-staging direction flattered us.** Before staging,
Senseiver ran at **0.57x our model's compute per wall-hour** -- a "20 h wall-clock matched" claim would
have handicapped the baseline by nearly 2x, in the direction that improves our result, and it would
have been **invisible in every metric we report**. That is a genuine fairness bug, not a performance
nuisance.

**Corrected disparity, using pure compute (16c) rather than the erroneous 79%/80% figures:**

| method | compute-hours | vs ours |
|---|---|---|
| **N29 (ours, unstaged)** | **5.87** | 1.0x |
| Latent FM (est.) | ~12.0 | **2.0x** |
| Senseiver (staged, 88.6% x 19.5 h) | ~17.3 | **2.9x** |

So every baseline now receives **2-3x our reference's actual compute**. Safe direction, but far larger
than the "matched budget" the paper claims, and it must be stated.

**RESOLUTION, now in flight:** the checkpointed rerun of our own model (**job 17002469 `ckptvar_n29`**,
plus `17002515 ckptvar_eval` and `17003171 ckptgen_fb`) is **staged and fully instrumented** -- the same
treatment every baseline got. **It becomes the new reference**, and the comparison becomes
like-for-like on compute-hours, optimizer steps and duty cycle simultaneously. This also finally gives
our model the per-epoch timing record it has never had (16f).

Caveat: the 5.87 h figure rests on the paper's 0.44 s/step, which is itself pending re-benchmark with
all loss terms included (16c). The rerun settles that too.

### 16h. A submitted safeguard is not an armed safeguard

The S3GM **abort watchdog silently failed**: `17001894` died in **6 seconds** because the script had
been rewritten to step-based flags (`--gate-steps`) while its launcher still passed the old ones
(`--gates 200 600 --min-epoch 40`), which `argparse` rejected. The job appeared submitted, queued, ran,
and died instantly -- **while training continued for hours with no abort criterion attached.** That is
precisely the "discover it at hour 19.4" scenario the gate existed to prevent. It surfaced only
because the agent checked job *states* rather than trusting the submission. Fixed and re-armed as
**17003576**, verified against the live log before resubmitting.

**Generalisable rule: a chained job that dies on startup is invisible until you need its output.**
Checking for `DependencyNeverSatisfied` does **not** catch it -- such a job sits happily PENDING with a
satisfiable dependency and only dies when it runs.

Fleet sweep performed: **no jobs with unsatisfiable dependencies**; the only sub-30-second failures in
the window are the two already-fixed smokes and this watchdog. Six chained jobs carry the artifacts
that make the audit worth having, and each owner was asked to verify -- by diffing launcher flags
against the script's `argparse`, confirming run directories resolve at *run* time rather than
submission, and preferably running the parse on the login node or a two-minute smoke:

| job | carries |
|---|---|
| `17002515 ckptvar_eval` | **the calibration-sigma answer** -- whether the paper's headline survives |
| `17001659/60 sen_eval` | Senseiver's acceptance gate |
| `17002675/76_ sitm_seval` | SiT's matched-capacity numbers |
| `17001895 s3gm_arch` | S3GM's checkpoint window |

Two verification properties the S3GM agent established and worth reusing: the watchdog **rescans the
whole log each poll**, so a late start still evaluates gates retroactively; and its gate points
(15,000 / 45,000 steps) are **exact multiples of the monitor interval** (75 steps/epoch x
`save_every: 40` = 3,000), so they fire on the intended epoch rather than one monitor late. It tested
the abort path against the live log using a **fake job id** so the test's `scancel` could not touch the
real run.

### 16i. Chained evals verified by RUNNING, not reading — plus one operational note

Senseiver's launcher smoke (`17003759`) submitted the **unmodified** eval launcher with a glob
argument, identical in form to the chained jobs. One output settled six questions at once:

```
resolved run dir: .../Baseline_senseiver_Stage1_DemoN41_20260828_152643
[eval] checkpoint=.../budget.pt epoch=38
[seedcheck] snap=29 sensors=39062 idx_sum=37987162596
[eval] snapshot 1/50 (id=29) rel_l2=0.90691 crps(MAE)=0.65865 senconsis=0.80569
```

- script args **survive `sbatch`** (a dropped `$1` would abort under `set -u`)
- **argparse accepts every flag** -- the S3GM failure mode is absent
- the glob resolved **at run time** to the right dir, excluding both `ABORTED_*` dirs
- the **canonical sensor fingerprint reproduces a third time**, now through the real launcher rather
  than a standalone harness
- **CRPS is finite**, confirming the K=2 `det_ensemble` fix (a literal K=1 array would give NaN from
  `fair_crps`'s `K(K-1)` divisor)
- snapshot order `29, 31, 0, ...` is the `rng.choice(50, 50, replace=False)` permutation over all 50

**OPERATIONAL NOTE for whoever picks this up:** `resolved run dir:` and `no run dir matching:` are
emitted **before** `LOG=` is defined, so they land in **`slurm-<jobid>.out`, not the named
`eval_senseiver_iclr_<jobid>.log`**. If a chained eval ever exits 2, the reason is in `slurm-*.out`.
This was deliberately **not** "fixed": editing a file that two pending critical jobs will read at run
time, purely for log tidiness, is exactly the trade that produced the S3GM watchdog bug.

Also worth reusing: the `ABORTED_*` **prefix** rename pushed superseded run dirs *outside* the glob
pattern. A **suffix** rename would have left them matching and the eval would have silently loaded the
wrong model. Prefix, not suffix, when retiring a run directory.

### 16j. A reusable diagnostic: does relL2 respond to sensor count at all?

Senseiver's eval smoke validated its acceptance gate on a deliberate throwaway (38-epoch) model, and
the *reason* it fails is the useful part:

```
[sweep] n_obs=1953  rel_l2=0.90842      n_obs=3906  rel_l2=0.90860
        n_obs=7812  rel_l2=0.90859      n_obs=15625 rel_l2=0.90861
        n_obs=19531 rel_l2=0.90861
[gate] sensor_consistency=0.743  monotone_in_sensor_count=false  passed=false
```

**relL2 flat to five significant digits across a 10x sweep in sensor count.** The model is ignoring
its conditioning entirely, and `monotone_in_sensor_count` catches it. The gate detects exactly the
failure it exists for.

**Generalise this: "does relL2 respond to sensor count?" is a cheap, powerful check for whether a
model uses its conditioning at all**, and it is far more diagnostic than a loss curve. Every
conditioned baseline should be swept before its number is trusted. It costs a handful of evaluations.

**And it connects directly to our own central limitation.** Our model *passes* this test on the
observed channels -- Ux improves 0.337 -> 0.098 across 0.1%-10% density -- but on Uy it reads
**1.048 -> 1.044, flat**. That is the *same signature* as a model ignoring its sensors, restricted to
the channels no sensor observes. The diagnostic that catches a broken baseline also characterises our
unobserved-channel failure, and in the same units. Worth using that framing in the paper: it is a
sharper statement than "correlation with truth is 0.087".

Also verified in the same smoke: **50/50 seedcheck lines report `sensors=39062`** -- the canonical draw
is stable across every snapshot, not merely at snapshot 29 where it was first checked.

## 18. FAIRNESS RISK created by our own parameter-matching procedure

The SiT agent measured that its capacity-matched width (hidden **240**, head_dim 60) is **~14% slower
per steady-state batch than hidden 256** (1.14 s vs 1.00 s) **despite 12% fewer parameters**. Cause:
tensor-core alignment -- non-power-of-2 / non-multiple-of-8 dimensions fall off the fast GEMM paths.

**This is a fairness problem, not a performance nuisance.** Our protocol matches parameters and then
budgets wall-clock. If shrinking a baseline's width to hit 6,506,253 lands it on a misaligned shape,
that baseline completes **fewer optimizer steps inside the same budget** -- a hidden handicap
introduced by *our own matching procedure*, invisible in the parameter count we report.

Exposure across the fleet:

| baseline | matched width | aligned? |
|---|---|---|
| **FNO** | **27** | **worst -- odd, nowhere near a multiple of 8** |
| SiT | 240 | measured 14% penalty vs 256 |
| S3GM | nf 32, ch_mult (1,2,3,4) -> 32/64/**96**/128 | 96 is a multiple of 8, likely fine |
| Senseiver | latent_dim 320 | multiple of 64, fine (chosen deliberately over 322) |
| CoNFiLD | 256 / 384 | fine |

FNO asked to measure width 27 vs 24 vs 32 at fixed modes and depth, and to report s/step with the
resulting parameter counts so the trade is explicit. Two defensible outcomes: keep 27 if the penalty
is small (closest parameter match), or move to 24/32 if it is material (accepting a worse parameter
match to avoid an alignment artifact). **Report the choice and the reason either way** -- a reviewer
can reasonably object to a baseline slowed by our matching procedure.

Note Senseiver's agent already avoided this deliberately: it chose `latent_dim=320` (-1.33%) over
`322` (-0.20%) precisely because 322 "costs ~10% throughput". That instinct was right and should have
been propagated at the time.

### 18a. Duty cycle varies for FIVE distinct reasons, not one

Superseding 16b's two-mechanism account. Measured across the fleet:

| method | duty | cause |
|---|---|---|
| Senseiver (unstaged) | **45%** | uncached HDF5 re-read per access -> **fixed by staging, 88.6%** |
| **SiT** | **66-71%** | **`num_workers` spawn + IPC for 750 MB batches -- NOT reads.** Staging measured and *rejected*: Lustre 5.45-7.23 s vs /tmp 5.91-5.98 s, no benefit. `nw=0` batch-1 is 0.6 s vs 3.5-5 s at `nw=4`. Won 2.8% from `persistent_workers` instead |
| Latent FM | 86.6% | page cache absorbs the file under `--mem=96G` |
| FNO | 89.5% | compute-bound; staging measured, no effect |
| CoNFiLD | 99.9% | explicit in-RAM snapshot cache; hot loop never touches HDF5 |

**The lesson is not "stage everything" -- it is "measure before fixing".** SiT's agent rejected staging
on evidence and took a different, smaller, real win. My original broadcast would have had it restart
for nothing.

### 18b. The archive-window prediction paid off

I flagged that archive windows specified as late epoch ranges are never reached if throughput falls
short. SiT measured it: 6,000 epochs at an all-in 14.98 s/epoch projects to **24.97 h against a 24 h
wall**, killing the run at ~epoch 5766 -- **just inside** its 5700-6000 window, yielding **3 archives
instead of 11**. Resized to 5,500 epochs with `archive_from: 5225`, `archive_every: 25` -> 12 archives,
all reachable, all multiples of `eval_every=5`. **Every agent with an epoch-range archive window
should re-check it against measured throughput, not projected.**

Also closed by SiT: the eval dependency was `afterok`, which would **never fire if training hits its
wall clock** -- but a timeout *is* the budget-matched case the protocol asks us to report, and `last.pt`
is written every 5 epochs. Switched to `afterany`. Worth checking on every chained eval in the fleet.

### 18c. Alignment hypothesis REFUTED for FNO -- measured, with a mechanism

Width sweep at fixed modes 16^3 / depth 4 / B=8, production step (job 17004486):

| width | params | dev vs 6.51M | s/step | vs w27 |
|---|---|---|---|---|
| 24 | 5,316,940 | **-18.3%** | 0.6857 | -12.3% |
| 26 | 6,239,874 | -4.1% | 0.7676 | -1.8% |
| **27** | **6,729,135** | **+3.4%** | **0.7816** | -- |
| 28 | 7,236,632 | +11.2% | 0.7927 | +1.4% |
| 32 | 9,451,620 | **+45.3%** | 0.8263 | **+5.7%** |

**Decisive test is the immediate neighbours.** If misaligned shapes fell off a fast path, 27 would
spike above the line through 26 and 28. It does not: mean(26,28) = 0.7801 vs measured 0.7816,
**+0.18%**, against 0.01% reproducibility across two jobs. **No cliff.**

**Mechanism for why SiT's effect does not transfer:** `torch.backends.cuda.matmul.allow_tf32 = False`
in this environment, and the cuFFT bf16 workaround runs the **entire FNO in fp32** -- so its GEMMs
never touch tensor cores. There is no fast path to fall off. Time is dominated by cuFFT on 125^3 and
bandwidth-bound pointwise work over 1.95M positions, both smooth in width. SiT's finding is real for
**bf16 tensor-core GEMMs**; this backbone is not in that regime.

**Width 32 is strictly worse on both axes** (5.7% slower AND +45% params), so "align upward" is not
available. Width 24 is 12.3% faster but -18.3% breaks the +-10% band -- handing FNO 21% fewer
parameters to buy 14% more epochs is the worse fairness objection. **Keep 27, no restart.**

**Tested negative result worth keeping:** confining the fp32 region to the spectral conv (to recover
bf16 tensor cores elsewhere) is **~2x SLOWER** -- +95.5% / +88.8% / +104.1% at widths 24/27/32 --
because per-layer bf16<->fp32 round-trips of ~1.5 GB `[8,w,125,125,125]` tensors cost far more than
the GEMMs save. Recorded behind a `spectral_fp32_island` flag defaulting to False.

**New entry for the fairness list: FNO trains in fp32 while the reference model uses bf16.** Forced by
cuFFT having no bf16 kernel, not chosen. **Favours FNO on accuracy, penalises it on throughput.**
Belongs alongside the constant-t conditioning channel and the missing domain padding.

**Corrected instrumentation: the production step is 0.7816 s, not 0.7623 s** -- the earlier figure
omitted the binned spectral loss. This is the number for the cost table, and **our own 0.44 s must be
re-measured data-resident with all loss terms before the two are compared** (see 16c).

FNO's archive window is safe by construction: the run is **wall-clock bounded** (`timeout` 70,200 s)
rather than epoch bounded, so SiT's failure mode cannot occur -- archives fire 66,600-69,840 s with
360 s margin, all 10 distinct and ~22 epochs apart.

## 19. Calibration drift generalises to ALL THREE datasets -- mechanism universal, exposure is not

Two-checkpoint test on FireBench (N18) and wing (N19), same protocol, both `best.pt` reruns verified
**bit-identical to the shipped numbers**, so this sits exactly on the published operating point.

| dataset | ckpt gap | relL2 delta | CRPS delta | **spread/err delta** | **cov90 delta** |
|---|---|---|---|---|---|
| **JHU** (N15) | 5210 ep | +0.014 | +0.023 | **-0.275** | **-0.196** |
| **Wing** (N19) | 2400 ep | -0.003 | +0.002 | **-0.090** (t=-11.7) | **-0.036** (t=-11.1) |
| **FireBench** (N18) | 1210 ep | +0.0006 | +0.0003 | **-0.019** (t=-9.6) | **-0.004** (t=-7.1) |

**The signature is identical on all three: dispersion shrinks monotonically with further training
while accuracy stays flat.** Sign-consistency makes it decisive -- spread/error fell on **8/8**
FireBench snapshots and **32/32** wing cases, while accuracy deltas are sign-inconsistent and null
(relL2 t = +1.7 and -0.47). So this is a **systematic property of our model's training**, not
checkpoint noise, and it is not a JHU artifact.

**But magnitude differs ~15x.** Ranked: JHU **0.275**, wing **0.090**, FireBench **0.019**.
- **FireBench: genuine negative result.** 0.756 -> 0.737 both support the reported "0.76"; cov90 moves
  0.004. The claim is safe. Report it as such.
- **Wing: marginal.** 0.68 -> 0.59 matters if the number is quoted to two decimals; cov90 -0.036 is
  ~a tenth of the reported 0.38.
- **JHU: the problem**, as already established in section 17.

Plausible reason JHU is the outlier: its epoch gap is 2-4x larger, and its condition (2 of 4 channels
at 1% coverage) leaves far more posterior width to lose.

### 19a. The uncomfortable structural finding

**On FireBench and wing the published numbers ARE the `best.pt` values, bit-for-bit -- endpoints of the
interval, not interior points.** Moving to `last.pt` pushes both strictly *down* (FB 0.756->0.737,
wing 0.685->0.589). So **the published calibration figures are the most favourable of the two
available checkpoints on both datasets.**

This is not cherry-picking -- `best.pt` is selected on validation loss, a legitimate and pre-declared
rule. But because dispersion shrinks monotonically with training while val loss keeps improving, that
rule **systematically selects the higher-calibration checkpoint**. Our calibration numbers are
therefore systematically optimistic by construction, across every dataset. That needs stating.

(JHU differed only because its headline came from a *different run* -- N29 spec02 -- than the probed
pair (N15), so it happened to land between them rather than at an endpoint.)

### 19b. Two further observations worth carrying

- **The wing's headline field is its most drift-exposed.** The paper quotes Cp only (0.126); Cp is the
  one field whose accuracy *worsens* with training (+0.014) while its spread/error collapses hardest
  (0.640 -> 0.416). The aggregate over 4 fields (0.473) hides this.
- **Widening the wing from 8 to 32 held-out geometries left the calibration delta essentially
  unchanged** (-0.090 vs -0.096) but **erased the apparent accuracy gain** (-0.010 -> -0.003,
  t = -0.47) -- confirming the n=8 accuracy movement was noise while the calibration movement is real.
  A good demonstration that the two metrics need different sample sizes.

Still not a sigma: both pairs are past the val-loss minimum, so this bounds drift rather than
estimating variance. Training-seed replicates remain the right instrument.

## 20. S3GM: the abort gate FIRED — and the stated explanation is falsified

The committed gate worked exactly as designed. Watchdog `17003576` hit gate 1 at **epoch 200 /
15,000 optimizer steps**, found `obs_rmse_z` above threshold, called `scancel` on `17001893`, wrote
`src/s3gm_verdict_17001893.json`, and exited. **Both jobs ended at the same second (21:43:46)** --
the signature of the abort. Not the throughput guard (15,000 steps reached at 3.52 h, inside the 6 h
deadline). Archiver cancelled too, correctly -- it would have made 10 identical copies of a static
checkpoint, which would have *looked* like a valid window.

(My "the watchdog finished before the cancellation" puzzle was my own error: the watchdog *started*
14:28 later because it was resubmitted after training began, so equal end instants with unequal
elapsed. Read end times, not durations.)

**Readings -- completely flat, no trend at any point:**

| epoch | opt steps | median `obs_rmse_z` | max_abs | clamped elems |
|---|---|---|---|---|
| 40 | 3,000 | 5.585e7 | 9.000e7 | 2,909,613 |
| 80 | 6,000 | 5.619e7 | 9.000e7 | 2,378,377 |
| 120 | 9,000 | 5.611e7 | 9.000e7 | 2,371,891 |
| 160 | 12,000 | 5.683e7 | 9.000e7 | 1,731,163 |
| **200** | **15,000** | **5.689e7** | 9.000e7 | 3,233,341 |

Duty cycle held at 0.991 (epoch) / 0.924 (job), so this is not a throughput story.

### 20a. DO NOT report the finding yet — "undertrained score" is dead

The agent's earlier explanation (untrained score -> huge x_hat_0 -> enormous gradient -> subsides with
training) is **falsified by its own loss curve**:

| epoch | 1 | 5 | 40 | 80 | 160 | 200 |
|---|---|---|---|---|---|---|
| train | 0.6915 | 0.3369 | 0.0830 | 0.0465 | 0.0308 | **0.0265** |
| val | 0.6112 | 0.3179 | 0.0732 | 0.0509 | 0.0331 | **0.0350** |

A loss of 1.0 is a zero score, so **0.027 means ~97% of the denoising signal is captured** -- the score
trained *well*, while `obs_rmse_z` stayed pinned at 5.6e7 with zero trend.

**Two possibilities remain and they must not be conflated:**
- **(a)** upstream's guidance rule genuinely does not converge at this scale/protocol -- reportable; or
- **(b)** a bug in our port of the sampler -- an implementation artifact dressed as science.

A well-trained score that still diverges raises (b) materially. Reporting (a) when the truth is (b)
would do to S3GM precisely what the pre-audit code did to Senseiver and latent FM.

**Discriminating experiment approved and launched** (~15 min; the epoch-200 checkpoint survives the
abort): sample at **alpha = 0** (pure unconditional -- if the sample is sane with std ~ 1, the score and
predictor paths are correct and the DPS term is solely responsible), then sweep
**alpha = 5.0 / 0.5 / 0.05 / 0.005**. This converts a binary "it diverged" into
**"the DPS term diverges above alpha ~ X"** -- precise and falsifiable -- or exposes the bug.

Required in the report: `obs_rmse_z` at each alpha against the **1.41 independence reference**;
whether **alpha = 5.0 is upstream's own rule at our 1% sensor fraction** (if divergence appears only at
the alpha upstream prescribes for our coverage, that is a far stronger claim -- the rule was calibrated
on denser observation than our protocol supplies); and, if a stable alpha exists, presentation as a
**clearly-labelled second row**, never a silent substitution -- the same standard applied to
Gen4Turb's annealing arm.

## 21. SPREAD FLOOR EXPLAINED — and the calibration claim is not defensible as written

**H1 (sampler artifact) rejected.** At NFE 16 the spread is multiplied by ~1.5 **at every sensor
distance**, leaving the profile shape and its floor intact (at-sensor spread 0.0917->0.1388 while
at-sensor |error| is unchanged, 0.090->0.096). The sampler sets the *level* of the spread; it does not
restore information-dependence. At NFE 4 on matched 50-snapshot sets, 1%->10% is a **10x sensor
increase for a 3% spread change** (0.0981->0.0951) against a 44% RMSE drop.

**H2 (uncontracted prior) rejected.** RFF-GP pointwise std at the query points is **1.018/1.006/
0.989/1.003**; against the ~0.10 output floor the flow contracts the prior **10x**. Caveat: the
residual keeps prior-like structure (deviation correlation length 0.168-0.184 vs prior 0.176-0.194,
**identical at 1% and 20% density**), and unobserved Uy is barely contracted (deviation std 0.87).

**H3 — the mechanism, and it is clean.** Spread and |error| binned by distance to the nearest
*same-channel* sensor:

| density | at sensor | d>=3.5 | d>=7 | d>=10 | d>=15 |
|---|---|---|---|---|---|
| 0.1% | 0.083 / 0.063 | 0.090 / 0.187 | 0.180 / 0.294 | 0.374 / 0.515 | 0.516 / 0.708 |
| 1% | 0.092 / 0.090 | 0.110 / 0.152 | 0.284 / 0.341 | -- | -- |
| 10% | 0.093 / 0.063 | 0.375 / 0.239 | -- | -- | -- |

**The spread is the same function of sensor distance at every density**, with a floor of 0.083-0.093
at zero distance that never falls. Density only reweights *where points sit on that profile*. Because
the spread profile is flatter in the tail than the error profile (6x vs 11x growth), the point-average
spread saturates while average error keeps falling -- that alone produces the 0.44->1.12 sweep.
**The model's behaviour does not change across the sweep; only the mixture of near/far points does.**

### 21a. PAPER CLAIM FALSE: spread IS distance-dependent

`main.tex:190` and `:426` state the standard deviation is "nearly independent of distance to the
nearest sensor -- flat at 1% coverage and still flat at 0.1%". **False per channel.** At 0.1% the
Ux/Uz spread grows **6.2x** from the sensor out to 15 cells. The published flatness is an artifact of
averaging over all four channels (unobserved Uy's ~0.87 spread dominates) or of binning that hides the
tail -- at 1% density **92% of points lie within 3.5 cells**, where the profile genuinely is flat to 8%.

### 21b. The calibration claim's operating-point sensitivity EXCEEDS the claimed advantage

- **NFE 4 -> 16** moves spread/error **+0.217** and coverage **+0.127**
- **K 8 -> 32** moves coverage **+0.107** (spread is stable: 0.0985 vs 0.0981, +0.4% -- so K=8 spread
  estimates stand; it is *coverage* that K=8 biases low)
- Claimed "decisive" separation over latent FM: **+0.097 / +0.087 at t~9-28**

**Both knobs move the metric more than the entire claimed advantage.** The 1% number reproduces
exactly, but it is a property of (density, NFE, K), not of the model. **As written the claim is not
defensible**: it requires the operating point stated and the baseline re-compared at matched NFE *and*
K. This is independent of, and additional to, the checkpoint drift in sections 17 and 19.

### 21c. Our production step time, measured honestly

| config | s/step | compute-h / 48k steps |
|---|---|---|
| compiled + spectral loss (**production**) | **0.5740** | **7.65** |
| compiled, spectral OFF | 0.4264 | 5.69 |
| eager + spectral | 0.7983 | 10.64 |

**The paper's 0.44 s is a spectral-loss-free benchmark** (0.4264 reproduces it). Honest production is
**0.574 s/step, +34.6%**. Duty cycle **58%** vs loop wall, **46%** vs SLURM elapsed. NVMe staging did
**not** change our epoch time -- our exposure is per-item CPU work (625 MB/batch), not storage
bandwidth. That is a **sixth** distinct duty-cycle mechanism in the fleet.

Validation: fingerprint matches; the 400k-subset protocol reproduces the published 1% row to three
decimals (relL2 0.5928 vs 0.5926, s/e 0.627 vs 0.627), so subsetting is an exact subsample. Note the
*published* sweep mixes 12- and 8-snapshot sets across densities; all of the above is one 50-snapshot set.

## 22. S3GM: it is NOT the DPS term -- it is the overlap-consistency term (beta)

The isolation sweep on the surviving epoch-200 checkpoint (N=200, canonical snapshot 29, fingerprint
`sensors=39062 idx_sum=37987162596`) discriminates cleanly:

| arm | std | `obs_rmse_z` | reading |
|---|---|---|---|
| **alpha=0, beta=0** (pure unconditional) | 0.42 | **1.10** | **sane** -- score, VESDE discretization, reverse-diffusion predictor and 14-window assembly are all CORRECT |
| **alpha=0.005, beta=0** (DPS only) | 0.41 | 0.995 | **stable**, already below the 1.41 independence line |
| alpha=0.05 / 0.5, beta=0 | -- | **0.094 / 0.047** | DPS *works well* -- an order of magnitude below independence |
| **alpha=0, beta=0.4** (consistency only) | **758** | **787** | **DIVERGES with no DPS involved at all** |
| alpha=5.0, beta=0.4 (full upstream) | -- | 5.589e7 | reproduces the training-run gate value 5.689e7 to 3 digits |

**Correction to sections 15a/20 and to what I reported:** I attributed the runaway to the DPS guidance
term, and the abort verdict's reason string says "the DPS term". **That is wrong.** DPS is stable and
effective. The unstable component is **beta, the window-overlap consistency term.** The verdict JSON
wording is being corrected rather than left misleading in the record.

Note this also means the earlier DPS analysis in 15b -- sum-of-squares vs norm, the sqrt(M) scaling,
alpha = alpha_case/sqrt(fraction) -- while correct on its own terms, was **not** the cause.

### 22a. The irony: our fidelity restoration is what broke it

`beta` was **absent from our port** (b=1) and was restored as class-(i) bug #10 during this audit.
So the pre-audit S3GM would not have diverged -- but it also was not faithful. **Restoring fidelity to
upstream is what introduced the divergence.** That is an honest and slightly uncomfortable finding, and
it belongs in the paper if this row is reported at all.

### 22b. The open question decides how narrow the claim is

beta=0.4 is upstream's **own published value**, and upstream's demos do run ~11 windows against our 14
-- so the window count is not the difference. The candidate difference is **problem size**: our
128^2 x 4-channel planes versus their 64^2 x 2-channel frames, with the consistency gradient scaling
with tensor size through the network Jacobian.

- If beta=0.4 diverges *because of* problem size, the honest claim is the **narrow** one:
  *"upstream's consistency weight does not transfer to this problem size."*
- Only if it diverges for a reason independent of scale would the **broad** claim
  (*"S3GM's guided sampler does not converge"*) be warranted.

**The narrow claim is the one to prefer unless the beta sweep forces otherwise** -- it is more precise,
more defensible, and does not overstate a failure of a published method. Recommendation held until the
sweep lands.

## 23. Senseiver: the restoration worked, and the floor comparison reframes the baseline

### 23a. NEAR MISS -- an early visual check would have produced a false negative

I asked for an epoch-250 eyeball as an early-warning signal. **At epoch 250 the reconstruction is flat
and visually indistinguishable from the pre-audit failure** (relL2 0.946). Reporting only that would
have suggested the faithful restoration had not worked. **At epoch 2250 it visibly tracks the truth**
(relL2 0.673) -- the large-scale layout is in the right places, heavily smoothed, no fine scales.

**It took ~1,000 epochs to start.** New standing rule: *an early visual check needs the model's
start-up timescale established before a null reading means anything.* A flat figure early is not
evidence of failure.

Trajectory on the fixed held-out snapshot (all still falling at 2250):

| epoch | Ux | Uy | Uz | p | SenConsis |
|---|---|---|---|---|---|
| 250 | 0.946 | 0.944 | 0.957 | 1.022 | 0.927 |
| 1250 | 0.783 | 1.026 | 0.793 | 0.991 | 0.768 |
| 2250 | **0.673** | 0.862 | **0.707** | 0.915 | **0.670** |

**The restoration demonstrably changed the behaviour**: the pre-audit model was flat at every epoch and
constant to five digits across a 10x sensor sweep. This one improves monotonically.

### 23b. The reframing: Senseiver's value-add is cross-channel inference only

Voronoi nearest-sensor floor is **model-independent** (a function of the sensor draw and the truth
alone), so it can be reported before the run finishes:

| field | Voronoi floor | Senseiver @2250 | verdict |
|---|---|---|---|
| Ux (observed) | **0.210** | 0.673 | **loses badly** |
| Uz (observed) | **0.219** | 0.707 | **loses badly** |
| Uy (unobserved) | 1.000 | **0.862** | **beats it** |
| p (unobserved) | 1.000 | **0.915** | **beats it** |

**Senseiver's entire value-add here is cross-channel inference; where it has direct measurements it is
strictly worse than trivial interpolation.** Mechanism: a 20,480-scalar bottleneck cannot carry a
7,812,500-value field at the fidelity 19,531 local samples trivially provide (381x compression).
That is a far more informative statement than a pass/fail on the acceptance gate.

**Our own model gets Ux 0.177 -- it beats the floor where Senseiver does not.** That contrast is a
genuine point in our favour and should be made.

### 23c. OPEN: three different interpolation floors, spanning 38%

| source | Ux relL2 | method |
|---|---|---|
| Senseiver agent, Voronoi | **0.210** | nearest-sensor fill, 1 snapshot |
| classical agent, KD-tree | **0.290** | `cKDTree` NN, 50 snapshots |
| classical agent, IDW k=8 | **0.235** | inverse-distance weighted, 50 snapshots |

These should be the same idea. The floor decides whether any method beats "do nothing clever", so it
**cannot be ambiguous in the paper**. Candidate causes: 1-snapshot vs 50-snapshot; nearest-fill vs
KD-tree query may not be the same operation; different sensor draws; periodic vs clipped boundary
handling (`periodic_kdtree: True` in the classical launcher). Reconciliation requested; the likely
resolution is to adopt the 50-snapshot canonical numbers as *the* floor. **The qualitative claim holds
under all three** -- ours 0.177 beats every one, Senseiver's 0.673 loses to every one -- but the margin
depends on which is quoted.

## 24. S3GM: BOTH of upstream's guidance hyperparameters diverge independently at our scale

Refines section 22. I reported that "DPS is stable and effective, beta is the unstable one". That is
**true only at small alpha**. Measured on the epoch-200 checkpoint:

| setting | result | stable? |
|---|---|---|
| alpha=0, beta=0 (unconditional) | std 0.42, `obs_rmse_z` 1.10 | baseline |
| alpha=0.05 / 0.5, beta=0 | `obs_rmse_z` 0.094 / **0.021** | **yes** -- DPS works *well* |
| alpha=1.0, beta=0 | std 797 | **no** |
| **alpha=5.0** (upstream's rule at 1%), beta=0 | **std 3.99e6** | **no** |
| beta=0.04, alpha=0 | std 1.213, max_abs 170.7 | **no** |
| **beta=0.4** (upstream), alpha=0 | std 758 | **no** |
| **largest stable beta** | **0.004** (std 0.440) | yes |

**So both of upstream's published guidance settings diverge independently at our problem size** --
alpha=5.0 alone, and beta=0.4 alone. That is a cleaner and stronger statement than "beta is the
culprit", and it is the accurate one.

The alpha threshold has an **analytical prediction that matches the measurement**: linearising the
guided update gives stability iff `|1-2*alpha| < 1`, i.e. `alpha < 1`. Measured edge sits between
alpha=0.5 (stable) and alpha=1.0 (diverges). Prediction stated *before* the data.

**Recommended deviation, minimal and structural:** keep upstream's `alpha ∝ 1/sqrt(observed fraction)`
**rule** -- it is the right structure for a density sweep -- and recalibrate `alpha_case` **0.5 -> 0.05**,
giving alpha=0.5 at 1% coverage. Combined with beta=0.004. One honest sentence, rather than abandoning
the rule.

### 24a. Cautions before the row is written

1. **The sweep was measured at epoch 200, not on a converged model.** The stability boundary may move
   with a better score. The pair (alpha=0.5, beta=0.004) has **not yet been tested together** and will
   be re-confirmed on the final checkpoint. Do not fix hyperparameters from a 15,000-step model.
2. **Set expectations low.** Even at the best stable setting the aggregate relL2 was **1.078** --
   observed channels improve (Uz 0.73, Ux 0.90) but unobserved are worse than predicting zero
   (Uy 1.52, p 1.16). Full training should improve it, but S3GM may land weak *while sampling stably*.
   **That is a real result, not a failure to report.**

### 24b. Relaunch and the corrected gate

Training **17014298** (full 1400 ep, DemoN94, distinct dir from the aborted DemoN93), watchdog
**17014339**, archiver **17014342**. The watchdog now aborts **only on training pathology** --
non-finite loss, or train loss > 0.30 at 15,000 steps (0.0265 was achieved; 1.0 = zero score) --
and prints `obs_rmse_z` labelled `(non-fatal)`. Regression-tested **against the very log the old gate
killed: PASS**; negative control with losses forced to 0.9: ABORT.

`s3gm_verdict_17001893.json` annotated with a `SUPERSEDED` block so its misleading "the DPS term"
reason string cannot be cited.

**Claim scope unchanged:** *"upstream's consistency weight does not transfer to this problem size"*,
now extendable to *"neither of upstream's guidance settings transfers"* -- and it must carry the note
that **we introduced beta ourselves as a fidelity restoration**. The pre-audit port had T=1, hence
b=1, hence `loss_consis` identically zero; it would not have diverged from beta, though it would have
diverged from alpha=5.0 had the alpha rule been implemented, which it was not.

## 25. CoNFiLD: the bottleneck effect, measured under control for the first time

Arms C (1024-d code) and F (384-d) differ **only in bottleneck size** -- same codec config, same seed,
same data, same schedule. This is the paper's latent-compression claim measured directly rather than
inferred across different methods.

`heldout_codec_rel_l2_zscore` at matched epochs:

| epoch | C (1024-d) | F (384-d) | P (1024-d) | **F - C** |
|---|---|---|---|---|
| 27 | 0.8925 | 0.8999 | 0.8925 | +0.007 |
| 139 | 0.8959 | 0.9355 | 0.8988 | +0.040 |
| 223 | 0.7357 | 0.9009 | 0.7430 | **+0.165** |
| 251 | 0.7244 | 0.8848 | 0.7089 | **+0.160** |

**The gap widens with training** -- so this is a capacity ceiling, not a transient. **C and P track each
other to within noise at every epoch**, which is the control: they share a stage-1 config and seed and
differ only in stage 2, so the codec replicate is faithful and the C-vs-F gap is attributable to the
bottleneck alone. F is also **slower** (57.7 vs 35.1 s/epoch), so at matched *wall-clock* it falls
further behind still -- **report both the matched-epoch and matched-wall-clock readings.**

**No arm is collapsed.** `latent_dependence` = 0.558 / 0.490 / 0.562 (C/F/P) against the **0.010** that
characterised the four arms that died earlier. Figures confirm: C at epoch 410 reproduces the truth's
large-scale structure with correct placement and sign in all four channels, error concentrated on thin
filaments -- the visual signature of a compressive bottleneck. F at 251 is **alive but severely
degraded**: faint, washed-out, low contrast.

### 25a. Spectral bands added -- the request would otherwise have been silently unmet

`ensemble_metrics` produces **no spectral output**, so the dissipation-band numbers I asked for would
not have appeared. `shell_spectrum` and `spectral_bands` were added to `confild_eval_unified.py`,
replicating `compare_spectra.py:21-26` and its bands (inertial k8-31, dissipation k32-62). Verified
**bit-identical** to `compare_spectra.py` on a random 125^3 field; a low-pass field returns inertial
0.244 / dissipation **0.000**, an identical field returns 1.000/1.000.

Reported for **both the ensemble mean and member 0**. The mean is smoother than any member by
construction, so quoting only the mean would understate small-scale energy for *every* method and mask
precisely the effect the bottleneck has. This applies to every baseline, not just CoNFiLD.

### 25b. Prediction on the record, before the data

Arm C's **dissipation-band ratio well below 1** with inertial closer to 1; arm F's dissipation ratio
**lower than C's**. If the evaluation does not return that, suspect the evaluation before the claim.

### 25c. Self-correction: staging is NOT active on these arms

SLURM snapshots the batch script at submission, and the arms were submitted before the launcher was
rewritten -- so they read from Lustre. **Not restarted, because the measurement says it was never
needed:** duty cycle **99.26 / 99.32 / 99.36%** (C/F/P), measured not extrapolated. `cache_snapshots`
genuinely insulates the loop on the shared filesystem under full fleet load. Consequence:
`duty_cycle_vs_slurm` will be absent from `instrumentation.json` for these arms; the table above is
computable from the logs regardless.

## 26. RECURRING TRAP: the cutout is NOT periodic, and the code assumes it is in at least three places

The JHU cubes are **125^3 sub-blocks of the 1024^3 DNS**, spanning **0.7609 of the 6.2832 (=2*pi)
periodic domain -- 12.11%**. **A sub-block of a periodic domain is not itself periodic.** Opposite
faces are 0.761 physical units apart, many integral scales, essentially uncorrelated. This has now bitten
the project three separate times:

| # | where | symptom | status |
|---|---|---|---|
| 1 | spectral analysis | raw FFT leaked a broadband floor; every method's band ratio drove toward unity | **fixed earlier** -- Hann taper, `shell_spectrum(window=True)` |
| 2 | `periodic_kdtree: True` in the classical baseline | wraps queries onto the opposite face -> a "nearest" sensor 0.761 away. Corrupts a boundary shell ~1 mean-sensor-spacing thick; at 19,531 sensors (mean spacing ~4.6 cells) **10-20% of query points exposed** | **diagnosed, confirmation running (17014430)** |
| 3 | `domain_padding=None` for the FNO (`evaluate_ffm.py:258`, `fno3d_backbone.py:340/390`) | spectral convolutions impose wrap-around across faces on a non-periodic field | **flagged class-(iii), NOT changed** |

**Rule to apply everywhere: any operation with an implicit periodic boundary -- FFT, spectral
convolution, periodic k-d tree, circular padding -- is wrong on these cutouts unless explicitly
corrected.** Worth one sweep of the codebase for further instances before the paper freezes.

### 26a. This resolves the three-way floor discrepancy

The 0.290 (classical, periodic) vs 0.210 (Senseiver agent, non-periodic) gap is **the periodic wrap**,
not a methodology difference. Ruled out first: the 0.210 **is** a 50-snapshot mean (my "1 vs 50
snapshots" hypothesis was wrong), and `nearest_sensor_fill_nodes` (`helpers_baseline.py:2330-2366`)
computes *exactly* the same hard nearest-neighbour quantity as a `cKDTree` query -- `sigma=0.05` only
shapes a `support` channel that is discarded.

**So the 0.290 is not a stricter floor; it is a floor computed under a false boundary condition.**

**Recommendation: quote the non-periodic nearest-neighbour floor** (50 snapshots, canonical draw), with
IDW as an optional smoother secondary row. Note this **cuts against us**: it makes the floor stronger,
so our model's margin narrows from 39% to **16%** (0.177 vs 0.210), and the claim must be stated as
*"beats nearest-neighbour interpolation"* rather than anything stronger. Senseiver loses to it by more.

### 26b. Consequence for the FNO row

`domain_padding=None` means the FNO is solving a periodic problem on non-periodic data. Upstream
neuralop provides `domain_padding` **specifically** for this case, so leaving it off is a choice that
**handicaps the baseline**. Before the FNO row is reported, either enable domain padding (fairer) or
disclose that we did not -- alongside the existing fp32-vs-bf16 and constant-t-channel entries in its
fairness list.

## 27. DeepONet (deterministic) — and the tightest bottleneck in the fleet

Upstream: `lululxvi/deepxde` @ `99b6620` plus the paper's own `lululxvi/deeponet` @ `8d62345`.
Reproduced exactly: unstacked branch/trunk + inner product + trainable scalar bias init 0;
**branch output linear, trunk output activated** (the asymmetry is upstream, `fnn.py:56-67`); relu;
Glorot-normal; plain Adam lr 1e-3, constant, **no clipping, no schedule**. **No class-(i) items** --
nothing existed in the repo to fix.

**Capacity: 6,505,732 trainable = −0.008% of target.** p=768, branch 736 wide, trunk 960 wide.

### 27a. The bottleneck is the story, and it is structural

| quantity | value | compression of the 7,812,500-value field |
|---|---|---|
| inner-product latent (4p) | 3,072 scalars | 2,543x |
| **pooled branch vector (binding)** | **736 scalars** | **10,615x** |

All sensor information passes through **736 numbers**. Against Senseiver's 20,480 and our 32,768,
this is by far the tightest bottleneck in the fleet.

**Architectural frontier, and this is the interesting part:** at 6.5M parameters this design **cannot
exceed a ~1,216-scalar pooled channel (>=6,424x) or rank ~1,376 under any width/p split**, because the
dense p-output layer forces the trade. **Attention architectures escape this by reusing weights across
latent slots.** That is a structural statement about operator-learning architectures rather than a
tuning observation, and it belongs beside the CoNFiLD C-vs-F bottleneck comparison.

### 27b. Fixed-vs-random sensors: option (a) is unevaluable, not merely less comparable

I framed this as a comparability trade. The sharper argument: evaluation draws sensors per snapshot
from `torch.manual_seed(seed*777+snap)`, and **a fixed-sensor branch cannot consume those draws at
all**. Running it would require changing the *evaluation* sensor set, breaking the paired per-snapshot
CIs and TOST tests fleet-wide. A third option — **(c) voxelise sensor values onto a fixed coarse
grid** — was rejected because it is exactly the zero-filled average-pool that crippled latent FM
(conditioning std 0.0014 vs field std 0.841).

**Chosen: (b), a permutation-invariant set-encoder branch.** Cost stated plainly: **this is not
vanilla DeepONet** and the table caption must say so. What keeps it recognisably DeepONet: branch ->
coefficients, trunk -> basis, inner product + scalar bias, and **no attention** — no sensor ever
attends to a query, which is the line separating it from Senseiver.

**Correction to my own framing:** (b) is *harsher* on capacity than (a), not softer. A fixed-sensor
branch with m=39,062 inputs spends 39,062*h on layer 1 alone; at 6.5M that affords h~128, so vanilla's
channel would be **~128 scalars** against our 736.

### 27c. RULING: persistent workers enabled — 33% duty was a 2.7x handicap

The agent measured **duty cycle 0.33, loader wait 0.43 of wall** (~3.5 s/epoch of DataLoader worker
respawn) and **deliberately left it at the fleet default** so "19.5 h means the same thing it did for
Senseiver" — correct reasoning when wall-clock was the budget unit.

**It no longer is.** Against CoNFiLD 99.3% / FNO 89.5% / Senseiver 88.6% / latent FM 86.6%, a 33% duty
cycle gives DeepONet **~6.4 compute-hours against Senseiver's ~17.3 — a 2.7x handicap** from worker
respawn, unrelated to the method. **Overturned: enable `JHU_PERSISTENT_WORKERS=1` and restart.**
Standing rule: generous-to-the-baseline is defensible, stingy is not.

### 27d. Class (iii), flagged and NOT changed

- **No positional encoding on the trunk.** Upstream has none; a relu-MLP basis on raw 3-D coordinates
  is spectrally biased. Adding PEs would likely help a lot but would no longer be DeepONet.
  **If it loses badly to the 0.215 floor, that is the finding, not a bug.**
- **Mean** aggregation over sensors (not sum/max) — count-invariant, standard DeepSets, but it changes results.

Fingerprint confirmed on a compute node: `snap=29 sensors=39062 idx_sum=37987162596`. Non-periodic
Voronoi floor independently reproduced at **0.2148** observed-channel mean, matching the fleet
reference (Ux 0.210 / Uz 0.219).

## 28. RETRACTION — the DeepONet "2.7x handicap" does not exist, and fleet duty cycles are NOT comparable

I overturned the DeepONet agent's decision on the basis of a 33% duty cycle against the fleet's
88-99%. **That comparison was invalid.** Verified in the source:

- `model_baseline.py:4911`: `self._cumul_train_time_s += epoch_time_s` -- it accumulates **epoch wall,
  loader waits included**.
- `train_Det_Baseline.py:242` feeds that as "cumul_compute".
- So the **fleet's `duty_cycle_compute_over_wall` is train-loop-wall / job-wall**, excluding only
  validation, figures and checkpoint I/O. **It is the same loop-wall/job-wall conflation flagged in 16c
  and 16e, one level finer, and I did not notice it was still present.**
- Confirmed on Senseiver's own file: `step_time_excl_val x steps = 0.805 x 8 = 6.44 s` and
  `epoch_time_excl_val = 6.44 s` -- **identical**, i.e. its "step time" is epoch-wall / steps and
  already contains the loader wait.

**DeepONet's 0.33 was strict pure fwd/bwd/step over wall -- a tighter quantity.** On identical smoke
data it reports **0.321 strict** and **0.767 fleet-definition** (~0.95 in production). Senseiver's
88.6% and DeepONet's 33% were never the same measurement.

**Second, independent refutation:** on the same dataset and the same 4.69 GB/epoch read,
**Senseiver is slower than DeepONet** -- 6.213/6.253 s per epoch and 0.777/0.782 s per step, against
DeepONet's **5.587 s and 0.698 s**. In equal wall-clock **DeepONet completes ~11% MORE optimizer steps
than Senseiver, not 2.7x fewer.**

Persistent workers were kept anyway (free, directed, removes a confound) but bought **0.3%**, not 2.7x:
epoch 5.603 -> 5.587 s, loader wait 3.221 -> 3.251 s. **Worker respawn was never the cost.** A train
epoch reads all 150 snapshots = **4.69 GB**; at the ~1.4 GB/s this filesystem delivers that is
**3.35 s of unavoidable read against 2.34 s of compute** -- matching the measured 3.25 s to 3%.

### 28a. Consequences

1. **No configuration of this arm can exceed ~0.45 strict duty on this filesystem.** The fleet is
   **I/O-bound at the epoch level** because every arm streams the same bytes. The large spread in
   reported duty cycles (33%-99%) is **mostly definitional, not real**.
2. **Duty cycle must not be compared across arms** unless measured identically. Only DeepONet currently
   times loader wait separately.
3. **For the paper, report optimizer steps and wall-clock as the budget evidence** -- both unambiguous
   and auditable. Treat duty cycle as a diagnostic and state its definition wherever it appears.
4. This does **not** undo section 16g: Senseiver's staged-vs-unstaged 13.49 -> 7.02 s/epoch was a real
   1.92x wall speedup. The reconciliation is that the read cost is **contention-dependent** -- which was
   the original point, that wall-clock is contaminated by what else is running.

**Credit where due: the agent refused to bank an unearned 2.7x.** It implemented the change I directed,
measured it, found it recovered 0.3%, and said so rather than letting the record claim otherwise.

## 29. CORRECTION to the I/O narrative — the steady-state cost is CPU augmentation, not the filesystem

Independent profile of `__getitem__` components (login node, so absolute times are inflated by
contention; the **ratio** is what matters and it is unambiguous):

| component | time / snapshot | share |
|---|---|---|
| `h5_read` | 12.2 ms | **1.8%** |
| `standardise` | 286.9 ms | 41.5% |
| **`octahedral_augment`** | **392.7 ms** | **56.8%** |

**I/O is under 2% of the per-item cost; CPU preprocessing is ~98%**, dominated by the octahedral
augmentation — a gather/permute over a 1,953,125x4 tensor performed per snapshot. The DeepONet agent's
direct profile on a compute node found the same shape (h5 5.1 ms / standardise 200.6 ms /
augment 500.6 ms).

Corroborating evidence it supplied: **node-local NVMe is *slower* than Lustre for this arm** —
staged loader wait 3.853 s vs 3.251 s on shared storage. If bandwidth were binding, staging would help.
This also vindicates the comment at `model_baseline.py:4859-4869` ("staging changed nothing, 5.9 s
Lustre vs 5.9 s tmpfs"), which I had read as describing worker spawn.

### 29a. What this does and does not change

**Partly misdiagnosed:** sections 16/16a/16b framed the steady-state ~3.5 s/epoch as filesystem
contention. It is **augmentation CPU**. The 868 s/epoch `ABORTED_iocontention` event *was* real Lustre
contention — a spike, not the steady state — so section 16g's measured 1.92x staging speedup for
Senseiver stands as a contention-relief result, not a bandwidth result.

**Does NOT change the baseline comparison.** The augmentation cost is **identical across arms** — same
dataset, same 48-element octahedral family, same `num_workers=4`, same batch size. It is a **uniform
tax, not a differential handicap**, so fairness between baselines is preserved and **no run needs
restarting**. Confirmed empirically on the production node under identical settings:
Senseiver **6.213 s/epoch, 0.777 s/step** vs DeepONet **6.210 / 0.776** — dead even.

**Correct actions:**
1. **Do not restart anything.** The tax is uniform.
2. **Do not raise `num_workers` for one arm** — that manufactures exactly the advantage we are trying
   to avoid. Fleet-level change or none, and not mid-fleet.
3. **Report optimizer steps and wall-clock** as the budget evidence (per section 28). The strict duty
   cycle will be ~0.33 for **every** arm once measured the same way; none are 88-99% strictly.
4. **For future runs** (not this fleet): `cpus-per-task=8` with more workers, or caching the augmented
   tensor, would roughly triple throughput. `standardise` at 41.5% is also unnecessarily expensive for
   a broadcast subtract/divide and is worth a look.

**Credit:** the agent retracted its own bandwidth explanation ("matches to 3%" was a coincidence) after
measuring, and declined to raise its own worker count unilaterally despite it being a clear local win.

## 30. FINAL — the spread floor, on a single matched 50-snapshot set (NFE 4 complete)

Supersedes the numbers in sections 11 and 21, which came from the published sweep that **mixed 12- and
8-snapshot sets across densities**. This is one matched set at every density, so the floor is not a
snapshot-set artifact:

| density | spread | RMSE | spread/error | coverage90 |
|---|---|---|---|---|
| 0.1% | 0.1478 | 0.3383 | 0.437 | 0.399 |
| 1% | 0.0981 | 0.1792 | 0.547 | 0.548 |
| 10% | 0.0951 | 0.1000 | 0.951 | 0.750 |
| 20% | 0.0945 | 0.0844 | **1.120** | 0.800 |

**From 1% to 20% -- a 20x sensor increase -- the spread moves 3.7% while RMSE falls 52.9%.**

**spread/error crosses 1.0 at ~12.2% coverage, with both bracketing points measured** (no
extrapolation). Under-confident below, over-confident above. The floor asymptote is ~0.0945; the
RFF-GP prior has pointwise std ~1.00, so **the flow contracts the prior 11x and then stops**,
regardless of how much data it is given.

Verdicts unchanged and now on final data: **H1 rejected** (NFE rescales the spread level uniformly
across sensor distance without restoring information-dependence), **H2 rejected** as an amplitude floor,
**mechanism is H3** -- a distance-to-sensor spread profile that is nearly density-invariant with a hard
~0.09 floor at zero distance, so raising density only reweights points onto the floor.

**Still pending:** 10% and 20% at NFE 16 (the second pins the NFE-16 crossing, currently extrapolated
at ~7%), and K=32 at 10%. One array task was cancelled and resubmitted as **17023433** with a single
query chunk after it was found pacing at ~18 min/snapshot and would have hit walltime at 44/50
snapshots producing **no** output -- the sensor encoding is recomputed per chunk per ODE step, so one
chunk instead of two roughly halves the cost.

## 31. H1 DEFINITIVELY REJECTED — and there is NO (density, NFE) setting that is calibrated

The decisive experiment landed. One model, no retraining, K=8, Ux/Uz, 50 matched snapshots:

| density | NFE 4 sp/err · cov90 | NFE 16 sp/err · cov90 |
|---|---|---|
| 0.1% | 0.437 · 0.399 | 0.584 · 0.542 |
| 1% | 0.547 · 0.548 | 0.810 · 0.713 |
| 10% | 0.951 · 0.750 | **1.353 · 0.875** |

**H1 rejected.** At NFE 16, 1%->10% is a **10x sensor increase for a 2.2% spread change** while RMSE
falls 41% -- the same floor as at NFE 4 (3.1% vs 44%). The **NFE16/NFE4 spread ratios are 1.35, 1.52,
1.53** across densities: near-constant. **Finer integration rescales the entire spread profile
uniformly and restores no information-dependence.** The floor belongs to the learned velocity field,
not the integrator — exactly as the sensor-distance profiles predicted.

### 31a. The consequence that settles the claim

**NFE 16 does not fix calibration; it overshoots it.** At 10% sensors it is strongly *over*dispersed
(spread/error 1.353) while at 0.1% it is still underdispersed (0.584). **There is no NFE setting that
is calibrated across densities** — NFE and density trade off, and any single (density, NFE) pair that
looks calibrated is a coincidence of that pair.

Quantitatively, over just this 3x2 grid:
- spread/error spans **0.437 -> 1.353**, a factor of **3.1**
- coverage90 spans **0.399 -> 0.875**
- the claimed advantage over latent FM is **+0.097 / +0.087**
- **the surface moves 9.4x and 5.5x the claimed effect**

The paper's 0.63 / 0.54 at 1% / NFE 4 is one point on a two-dimensional surface, and the separation it
claims is an order of magnitude smaller than the movement available from either knob. **This is
independent of the checkpoint drift (sections 17, 19) and of the K bias (section 21b), and it is on its
own sufficient to require the claim be restated with its operating point.**

### 31b. The two calibration metrics disagree — worth its own sentence

At 10% / NFE 16: **coverage90 = 0.875, essentially at the 0.90 ideal**, while **spread/error = 1.353,
35% overdispersed**. A reader shown only coverage would call that model well calibrated. Reporting
both, always, is not optional -- they answer different questions and can point opposite ways.

Remaining confirmatory: 20% @ NFE 16 (17023433) and K=32 @ 10% (16998708_1). Neither can change the
verdict.

## 32. All three measurement artifacts now EXCLUDED — the floor is real

K=32 landed at both densities, completing the artifact eliminations:

| K | density | spread | rmse | spread/err | cov90 |
|---|---|---|---|---|---|
| 8 | 1% | 0.0981 | 0.1792 | 0.547 | 0.548 |
| **32** | 1% | 0.0985 | 0.1766 | 0.558 | **0.657** |
| 8 | 10% | 0.0951 | 0.1000 | 0.951 | 0.750 |
| **32** | 10% | 0.0956 | 0.0955 | 1.000 | **0.860** |

| candidate artifact | test | verdict |
|---|---|---|
| snapshot set | single matched 50-snapshot set at every density | **excluded** (s30) |
| sampler (NFE) | floor persists at NFE 16; ratios 1.35/1.52/1.53 = uniform rescale | **excluded** (s31) |
| ensemble size (K) | spread moves **+0.4% / +0.5%**; K=32 floor 0.0985->0.0956 = **2.9% for 10x sensors** vs K=8's 3.1% | **excluded** |

**The floor survives all three.** Nothing in the analysis depends on how it was measured.

**Coverage is biased low by a flat +0.11**, not by anything model-dependent: 0.548->0.657 at 1% and
0.750->0.860 at 10% — the expected narrowness of an 8-member empirical 5-95 interval. The paper already
cites the 0.55->0.66 correction at 1% (`main.tex:149`); this **confirms it and extends it** — the offset
is **operating-point independent**, so it changes no ranking and no density trend. That part of the
paper is safe as written.

Remaining: 20% @ NFE 16 (17023433, ETA ~10:00), purely confirmatory. The N29 checkpointed rerun
(17002469, ETA ~10:30) with the variance decomposition chained (17003876) is the last open piece, and
it addresses checkpoint drift rather than the floor.

## 33. PERMANENT NOTE — the z-collapse bug, and an automated guard so it cannot recur silently

**This bug has now appeared three times.** It is invisible in every metric and it corrupts the one
tool we rely on to catch everything else.

**What it is.** A 125^3 field has 1,953,125 points but only 15,625 distinct (x,y) locations. Handing
all of them to a 2-D triangulation superposes ~125 z-planes per location and renders a **smear**, not a
cross-section, with the colour scale stretched to the volume min/max. Taking a **z-mean** instead of a
**z-slice** is the same error wearing a different hat.

**Why it is dangerous.** A collapsed render is far smoother than any real slice, so **every method
looks better and more similar than it is.** It survives review because the *numbers* are computed on
the full volume and are correct — only the picture lies. And the picture is what caught the flat
Senseiver, the collapsed CoNFiLD arms, and the latent-FM over-smoothing.

**History:** commit `988bc86` fixed it in the SiT and S3GM visualizers (they triangulated on (x,y)
only); reported again 2026-08-29 against the periodic per-epoch figures.

**THE RULE.** Any 2-D render of a volumetric field must take a genuine **slice at fixed z** — via
`helpers_baseline.midplane_slice`, or `reshape(nx,ny,nz,-1)[:, :, zmid, :]`. **Never** `.mean(axis=z)`.
**Never** hand all N points to a triangulation on (x,y). Use the **same plane for every method**.

### 33a. Automated guard: `src/check_no_zcollapse.py`

`python check_no_zcollapse.py` audits every plotting function for the signature and exits non-zero if
any renders a field without a slice guard. Run it before any figure goes in the paper, and after any
new visualiser is written. False positives are silenced with `# zcollapse-ok: <reason>`.

### 33b. Audit result 2026-08-29 — all ACTIVE per-epoch paths are CLEAN

Verified to slice correctly: `helpers.visualize_reconstruction` (ours),
`model_baseline.visualize_reconstruction_*` (latent FM / SiT / Senseiver / AE), `s3gm3d.py` (via
`midplane_slice`), `confild_upstream_training.py` (reshape -> plane -> imshow), `train_deeponet.py`
(`[:, :, zmid, :]`), `baseline_classical_figs.py` (`midplane()`). **I could not reproduce the collapse
in any active path** — the reported figures may predate a fix or come from a one-off script.

Ten flagged, triaged: **benign** — pairwise/summary heatmaps (matrices, not fields), `qualitative_wing`
(genuinely unstructured surface), `_build_structured_triangulation` (builder, caller slices).
**NEEDS CHECKING** — `helpers.save_smooth_mask_plot` (in *shared* helpers),
`qualitative_firebench.main` (FireBench is 152x126x192), `evaluate_coherence.save_worst_direction_spatial_map`,
plus `View_Dataset` and `train_finetune` (one-off scripts).
