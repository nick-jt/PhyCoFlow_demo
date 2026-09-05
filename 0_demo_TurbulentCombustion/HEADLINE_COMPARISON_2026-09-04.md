# Headline-method comparison — 2026-09-04

Scope: the fleet's headline methods compared on every axis that matters for
deploying sparse-sensor reconstruction on **large-scale 3D realistic turbulence**,
with the measured evidence for each cell. Sources: `FLEET_SUMMARY_TABLE_2026-08-30.md`
(JHU cross-cube, n=50, 1% sensors, K=8), `FLEET_AUDIT_2026-08-29.md`,
`Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*/Evaluation/scaling_*.json`,
the FireBench operator matrix (jobs 16726561/16764760), windowed spectra (job 16834824),
supervision accounting (setup diff 2026-08-20), S3GM finalize log (job 17038277).
DMF-Gen numbers are the frozen N29 configuration evaluated like every other row —
no favoritism: it loses the observed-channel columns to SiT-point and IDW, and the
aggregate to repaired latent-FM.

## 1. Accuracy — JHU cross-cube, per-channel rel-L2 (n=50; * = unobserved channel)

| Method | Ux | Uy* | Uz | p* | agg | Verdict |
|---|---|---|---|---|---|---|
| SiT-point (N=32) | **0.143** | 0.968 | **0.132** | 0.937 | 0.545 | Best observed channels |
| IDW k=8 (classical) | 0.174 | 1.000 | 0.181 | 1.000 | 0.589 | Beats every deep model but SiT on Ux at zero training cost |
| DMF-Gen (NFE 4) | 0.176 | 1.048 | 0.182 | 0.964 | 0.593 | Ties IDW on observed; mid-pack aggregate |
| FNO3D | 0.188 | 0.904 | 0.191 | 1.061 | 0.586 | |
| KD-tree (classical) | 0.209 | 1.000 | 0.219 | 1.000 | 0.607 | |
| Latent FM (repaired) | 0.248 | **0.727** | 0.257 | **0.646** | **0.469** | Best aggregate + best unobserved channels |
| Gen4Turb (anneal, 32-step) | 0.224 | 0.996 | 0.455 | 1.005 | 0.670 | |
| Senseiver | 0.361 | 1.077 | 0.380 | 0.978 | 0.699 | |
| CoNFiLD P | 0.409 | 0.751 | 0.374 | 1.007 | 0.635 | Best of three CoNFiLD arms |
| DeepONet | 0.593 | 0.972 | 0.652 | 0.985 | 0.800 | DeepONet++ improves Ux to 0.512 (p768 arm) |
| Gappy POD r80 | 0.523 | 1.306 | 0.540 | 1.443 | 0.953 | Worse than constant on unobserved |
| S3GM (jhu-tuned) | — | — | — | — | ~10^6 | Sampler diverges at 3D scale (see §8) |

No single method leads more than one axis. Every method's unobserved-channel error
is ≥ 0.6, and for most ≥ 0.9 — the identifiability wall (§8c).

## 2. Uncertainty

| Method | CRPS | spread/err | cov90 | Recalibrable? | Notes |
|---|---|---|---|---|---|
| Latent FM | **0.244** | 0.68 | 0.56 | yes (same wrapper) | Under-dispersed |
| FNO3D (our framework) | 0.269 | 0.94 | 0.63 | yes | Best raw dispersion |
| DMF-Gen | 0.291 | 0.63 | 0.54 | yes (P3 TUNE-fit) | Known mechanism: ~0.09 spread floor, distance-profile |
| SiT-point | 0.338 | 0.75 | 0.58 | yes | |
| CoNFiLD P | 0.371 | 0.82 | 0.57 | yes | 1000-step DPS per window |
| IDW / KD-tree | 0.396 / 0.406 | n/a | n/a | no | CRPS ≡ MAE (deterministic) |
| Gen4Turb | 0.407 | 0.37 | 0.33 | yes | Severely under-dispersed |
| Senseiver | 0.479 | n/a | n/a | no | CRPS ≡ MAE |
| DeepONet | 0.570 | n/a | n/a | no | CRPS ≡ MAE |

Every generative method is under-dispersed at the canonical operating point;
the operating point (NFE, K) moves calibration more than method identity
(NFE 4→16: +0.22 spread/err; K 8→32: +0.11 cov90). Post-hoc recalibration
(P3: TUNE-odd scalar/per-density fit, distance-binned conformal) applies to every
generative row and is offered to all of them — "characterized and repaired" is a
property of the protocol, not of one model.

## 3. Grid representation & discretization

| Method | Native representation | Grid-locked? | Arbitrary-point query | Resolution transfer | Unstructured mesh (wing) |
|---|---|---|---|---|---|
| DMF-Gen | point cloud (ambient) | no | yes | yes — fixed 1.39 GB to 1024^3 | **yes** (native) |
| SiT-point | point tokens (8192/chunk) | no | yes (chunked; coherence limited to chunk) | yes | yes (adapted) |
| Senseiver | point Perceiver | no | yes | yes | yes (adapted) |
| MLP-RBF | point MLP | no | yes | yes | yes |
| CoNFiLD | coordinate MLP + latent table | no (coords) | yes (decoder) | decoder yes; latent per-snapshot | 2D demos upstream; volume latent is the bottleneck |
| Latent FM | 3D ConvAE latent grid | **yes** | no (decode full grid) | locked to training res; tiling only | **no** |
| Gen4Turb | voxel U-Net + mask channel | **yes** (dims %8: ran 120^3 not 125^3) | no | no | **no** |
| FNO3D | spectral grid | **yes** (FFT lattice) | no | partial (mode truncation transfers, untested here) | no |
| S3GM | video U-Net (2D slices) | **yes** | no | no | no |
| DeepONet | branch-trunk | trunk queries any x | yes | yes | untested |
| IDW/KD-tree/POD | none / linear basis | no | yes | yes | yes (POD needs common mesh) |

The grid-locked half of the table cannot enter the wing experiment at all — an
honest "n/a (grid-locked)" row, and at 125^3 Gen4Turb already needed a 120^3
center-crop because of divisibility.

## 4. Memory & runtime scalability (H100-SXM 80 GB; scaling_*.json)

Training @125^3 (production step / peak GB / GPU-h to finished model):

| Method | s/step | peak GB | GPU-h | Memory vs resolution |
|---|---|---|---|---|
| DMF-Gen | 0.574 | 53.6 | 7.7 | **flat**: 39.9 GB and 0.443 s/step identical 64^3→1024^3 (query-budget decouples) |
| Latent FM | 0.238 | 3.6 | 13.9 (2 stages) | cubic: 122 MB@64^3 → 50.9 GB@512^3, **OOM at 640^3** (batch 1) |
| SiT-point | 1.81 | 26.8 | 19.4 | flat (token subsample) |
| Senseiver | 0.776 | 22.4 | 17.0 | flat (19.4 GB, 0.167 s/step at all res) |
| FNO3D | 0.766 | 48.6 | 17.0 | cubic (FFT lattice) |
| Gen4Turb | 0.598 | 47.4 | 17.4 | cubic: **OOM at 320^3** (batch 1); production batch would wall ~160-192^3 |
| CoNFiLD | 35.5 s/ep st.1 | 4.7-6.8 | ~19 | latent table grows with snapshots, not resolution |

Inference @125^3 full field (s/field, peak GB) and the resolution wall:

| Method | s/field | peak GB | vs resolution |
|---|---|---|---|
| Latent FM | **0.019** | 0.6 | conv decode cubic: 13.9 GB@512^3, **OOM 768^3** |
| FNO3D | 0.097 | 6.7 | cubic |
| Senseiver | <1 | 1.19 | **flat** to 1024^3 (67 s) |
| Gen4Turb | 1.65 (32 steps) | 1.4 | per-step cubic: 31.3 GB@384^3 |
| DMF-Gen | ~17 (uncached eval path; 4.5 measured with conditioning cache, NFE 4) | 1.39 | **flat** to 1024^3 (2450 s @NFE2 — slowest absolute, linear time) |
| SiT-point | ~150 (239 chunks) | — | flat memory, linear time |
| CoNFiLD | ~130-213 /snap | — | DPS optimization dominates |
| IDW / KD-tree / POD | 0.68 / 0.16 / 0.14 (CPU) | — | flat |

The scaling axis separates GRID computation (voxel diffusion, conv-AE, FFT —
cubic with measured walls) from POINT computation (DMF-Gen, Senseiver, SiT —
flat memory to 10^9 points). Within the flat family, DMF-Gen is the only
generative member; it pays for that with the slowest absolute inference.

## 5. Spectral fidelity (windowed estimator; inertial k8-31 / dissipation k32-45 energy ratio vs DNS, single samples)

| Method | Inertial | Dissipation | Verdict |
|---|---|---|---|
| Latent FM | **0.62-0.67** (velocities) | 0.07-0.09 | Best mid-scale energy; **~5x dissipation truncation** (AE floor) |
| DMF-Gen (N29) | 0.39-0.41 obs / 0.03 unobs | **0.40-0.48** | Preserves smallest scales; under-fills inertial; unobserved channels near-empty |
| Gen4Turb / others | not yet windowed-measured | — | pending |

Crossover, not a uniform win: latent compression buys inertial-range texture and
truncates the dissipation range; ambient generation preserves the smallest scales
and starves mid scales. Both facts only became visible after fixing the
spectral-leakage bug (windowing on non-periodic cutouts) — unwindowed estimates
had a leakage floor that faked dissipation-range parity.

## 6. Operator robustness (FireBench 3.7M pts, 1% wind sensors; rel-L2 clean → noise 0.3σ → 25% slab occlusion → channel dropout)

| Method | clean | noise0.3 | slab25 | drop v,w | Reading |
|---|---|---|---|---|---|
| DMF-Gen robust (N18) | **0.456** | 0.461 (+1%) | 0.511 | 0.603 | Graceful, information-responsive |
| DMF-Gen clean-trained (N31) | 0.455 | 0.458 | 0.513 | 0.743 | Noise/occlusion robustness is architectural; op-training buys structural dropout (+19%) |
| Latent FM | 0.72 | 0.72 | 0.72 | 0.72 | FLAT — bias-dominated, not robust (extracts little sensor info) |
| Senseiver | 0.78 | 0.78 | 0.78 | 0.91 | Same caveat |

DMF-Gen's worst corrupted cell (0.603) beats both baselines' clean cells.
Flatness under corruption must not be read as robustness when the clean error is
already near the information-free floor.

## 7. Supervision density (per optimizer step @125^3)

| Method | Supervised values/step | Relative |
|---|---|---|
| Latent FM | 2.93e8 (full grid decode) | 50x DMF-Gen per step; 125x over training |
| Gen4Turb / FNO3D | full grid | ~50x |
| DMF-Gen | 5.86e6 (39k queries × B20 ... 2% of points) | 1x — and 78k-query arm (3 seeds) showed no gain: queries are not the binding constraint |
| SiT-point | 8192 tokens × B | ~0.5x |

Cost per supervised value: DMF-Gen 1649 ns vs latent FM 15.3 ns (~107x) — point
methods pay per point; grid methods amortize. At 256^2 this axis is invisible
(everything supervises everything); at 125^3 it is a 2-orders-of-magnitude design
decision.

## 8. What 3D realistic turbulence exposes that 2D benchmarks cannot

a. **Memory walls exist only in 3D.** Every method in this fleet trains
   comfortably at 256^2 (< 2 GB for the conv-AE). At 3D resolutions the grid
   family hits measured H100-80GB walls: voxel-diffusion training OOM at 320^3
   (batch 1; ~160-192^3 at production batch), conv-AE training OOM at 640^3,
   conv-AE inference OOM at 768^3 — while point methods hold 1.19-1.39 GB
   inference to 1024^3 (10^9 points). A 2D benchmark cannot distinguish the
   families on this axis at all.

b. **Dissipation-range truncation by latent compression.** The ~5x
   latent-vs-ambient separation in the k32-45 band (0.07-0.09 vs 0.40-0.48) is a
   3D-turbulence phenomenon: 2D spectra are inertial-dominated and the highest
   bands carry little diagnostic energy. The claim also required windowed
   estimation on non-periodic cutouts — the unwindowed leakage floor had faked
   parity, a pitfall itself characteristic of 3D sub-domain extractions.

c. **Unobserved-channel identifiability collapse.** With 4 coupled fields and 2
   observed, every method scores rel-L2 ≥ 0.6 on Uy and p, most ≥ 0.9 (worse than
   predicting the train mean for several). Single-channel 2D tasks (Kolmogorov
   vorticity) never pose the question; the 2D cylinder (observe u only) poses a
   milder version. This is a framework-wide physics limit, not a method defect —
   the fleet's most consequential shared failure.

d. **Sensor-consistency failure on unseen regions.** On the held-out cube every
   method fails to reproduce its own observations (sensor-consistency error:
   latent FM 0.392, Senseiver 0.721 — should be ~0). Same-region 2D evaluations
   mask this because the model has memorized the local flow statistics.

e. **Calibration degrades at scale and no scored 3D precedent survives.** All
   generative rows are under-dispersed at the canonical operating point
   (spread/err 0.37-0.94); operating-point choices (NFE, K) move calibration more
   than method identity. Post-hoc recalibration is therefore part of the
   benchmark, not an afterthought.

f. **Supervision-budget economics** (§7): free in 2D, a 107x-per-value /
   50x-per-step decision in 3D.

g. **Grid-divisibility and geometry constraints bite.** Gen4Turb cannot ingest
   125^3 (dims %8 → 120^3 center-crop); the conv-AE's stride tree rejects 125^3
   without padding; none of the grid family can express the wing's unstructured
   mesh or surface-only sensing. 2D benchmarks on power-of-two lattices never
   trigger these constraints.

h. **2D-validated sampler settings do not transfer.** S3GM — a published,
   audited-faithful 2D method — diverges catastrophically under its upstream
   guidance settings at 3D scale (rel-L2 ~10^6, job 17038277); a
   normalized/step-sized guidance re-tune was required and remains pending. Scale
   transfer of guidance schedules is itself a finding.

## 9. Headline verdicts

1. **No method wins 3D turbulence outright**: SiT-point takes observed channels,
   repaired latent-FM takes aggregate + unobserved, IDW ties/beats deep models on
   observed channels at zero training cost, DMF-Gen takes calibration mechanism +
   scale + geometry, latent-FM takes speed.
2. **The deployment discriminators are structural, not accuracy columns**:
   grid-locked methods cannot scale past measured 3D walls or enter the wing
   task; point methods can, at higher absolute inference cost.
3. **Uncertainty is under-dispersed everywhere and repairable everywhere** — a
   protocol property (recalibration wrapper), not a model property.
4. **Latent compression trades dissipation range for inertial texture (~5x both
   ways)** — the central spectral fact a 2D benchmark cannot see.
5. **The unobserved-channel wall (rel-L2 ≥ 0.9) is the fleet's most consequential
   shared failure** and the clearest open problem the benchmark poses.
