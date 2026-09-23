# Cylinder surface-to-field task — results log (append-only; predictions frozen in SURFACE_TASK_PREREGISTRATION.md)

## 2026-09-08 classical anchors (job 3113594; 600 held-out frames, taps per field on the 360-cell ring, all 3 fields at the taps)

rel-L2 of the full wake (aggregate / Ux / Uy / p), CRPS = MAE for these deterministic rows:

| taps/field | kd-tree | IDW (k=8) | gappy POD r=80 | gappy POD r=20 |
|---|---|---|---|---|
| 32  | 2.16 | 2.08 | 9.98 (ill-conditioned) | **1.22** (0.78 / 2.45 / 0.45) |
| 64  | 2.15 | 2.12 | 4.37 | **0.92** (0.59 / 1.79 / 0.38) |
| 128 | 2.15 | 2.13 | 2.99 | **0.78** (0.51 / 1.48 / 0.36) |
| 360 | 2.14 | 2.14 | 2.32 | **0.69** (0.45 / 1.27 / 0.34) |

Reading against the pre-registration:
- **Prediction 2 (interpolation collapses) — confirmed, stronger than predicted.** IDW/kd-tree do not just fall to the train-mean floor (rel-L2 = 1); they are *worse than it* (2.1) at every budget, because nearest-tap extrapolation paints the whole wake with wall-adjacent values (near-zero velocity, wall pressure).
- **Prediction 3 (gappy POD may still win) — the classical floor does NOT survive.** Rank-20 POD, which reaches 0.056 with 1 % volume sensors, gets 0.69 from the full ring and 1.22 from 32 taps; rank-80 is ill-conditioned from surface data alone (least squares on a 1D manifold). Pressure is the recoverable channel (0.34–0.45), Uy is the failure (>1 at every budget). So surface-only sensing is where the cylinder stops being "solved by POD"; whether a learned method beats 0.69 is now the open question the fleet answers.
- Prediction 1 (DMF-Gen leads, gap grows as taps fall) — awaits the learned fleet (jobs 3113586–3113593, eval 3113684).

Caveat for the writeup: a ridge-regularized gappy POD would be the fairer high-rank anchor; if added it must be labelled as an improvement arm, not swapped in silently (handoff §4 spirit).

## 2026-09-08 23:05 — learned fleet (job 3116592; S3GM still running)

rel-L2 of the full wake, held-out Re {80,250}, n=50 frames, taps per field drawn
from the wall ring under the frozen protocol:

| method | 32 | 64 | 128 | 360 | CRPS@360 | sensor pool |
|---|---|---|---|---|---|---|
| Senseiver | 0.109 | 0.105 | 0.104 | **0.103** | 0.058 | 360 (mesh) |
| Geo-FNO | 0.161 | 0.119 | 0.119 | 0.119 | 0.073 | 62 (grid) |
| DMF-Gen | 0.271 | 0.252 | 0.247 | 0.245 | 0.111 | 360 (mesh) |
| MLP-RBF | 0.337 | 0.307 | 0.269 | 0.246 | 0.138 | 360 (mesh) |
| SiT | 0.494 | 0.476 | 0.476 | 0.476 | 0.180 | 62 (grid) |
| latent-FM | 0.631 | 0.569 | 0.569 | 0.569 | 0.220 | 62 (grid) |
| gappy POD r20 | 1.223 | 0.919 | 0.781 | 0.689 | — | 360 (mesh) |
| IDW / kd-tree | 2.08 | — | — | 2.14 | — | 360 (mesh) |

### Against the pre-registration

**Prediction 1 (DMF-Gen leads on CRPS and far-wake accuracy) — FALSIFIED.**
DMF-Gen is fourth at 0.245, behind Senseiver (0.103) and Geo-FNO (0.119) by
more than a factor of two. Its CRPS (0.111) is likewise behind Senseiver's
(0.058). We report this as it landed.

**The prediction's stated MECHANISM is nevertheless confirmed, exactly.** Grid
methods were predicted to lose as taps fall because they must rasterize the
ring. They do worse than that: their tap-count curves are *flat* from 64
onward (Geo-FNO 0.1191 at 64/128/360, identical to seven digits; same for SiT
and latent-FM), because the 360-cell ring rasterizes onto only 62 distinct
grid cells. Adding taps beyond ~62 is literally unrepresentable for that
family, while the three mesh-native methods keep improving to 360. That is the
capability-matrix claim measured rather than asserted.

**Prediction 2 (interpolation collapses) — CONFIRMED, and worse than stated.**
IDW and kd-tree sit at 2.1, i.e. worse than predicting the training mean, at
every tap budget.

**Prediction 3 (gappy POD may still win; "do not bury this outcome") —
FALSIFIED, and this is the headline.** POD's rank-20 basis, which *wins the
volume-sensing task outright at 0.056*, reaches only 0.689 from the full ring.
Senseiver beats it by 6.7x. **The ranking inverts between the two sensing
geometries on the same flow, the same frames and the same held-out Reynolds
numbers.** The classical floor is not a property of the cylinder; it is a
property of the cylinder *observed in the volume*.

### What this changes in the paper

The cylinder section currently says a learned method "only earns its
complexity here if it approaches the POD floor at comparable cost", and that
none does. That is true with volume sensors and false with surface sensors,
which is a sharper statement of regime-dependence than the paper currently
makes: the axis that decides the winner is the *sensing geometry*, not the
flow. Note also that grid-locked is not the discriminator here — Geo-FNO is
second — so the surface task separates deterministic from generative rather
than point-native from grid-locked.

---

# Unrelated finding from the 2D density sweep (2026-09-10, job 3116596)

Sweeping sensor count on Kolmogorov (65 / 164 / 655 / 1965 / 6554 = 0.1 / 0.25 /
1 / 3 / 10 % of 256^2) exposes a robustness failure in the row that WINS this
regime. Every model was trained with sensors drawn from U{65, 655}, so 1965 and
6554 are 3x and 10x beyond the training maximum.

| model | 65 | 164 | 655 | 1965 | 6554 |
|---|---|---|---|---|---|
| Senseiver | 0.942 | 0.929 | 0.913 | 0.908 | 0.907 |
| MLP-RBF | 0.936 | 0.780 | 0.639 | 0.593 | 0.578 |
| SiT | 0.940 | 0.780 | 0.494 | 0.310 | 0.261 |
| latent FM | 0.893 | 0.784 | 0.599 | 0.494 | 0.556 |
| **Geo-FNO** | 0.776 | 0.615 | **0.385** | 0.332 | **0.696** |

**CORRECTION (same session, 3 min later).** An earlier version of this entry
said Geo-FNO was the ONLY non-monotonic method. The latent-FM leg landed
afterwards and is non-monotonic too (0.494 at 3 % -> 0.556 at 10 %). The
corrected statement: **two of the six methods degrade beyond their training
density**, Geo-FNO severely (0.332 -> 0.696, +110 %) and latent FM mildly
(+13 %). Senseiver, MLP-RBF and SiT improve monotonically across the whole
range. Note SiT shares the grid interface and does NOT degrade, so
"grid-locked" is not the explanation; both degrading models carry a
fixed-resolution internal representation (FNO mode truncation, latent grid),
which is a hypothesis this sweep suggests but does not test.

Why it matters: the Kolmogorov headline is "a deterministic grid operator
learner beats the generative fleet". That claim is true only inside the sensor
density the model was trained on. A practitioner who adds sensors -- the one
intervention that always helps every other method here -- makes this model
worse. Reported alongside the accuracy row, not as a footnote.

Note this compounds with the budget finding: the same Geo-FNO row also received
4x the fleet's optimizer steps (200k vs ~50k). The budget-matched rerun is
training; whatever it returns, the density fragility is a separate property and
stands on its own measurement.

## S3GM row complete (2026-09-10, job 3125077, 1h35m)

Surface-to-field S3GM at 64 taps: rel-$L_2$ 0.8905, CRPS 0.4680,
spread/err 0.436, cov90 0.530. 32 taps: 0.897. Written to
`cyl_surface_ovr_s3gm_K8_nfe200.json` (the `_ovr` prefix is the density-override
marker; the job was submitted before the launcher change that also puts the
density in the name).

This completes Table~\ref{tab:surface}: all seven learned rows plus three
classical floors. **It changes no pre-registered conclusion.** S3GM is last
among learned methods on this task at 0.891, consistent with its documented
sampler fragility on the cylinder in the volume task (1.019 there), so it
neither rescues nor contradicts the two failed predictions already logged.

The 128- and 360-tap entries are NOT measured for this row. They are printed as
the 64-tap value with a dagger, because the wall ring rasterizes onto 62 unique
grid cells and a grid-locked model cannot represent a larger budget. For
Geo-FNO, SiT and latent FM the same identity WAS measured (bit-identical to six
decimals); for S3GM it is asserted from the pool cap. The paper now says which
rows are which rather than letting one claim cover both.
