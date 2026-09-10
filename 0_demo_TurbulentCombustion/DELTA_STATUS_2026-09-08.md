# PoF benchmark — Delta migration status (2026-09-08, session 1)

Delivered as a file per decision 0. Companion: `HPC_MIGRATION_AND_REMAINING_WORK.md`
(the plan this session executed), `SURFACE_TASK_PREREGISTRATION.md` (P0.5, frozen
before any run), `src/fleet_jobs_delta.txt` (label → SLURM job id, append-only).

## 1. What the host turned out to be

| Assumed in the handoff | Found on Delta |
|---|---|
| Delta, H100-class | **DeltaAI**: GH200 120 GB, **aarch64**, 4 GPU/node, partition `ghx4`, account `bilr-dtai-gh`, 750 GPU-h |
| `~/envs/jhtdb` shim exists | Nothing existed. Built `~/venvs/jhtdb_env` (torch 2.14+cu130 aarch64 wheel — works on the 595 driver) and the shim |
| checkpoints + eval JSONs available | **Neither transferred.** Retraining was started, then **cancelled at 13:20 on your instruction** — you are transferring the artifacts. Partial Delta runs parked in `Save_TrainedModel/_delta_partial_retrain/` |
| Shu Kolmogorov npy present | Not transferred; **re-downloaded from figshare** and verified frame-identical to the stride-4 H5 (needed by the protocol-B ablation) |
| wing data transferred | `shift_wing_processed_v3/` is **not** on Delta (deferred regime; irrelevant now) |

Portability fixes (all in the working tree, none committed):
- 314 files repointed (`/projects/ammoniacomb/...` → `/work/hdd/bilr/ntricard/datasets/...`, Kestrel worktree paths → this checkout); SLURM `--partition/--account` swapped everywhere.
- **KeOps × torch.compile crash** (`IndexError` in pykeops `complete_aliases`, reproduced in isolation): `Model._knn_search_keops` is now `@torch.compiler.disable`; the rest of the backbone still compiles. Also KeOps needs gcc-13 (`CXX` in the shim) to match the python module's libstdc++.
- Missing pips vs. the handoff list: `neuraloperator timm imageio pyvista pykeops`.
- `smoke_kolm2d.sh` did not actually gate (no pipefail) — fixed. `tectonic` 0.15 (aarch64) installed at `~/bin`.

## 2. Compute launched (all smoke-gated where a gate exists)

| Campaign | Jobs | Status at hand-off |
|---|---|---|
| Kolmogorov canonical fleet retrain | 7 | **CANCELLED** (transfer instead). Classical anchors kept and done: IDW 0.546, kd-tree 0.588, POD-80 0.680 |
| Cylinder canonical fleet retrain | 7 | **CANCELLED** (transfer instead). Classical kept and done: gappy POD r20 **0.056**, p 0.051; IDW 0.707 — reproduces the banked headline exactly |
| **Surface-to-field fleet (P0.5)**: all 7 methods + classical, taps 32/64/128/360 | 9 | smoke passed (DMF-Gen conditioned on 3×360 tap slots), fleet training |
| Seed replicates (P1-6): seeds 7 and 1337 × both fleets | 28 | queued/running |
| SiT patch-4 arms (P2-9), kolm + cyl | 2 | running |
| Observe-u-only cylinder (P2-10): DMF-Gen, Senseiver, classical | 3 | running |
| Evals chained on the NEW training (surface, u-only, 4 replicate fleets) | 6 | pending |
| Evals that need the transferred checkpoints — fleet evals (kolm, cyl), protocol ablation (all six arms), K-sweep, learned gallery dumps | — | **cancelled; resubmit after transfer** (one line each, §6) |
| Classical gallery dumps (kolm, cyl, surface) | 1 | done; the Kolmogorov sensor fingerprint (idx_sum 22128955) is **bit-identical to the H100 fleet's**, so transferred checkpoints can be evaluated here without an SKU caveat |

Budget: ~300 GPU-h planned of 750. Each eval leg is `--resume`-able; a timed-out eval is just resubmitted.

## 3. Paper

- `main.tex` scoped to **three regimes**; every cut block is verbatim in `Paper/pof2026/deferred_followup.tex`; compiles clean with tectonic (`main.pdf` rebuilt). Title unchanged ("across flow regimes" does not promise a count).
- Filled: the Strouhal `\todo` (St = 0.140/0.160/0.173/0.187/0.200/0.207 at Re 60–250, 3–5 % above Williamson/Norberg = the usual 2D over-prediction; `src/check_strouhal.py`). Capability heatmap regenerated with the wing column relabelled as an interface property.
- 19 `\todo`s remain; all but three are "fill from the JSONs once the 2D evals land" and will be filled in the next session. The three that need you: **author list/affiliations**, **acknowledgments**, **artifact DOI**.

## 4. What this session could NOT do — needs Nick

1. **P1-5 post-hoc recalibration TEST numbers** need the JHU checkpoints (86 GB, not on Delta). Either send the DMF-Gen N29 + latent-FM JHU checkpoints, or drop the "repaired" half of the calibration claim to a 2D demonstration.
2. **JHU eval JSONs** (small: `Save_TrainedModel/JHU/**/Evaluation/*.json`, `baseline_classical/*.json`, `Paper/iclr2027/figures/*.npz`) are needed to regenerate any mixed 2D/3D figure (`fig_perf_vs_sensors`, pareto, uncertainty, spectra) and for the paired-CI `\todo`s. Without them the existing PDFs stay as they are.
3. Authorship, acknowledgments, JHTDB redistribution terms.

## 5. Protocol notes worth knowing (surface task)

Grid-locked methods rasterize the 360-cell wall ring onto **62** grid cells (nearest fluid cell per surface point; stored as the sidecar `Cylinder2D_grid.surface_indices.npy` because running jobs held the H5 lock — re-run `src/add_grid_surface_indices.py` on an idle file to fold it in). Tap budgets above 62 are capped for that family and reported as such; that is the rasterization cost the task measures. Canonical draws are unchanged bit-for-bit when no pool is given (unit-tested).

## 6. Transfer list and what happens after

Put the Kestrel artifacts at the same relative paths under
`/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/`:

| What | Where | Why |
|---|---|---|
| `Save_TrainedModel/kolmogorov2d/{pointcloud_ffm,baseline_det,baseline_mlp_rbf,baseline_geofno,baseline_sit,baseline_s3gm,baseline_latent_fm}/<run>/` (at least `best.pt`, `run_config.yaml`/`args.json`, `dataset_stats.pt`; `Evaluation/*.json` if you have them) | same paths | canonical 2D fleet; the eval launchers resolve `ls -d <pattern> \| tail -1` |
| `Save_TrainedModel/cylinder2d/...` (same families; latent-FM stage 2 if it finished) | same | idem |
| `Save_TrainedModel/kolmogorov2d/litprotocol/` | same | protocol-ablation arms already measured |
| `Save_TrainedModel/JHU/**/Evaluation/*.json`, `Save_TrainedModel/JHU/baseline_classical/*.json` | same | mixed 2D/3D figures, paired-CI todos |
| `Paper/iclr2027/figures/*.npz` (qual_jhu, spectra_fields, qual_firebench) | same | JHU gallery / spectra figures |
| (optional) JHU DMF-Gen N29 + latent-FM stage-2 checkpoints | same | P1-5 recalibration TEST numbers |

**You do not need to tell me when it finishes.** A watcher is armed on
`Save_TrainedModel/{kolmogorov2d,cylinder2d,JHU}` and `Paper/iclr2027/figures`
(it ignores the directories this session's own jobs write to, so their traffic
cannot trigger or block it). It reports progress as checkpoints land and fires
once nothing has been written for 7 minutes.

On that signal I run `src/resume_after_transfer.sh`, which **trains nothing and
recomputes nothing**:

* per dataset and per method, a leg is submitted only when a run dir with
  `best.pt` exists *and* its fleet JSON is absent — any transferred JSON is
  taken as final;
* the eval driver runs `--resume`, so transferred `crps_snapN.json` files
  short-circuit those snapshots instead of re-sampling them;
* the protocol ablation gets only the arms whose JSONs are missing; the K-sweep
  and gallery dumps self-skip existing outputs.

Accuracy rows mix safely across the two machines: the GH200 reproduces the H100
sensor fingerprint bit for bit (Kolmogorov frame 256, `idx_sum` 22128955).
**Cost rows do not** — each JSON records its own `gpu`, and anything recomputed
here is GH200 timing that must not be merged into an H100 cost column. If you
transfer the Kestrel `Evaluation/*.json` too, nothing is recomputed at all and
the cost table stays pure H100.

Run `DRY=1 bash src/resume_after_transfer.sh` yourself any time for the
inventory without submitting anything.


## 7. Transfer landed — remaining compute (2026-09-08 14:10)

The artifacts arrived at **`/work/hdd/bilr/ntricard/Save_TrainedModel`** (a sibling
of the repo, not inside it), 76 GB. The repo now reaches them through per-family
symlinks, so every launcher's `ls -d <pattern> | tail -1` resolves normally and
this session's own outputs are untouched. The watcher never fired because it was
armed on the in-repo path; it has been stopped.

**Nothing canonical needs recomputing.** All 14 model x dataset fleet evaluations
arrived complete (n=50 snapshots each), and two items the handoff listed as
pending had in fact finished on Kestrel after it was written:

* GeoFNO Kolmogorov eval — done;
* cylinder latent-FM stage 2 (trained 2026-09-07) and its eval — done;
* DMF-Gen Kolmogorov K-sweep (K = 8/16/32/64) and the 5-density sensor sweep — done.

Remaining GPU-hours, measured from live epoch rates (not estimates):

| Campaign | Priority | GPU-h left |
|---|---|---|
| Seed replicates, 2 seeds x 2 fleets | P1-6 reviewer-critical | 72 |
| Surface-to-field fleet + eval | P0.5 new experiment | 21 |
| SiT patch-4 arms | P2-9 optional | 26 |
| Observe-u-only cylinder + eval | P2-10 optional | 4 |
| Protocol-ablation arms b_senseiver / b_sit / uniform_b_dmfgen | P0-2 last gap | 1.5 |
| Cylinder latent-FM gallery panel | P0-2 last gap | 0.3 |
| **Total** | | **~125** |

Allocation balance is 708 of 750 GPU-h, so this fits roughly six times over.
Wall-clock is about 14 h because ~40 jobs run concurrently; the critical path is
the cylinder SiT patch-4 arm.

Two jobs (both patch-4) had ETAs past their 12 h wall and would have timed out;
resubmitted at 24 h, resuming from their checkpoints. If you want the cheapest
credible cut, **drop the patch-4 arms (26 GPU-h)** — they only quantify how much
of SiT's error is patch granularity, which the galleries already show
qualitatively. The seed replicates are the expensive item but they close the one
error bar a reviewer will ask for.


## 8. Post-transfer work completed (2026-09-08 14:30)

**Two path problems the transfer carried with it, both fixed:**

1. 103 run configs (`args.json` / `run_config.yaml`) inside the transferred
   tree still pointed at `/projects/ammoniacomb/...`. Repointed to the Delta
   dataset paths, with the original kept beside each as `*.kestrel_orig`.
   All three in-scope datasets resolve; the three that do not (wing, CoNFiLD,
   JHU scale campaign) are the deferred material.
2. The latent-FM stage-2 checkpoints record their stage-1 path **inside the
   `.pt`**, so no config edit could reach it. `eval_kolm_ensemble.py` now
   relocates a stale stage-1 path by its `Save_TrainedModel`-relative tail and
   fails loudly if no local counterpart exists, rather than silently loading a
   different autoencoder.

**Paper (`main.pdf` rebuilt, 15 `\todo`s left, down from 19):**

* `tab:kolm` completed and a new `tab:cyl` created, both generated from the
  canonical JSONs by `src/figs_pof/make_2d_tables.py` rather than hand-copied.
* **The Kolmogorov section was wrong and is rewritten.** It claimed \dmfgen{}
  and SiT were tied at the front; the landed Geo-FNO row wins outright at
  0.385, 21% ahead of the best generative model. The section now reports that,
  plus the finding it exposes: Geo-FNO is beaten on CRPS by every generative
  model, so accuracy and the proper score disagree *across method families* --
  the distortion--perception tradeoff as a family-level split.
* **A claim repeated in four places was falsified by the new numbers.** "Every
  scored ensemble is under-dispersed" is false: cylinder latent FM (1.26) and
  Kolmogorov S3GM (1.07) are over-dispersed. Corrected in the abstract,
  introduction, calibration section and conclusion.
* Abstract headline numbers filled; Strouhal, cylinder cross-channel, and the
  discussion's "confirm with cylinder numbers" TODOs closed.
* **Paired statistics added** (`src/paired_ci_2d.py`): all methods share the
  same 50 frames and seeded draws, so comparisons are paired. Bootstrap CIs
  plus Wilcoxon, with a gap called resolved only when both agree. Geo-FNO's
  Kolmogorov lead is resolved; \dmfgen{}/SiT/S3GM are a genuine tie group. On
  the cylinder, gappy POD's win over the best learned method is resolved
  (-0.037, CI [-0.042,-0.032]). The classical floors were folded into the same
  paired test by subsetting their per-snapshot records to the fleet's frames.
* **P1-5 (post-hoc recalibration) is done and needed no GPU time** -- the
  scalar estimator runs off the summary JSONs. Fitted on TUNE, frozen on TEST:
  \dmfgen{} on JHU repairs at three densities (0.436/0.550/0.949 -> 1.001/
  1.017/1.016), latent FM 0.706 -> 0.993, and all four Kolmogorov generative
  rows land within 3% of unity, including S3GM which is *shrunk* rather than
  inflated. The cylinder is reported as a counterexample: the scalar does not
  transfer across two held-out Reynolds numbers. Artifacts in
  `Paper/pof2026/recalib/`.
* Figures regenerated from the landed data: performance-vs-sensors (now with
  the 2D curves), both reconstruction galleries, capability heatmap.

**One measurement pitfall caught rather than published.** A new 2D spectra
panel initially scored a "dissipation band" at $k\ge48$, where the true
spectrum holds 7e-4 of its peak -- the ratios there reached 119x and measure
spurious energy in near-empty modes, not fidelity. The bands were redefined to
the energy-carrying range and the far tail is now reported separately and
labelled as such. The panel is not yet cited in the text: the 3D convention
averages 4 snapshots and only one 2D frame is dumped, so the comparison would
be weaker than its counterpart.

Remaining compute is unchanged at ~125 GPU-h (balance 699/750); the two
canonical gaps (protocol-ablation arms, cylinder latent-FM gallery panel) are
running.


## 9. Fairness correction found in the 3D table (2026-09-08 15:05)

Reading the recovered fleet audit against the transferred artifacts turned up
a row that was **unfair to a cited method**. `tab:jhu` reported S3GM only as
"sampler diverges at 3D scale", but the finalized evaluation is in the
transferred tree
(`baseline_s3gm/matched/.../eval_s3gm3d_{jhu_tuned,normguided_valtuned}_best.json`,
canonical protocol: seed 0, n=50, K=8, 1% sensing) and shows the method works
once its guidance is retuned:

| arm | upstream-faithful | agg rel-$L_2$ | CRPS | $U_x$ | $U_z$ |
|---|---|---|---|---|---|
| published guidance (a=0.5, b=0.4) | yes | 8.9e5 (diverges) | -- | -- | -- |
| retuned (a=0.05, b=0.004) | no | 0.682 | 0.368 | 0.362 | 0.350 |
| normalized guidance, TUNE-fitted | no | 0.628 | 0.343 | 0.260 | 0.254 |

All three now appear in the table, the two working arms carrying an explicit
"deviates from the published setting" marker, with a new paragraph in the 3D
section and a corrected Pareto caption. The audit's warning ("never quote
Ux=2.2e6 as a model result") is now honoured in the paper itself, not only in
the archive. One refinement on the audit: it attributed the divergence to an
old baked-in coefficient, but the finalized run records that the *published*
coefficients each diverge independently at this scale, so the paper says that
instead.

I also checked whether the normalized arm's TUNE-fitted coefficients inflate
its reported number: fitted on odd snapshots, its tune-vs-test difference is
0.0002 relative $L_2$, so reporting it over all 50 is safe and the caption
says so.

Also completed: paired CIs for the 3D fleet (`src/paired_ci_jhu.py`). Latent
FM's lead is resolved against both rows behind it, but **FNO3D and \dmfgen{}
are statistically indistinguishable** (-0.007, CI [-0.027,+0.015], p=0.33)
despite being printed as separate ranks. Only three of seven rows could be
compared frame-by-frame: SiT-point and Gen4Turb have no per-snapshot payload
in the artifact set and Senseiver's uses an older schema. The script maps each
method to an explicit file path and *verifies* it reproduces the published
aggregate before using it, so a wrong file cannot enter the statistics
silently.


## 10. Capability-matrix error found and corrected (15:15)

Checking the capability matrix against what the runs actually consumed turned
up a wrong cell. The matrix claimed SiT-point enters all three regimes
grid-free. In fact SiT is point-tokenized **only in the 3D run**: both 2D
configs use `tokenizer: patch`, and the cylinder one reads
`Cylinder2D_grid.h5`, i.e. the resampled uniform-grid export rather than the
body-fitted mesh. The cylinder cell is now RS, both 2D cells are footnoted,
and the methods section says which configuration each regime reports.

I had also repeated the draft's phrase "the ambient point methods (DMF-Gen,
SiT, S3GM)" in the Kolmogorov section I rewrote. On that grid only DMF-Gen is
ambient point-based; SiT is patch-tokenized and S3GM voxel-tokenized. Both
sentences are corrected.

This matters beyond a table cell: "requires resampling" now rests on the
cylinder alone in the three-regime scope, so a cell that silently credits a
grid-locked configuration as grid-free weakens exactly the axis the paper says
is thin.


## 11. Remaining TODO ledger (15:21) -- 13 left, none silently stuck

| # | TODO | Blocked on |
|---|---|---|
| 1-3 | co-authors, acknowledgments, artifact DOI + JHTDB terms | **you** |
| 4 | abstract artifact DOI | **you** (numbers are filled) |
| 5 | cylinder cross-Re breakdown + surface-tap variant | surface fleet (running) |
| 6 | Kolmogorov operator variants | not run; needs a decision, not a blocker |
| 7 | 2D learned-fleet density curves | job 3114295 (queued) |
| 8 | observe-u-only arm | training (running) |
| 9 | Senseiver protocol-B cell | job 3113952 (queued) |
| 10 | seed-replicate numbers | replicate fleets (running) |
| 11 | conformal-quantile TEST coverage | job 3114350 (submitted, array 0-2) |
| 12 | CoNFiLD strict-2048 row | needs a protocol stamp; measured value recorded in the caption |
| 13 | paired CIs for 4 more 3D rows | per-snapshot payloads absent from the artifact set |

Closed this session beyond the earlier list: the cost-table "(d)" ambiguity is
now stated as an upper bound with the reason the two kinds are not merged (the
H100s are gone, so unifying would mean re-measuring every row on current
hardware -- a hardware restatement, not a footnote, so it is left as an
explicit choice); and the capability matrix is finalized after checking every
cylinder cell against the export its run actually opened.

Route-2 calibration dumps were submitted for \dmfgen{} at three densities. Their
canonical-fingerprint gate doubles as a check that the GH200 reproduces the 3D
sensor draws, the way it already does for the 2D ones. latent FM's route-2 dump
is not covered by that launcher and remains open if the conformal comparison
needs to be two-sided.


## 12. Training-budget defect found in the Kolmogorov fleet (15:27)

Sanity-checking that the replicate seeds took effect (they did: seeds 42/7/1337
give distinct losses) surfaced something more serious in the **canonical**
rows. Comparing configured against as-run optimizer steps:

| row | target steps | as-run | share |
|---|---|---|---|
| Senseiver | 50,000 | 50,000 | 100% |
| SiT | 52,000 | 52,000 | 100% |
| latent FM | 50,000 | 50,000 | 100% |
| Geo-FNO | (config 200k) | 57,200 | at the fleet budget |
| **MLP-RBF** | 50,000 | **22,200** | **44%** |
| **S3GM** | 51,200 | **25,600** | **50%** |

Two reported rows trained on roughly half the budget the paper calls matched.
That matters most for S3GM, which sits inside the leading tie group (0.498)
on half the training of the models it ties with, and it is exactly the kind of
asymmetry this benchmark exists to catch.

Both are now disclosed in the Kolmogorov section as lower bounds, and
full-budget reruns are training (jobs 3114377 / 3114378) under a **separate**
`kolmogorov2d_fullbudget` save root with `reload: false`, so the transferred
canonical artifacts are never overwritten. Their eval is chained
(`DATASET=kolmogorov2d_fullbudget`). When they land, both rows and the
disclosure paragraph get replaced. Cost columns for the reruns will be GH200
rather than H100 and must be labelled accordingly.

Geo-FNO is worth a note: its config asks for 200k steps but it stopped at
57.2k. That is consistent with the fleet's ~50k budget rather than short, so
it is left alone, but the config and the as-run value disagree and the config
should not be read as the budget.


## 13. Budget audit completed across both 2D fleets (15:32)

Extending the check: **the cylinder fleet is clean** -- all seven rows ran
49.4k-51.0k steps, i.e. their full budget -- so the paper's central cylinder
claim (rank-20 gappy POD beats all seven learned methods) is *not* confounded
by under-trained baselines. **DMF-Gen also completed its budget on both
regimes** (Kolmogorov 400/400 epochs = 51.2k steps; cylinder 850/850 = 51.0k).

That last point matters for how the defect reads. The two shortfalls are in
baselines, not in the benchmark's own method -- the direction that flatters
the authors -- so the fairness section now states the audit, its coverage (all
14 2D rows), its result, and the direction of the error, instead of leaving a
reviewer to wonder whether the proposed method was the better-trained one.


## 14. CORRECTION to sec.12: the under-training finding was WRONG (15:45)

Section 12 reported that MLP-RBF (22.2k steps) and S3GM (25.6k) trained on half
the Kolmogorov budget. **That was wrong, and the paper text asserting it has
been replaced.** Both sources I used cover only the final chunk of a *resumed*
run: `loss_history.csv` for those rows starts at epoch 1391 / 1786 / 81 and
`cost_train.json` counts only post-resume steps. The checkpoints settle it --
every `last.pt` sits at the configured final epoch -- so **every 2D row ran its
full epoch budget**. The two reruns launched on the bad finding were cancelled.

The audit did surface a real defect, in the opposite direction:

**Kolmogorov Geo-FNO trained 200k optimizer steps, 4x the ~50k every other row
received** -- and it is the row that wins that regime at 0.385. The budget was
fixed in *epochs* (2500) at batch 128 = 20 steps/epoch; Geo-FNO inherited 2500
epochs at batch 32 = 80 steps/epoch. Every other row across both fleets lands
within 4% of ~50k, including the cylinder Geo-FNO (49.4k), so this is specific
to one config.

Actions: the Kolmogorov section and the fairness section now report this
instead, with 0.385 stated as an upper bound; a budget-matched rerun (625
epochs = 50,000 steps, separate save root, canonical run untouched) is training
as job 3114517 with its eval chained (3114518). If the matched row still leads,
the headline survives with a corrected number; if it does not, the Kolmogorov
leaderboard changes and the paper says so.

`src/audit_train_budgets.py` now does this check reproducibly and labels each
number `instrumented` / `checkpoint` / `inferred`, because mixing those
silently is precisely what produced the wrong claim.


## 15. Seed-variance analysis ready ahead of the replicate evals (15:55)

`src/seed_variance_2d.py` is written and dry-run. It answers the question P1-6
exists for: is the gap between two methods larger than the spread from
retraining the same method with a different seed? It reports mean +/- std over
seeds 42/7/1337 per method, then walks the canonical ranking and flags any
adjacent pair whose gap is smaller than the pooled seed spread -- those pairs
are unresolved at one run per method whatever the point estimates say. It runs
the moment the replicate evals land.

The dry run caught a bug in my own script worth recording, because it is the
same class as the K16-vs-K8 trap elsewhere: a plain `sorted()` over
`kolm_fleet_dmfgen_K*.json` returns "K16" before "K8", so the script silently
reported the ensemble-size-sweep value (0.4804) instead of the canonical K=8
row (0.4874). Fixed by excluding the sweep files and preferring K8/K1
explicitly. `make_2d_tables.py` and `paired_ci_2d.py` already guarded against
this; `seed_variance_2d.py` did not.


## 16. Audit tool hardened; state at 16:05

Three more fixes to `src/audit_train_budgets.py`, all the same family as the
errors that already caught me once:

* a shadowing bug (the print loop reused the name holding the argparse
  namespace, so the second dataset crashed);
* `--fast` reproduced my original WRONG numbers complete with "SHORT" flags,
  because it reads exactly the two sources that under-report resumed runs. It
  now prints a warning naming the trap and suppresses any verdict its sources
  cannot support -- only checkpoint-derived rows are ever flagged;
* a **fleet-reference column**: "did this row run its own config" (frac) and
  "did every row get the same budget" (vs fleet) are different questions, and
  Geo-FNO scored 1.00 on the first while being 4x the second, so the anomaly
  was invisible in the original view.

### Compute state

18 trainers still running (surface fleet, both seed-replicate fleets,
observe-u-only, the two patch-4 arms). Queued behind them: the six chained
fleet evals, the protocol-ablation arms, the 2D density sweep, the calibration
dumps, and the budget-matched Geo-FNO rerun with its eval. Balance ~665 of 750
GPU-h. The queue is saturated by this project's own trainers, which is why the
small analysis jobs are waiting.

### What each remaining paper TODO is waiting for

Nothing is stuck on a decision except the four items that are yours
(authorship, acknowledgments, artifact DOI, JHTDB redistribution terms). Two
are genuine dead ends in the current artifact set and are documented as such
in the paper: the CoNFiLD strict-2048 row lacks a protocol stamp, and four 3D
rows have no per-snapshot payload for the paired analysis. Everything else is
a job in the queue.


## 17. Protocol ablation completed -- and it narrows a headline claim (17:25)

The three missing arms ran (job 3114701, 5 min). Full picture, same checkpoints
and frames throughout:

| arm | DMF-Gen | SiT | Senseiver | IDW |
|---|---|---|---|---|
| honest (held-out trajectory, scattered) | 0.487 | 0.494 | 0.913 | 0.546 |
| leaky split (seen trajectory, scattered) | 0.483 | 0.491 | 0.873 | 0.537 |
| uniform lattice (held-out trajectory) | 0.367 | -- | -- | 0.452 |
| uniform + leaky split | 0.364 | -- | -- | -- |

**The pre-registered expectation failed.** We predicted Senseiver would
collapse from 0.913 toward its 0.17 training loss under the leaky split; it
moved to 0.873. The direction holds (4.4% gain vs 0.9% for DMF-Gen and 0.6%
for SiT -- the memorizer does gain several times more) but the size does not:
a few percent cannot reorder a fleet whose adjacent gaps are 0.1+.

**Sensor layout carries almost the entire gap**: 0.487 -> 0.367 for uniform
sensing (25%), with the leaky split adding essentially nothing on top (0.364).
IDW moves the same way (0.546 -> 0.452), which is the tell that it is
measurement geometry, not model behaviour.

The leakage section now says this, and the paper's claim is narrowed
accordingly: "protocol, not architecture" survives, but *within* protocol it
is the sensing layout that dominates and the split matters mainly as a
correctness requirement. I also added the reconciliation with the 3D shuffled
numbers (latent FM 0.14 vs 0.56), which are not contradicted: that shuffle puts
an evaluation frame adjacent to a training frame, while this 2D arm holds out
frames already four DNS steps apart. Leakage severity is a dose-response in
distance to the nearest training frame.


## 18. Surface-to-field written up; a missing-figures gap found (2026-09-10 02:35)

The P0.5 experiment now has its own subsection (`sec:surface`), a generated
table (`tab_surface_body.tex` from `src/figs_pof/make_surface_table.py`) and
the side-by-side gallery. Two of three pre-registered predictions failed and
the text says so; the classical floor that wins volume sensing (0.056) reaches
only 0.689 from the wall ring while Senseiver improves to 0.103.

**Gap found while wiring it up: the paper included NO reconstruction gallery
for any of the three in-scope regimes.** The text asserts "we display single
posterior samples on fixed, pre-registered frames" and names the gallery frame's
error range, but `recon_gallery_{cylinder,kolmogorov,jhu}.pdf` were never
`\includegraphics`'d -- only the FireBench/wing ones had been, and the
three-regime cut removed those. So the paper claimed to show something it did
not show. All three are now figures (`fig:qual-cyl`, `fig:qual-kolm`,
`fig:qual-jhu`), which also resolves the cross-reference the new surface
caption needs. main.pdf grew 336 KB -> 2.0 MB accordingly.

Verified in the final compilation pass rather than the first: 0 undefined
references, 0 undefined citations. (Early-pass warnings are normal and resolve;
checking them at pass 1 would have been misleading.)


## 19. Seed study complete on both fleets (2026-09-10 03:20)

All 7 methods x 3 seeds on both 2D regimes. Adding S3GM raised the pooled
spread on both (Kolmogorov 0.0028 -> 0.0041 over six->seven methods; cylinder
0.0057 -> 0.0081), because S3GM is the most seed-sensitive row in either
regime (0.0082 / 0.0163) -- consistent with its documented sampler fragility.

**This falsified a claim I had already written.** The paper said "every adjacent
gap in the cylinder ranking is at least five times that spread". On the full
seven-method fleet the tightest gap (latent FM vs DMF-Gen, 0.030) is 3.7x the
pooled spread, not 5x. Corrected, and the text now names the tightest pair
rather than quoting a floor that depended on which methods happened to have
finished. The conclusion (cylinder ordering is seed-robust) survives with a
smaller margin.

Lesson repeated from sec.17: quoting an aggregate computed over a partial fleet
invites exactly this. Where a number depends on which jobs have landed, the
text should name the binding case, not the summary statistic.


## 20. I corrupted a canonical table row, and how it was recovered (2026-09-10 13:01)

**What happened.** To get sensor-density sweep points for `perf_vs_sensors` I
added a `NOBS_OVERRIDE` env var to `src/eval_kolm_fleet.sh`. It changed the
sensor count but *not* the output filename. So `kolm_sweep_s3gm_n65`
(job 3122391) wrote its 65-sensor result into the same aggregate JSON that
holds the canonical 655-sensor Kolmogorov S3GM row, replacing
relL2 0.4978 / CRPS 0.2305 with 0.8974 / 0.4598.

**How it surfaced.** The job finished in 1:58. A full 50-snapshot S3GM eval
takes ~2.5 h on this hardware, so the runtime itself was the tell -- it had
resumed from per-snapshot files rather than computing anything. That is also
what made it recoverable.

**Blast radius.** One file: the `kolm_fleet_s3gm_K8*` aggregate. The 50
per-snapshot `crps_snap*.json` files are written under density-qualified names
and were untouched, so no measurement was actually lost -- only the roll-up
derived from them. `tab_kolm_body.tex` still carried the correct 0.498 / 0.231,
which is why the paper never showed a wrong number.

**Recovery.** Cancelled the two queued jobs that would have repeated the damage
(3122392, 3122393) plus 3122143; fixed the launcher to append `_ovr` to the
prefix whenever an override is active; re-ran the canonical density, which
re-aggregated the intact per-snapshot files. Verified: relL2 0.49777,
CRPS 0.23050 -- exact, not merely close, because nothing was recomputed. The
override sweep now lands in `kolm_fleet_ovr_s3gm_K8_nfe200.json`, confirming
the fix on the same run that would previously have clobbered.

**The actual defect** was not the missing prefix; it was adding a knob that
changes *what is measured* while leaving *where it is written* alone. Any
future sweep dimension has to be part of the output key. The canonical rows are
now the only artifacts written without a qualifier, which makes an accidental
overwrite of one visible as a name collision rather than a silent replacement.


## 21. The same bug, one level down: a mislabeled density (2026-09-10 14:10)

Section 20's fix stopped an override run from overwriting the canonical row.
It did not stop the override run from being **wrong**, and the next job proved
it.

`kolm_sweep_s3gm_n6554` (job 3125076) finished in 1m45s and produced a file
labelled `n_obs=6554` whose numbers were relL2 0.49777 / CRPS 0.23050 --- the
canonical 655-sensor values, bit for bit. Same tell as last time: 50 S3GM
snapshots take about 2.5 h, so a two-minute run had computed nothing.

**Cause.** The per-snapshot resume cache reads two filename spellings, the
density-qualified `crps_n<N>_snap<M>.json` and a legacy `crps_snap<M>.json`.
The legacy spelling records no density in its name, so a 6554-sensor sweep
found the canonical 655-sensor cache, accepted it, and re-aggregated it under
a 6554 label. The legacy fallback was something I added earlier this session to
fix a *different* resume-key mismatch; it fixed that one and opened this one.

**Two fixes, because there were two defects.**

1. `eval_kolm_ensemble.py` no longer trusts a filename for the density. Every
   cached per-snapshot record already stores the `n_obs` it was computed at;
   the resume path now reads that field and recomputes the snapshot on any
   mismatch or absence, logging `[resume] REJECT`. Filenames are a hint, the
   payload is the authority.
2. `eval_kolm_fleet.sh` now encodes the density in the output name
   (`kolm_fleet_ovr_n6554_...`) instead of a bare `_ovr` marker. The marker
   alone was one filename for every density, so consecutive override runs would
   have overwritten *each other* --- the identical defect as sec.20, one level
   down, which is why it deserved a real fix rather than a second marker.

**Blast radius.** One file, quarantined under
`Evaluation/QUARANTINE_2026-09-10_density_mislabel/` with a README rather than
deleted. It never reached a table or a figure: the density-sweep figure reads
`kolm_sweep_*` files and requires `protocol == kolm2d_matched_v1`, and S3GM's
sweep row was still rendering as "still running". No published number moved.

**The lesson I should have drawn the first time.** Sec.20 concluded that a
sweep dimension has to be part of the output key. That was right and I applied
it only to the output filename, when the same dimension was also part of the
*input* key --- the resume cache. A knob that changes what is measured has to
be threaded through every artifact keyed by the measurement, in both
directions. The four S3GM densities are re-queued (3125170-3) with the guard
active; they will now actually compute.


## 22. Session of 2026-09-10: what landed, what is queued, what needs Nick

**Paper TODOs 11 -> 7.** 26 pages, no undefined references or citations.

### Resolved with new results

**2D density sweep is now a first-class result.** Six of seven Kolmogorov rows
have all five densities {65,164,655,1965,6554}; S3GM is re-queued. Two findings
that a single-density table structurally cannot show:
* The case for learning is bounded on BOTH sides. At 65 sensors Geo-FNO (0.776)
  beats IDW (0.968); at 6554 plain IDW (0.208) beats every learned row (best
  DMF-Gen 0.258). The crossover is between 3% and 10% sensing -- i.e. the entire
  case for a learned reconstructor lives in the regime this benchmark targets.
* Geo-FNO and latent FM are NON-MONOTONIC, degrading past ~2000 sensors
  (Geo-FNO 0.332 -> 0.696). Both resample sensors onto a fixed representation.
  At the canonical 1% row Geo-FNO is the best learned method and nothing in
  that number warns that 10x the instrumentation halves its accuracy.

**Cylinder cross-Re breakdown.** Training is Re {60,100,150,200}, so Re 80 is
an interpolation in Re and Re 250 an extrapolation. Ranking is identical at
both; extrapolation costs 1.26-1.54x and costs the MOST ACCURATE rows most
(gappy POD 1.54x); and the never-observed pressure channel degrades faster than
the aggregate in 6 of 7 rows. That last point independently explains the
scalar-recalibration failure already reported in sec:calibration.

**2D spectral counterpart.** `spectra_kolmogorov.pdf` had existed since 8 Sep
and was never included, so sec:spectra argued its band-split thesis from 3D
alone. Now shown, with Geo-FNO and MLP-RBF backfilled. Deterministic rows
retain 0.06-0.08 of DNS inertial energy; generative/operator rows retain
0.34-0.81 but buy it with spurious tail energy at rates differing >10x
(SiT 119x, Geo-FNO 60x, DMF-Gen 8.0x). Two orderings contradict the accuracy
table, which is the point.

**Operator axis, in scope for the first time.** sec:sensors described noise /
slab occlusion / channel dropout as part of the benchmark; they had only been
measured on FireBench, which the three-regime cut removed. Now implemented and
launched on Kolmogorov.

### Two integrity defects I introduced and closed

Both are the same root cause: **a knob that changes what is measured must be
part of every key derived from the measurement.** See sec.20 and sec.21. The
third instance was caught by a smoke gate before any fleet compute (the
operator reached the cache key and protocol stamp but not the aggregate output
name, so three operator runs overwrote one file). The gate now asserts distinct
outputs rather than trusting they exist.

The guard that actually protects the density figure is **protocol equality, not
the filename glob** -- `kolm_fleet*s3gm_K*.json` matches operator and override
runs too, and they are rejected because they are stamped
`kolm2d_matched_v1_op_<tag>`. Worth knowing before anyone "tidies" that filter.

### Queued (all behind fair-share)
* `lfm_calib` array x3 -- latent-FM calibration dumps, makes the conformal
  comparison two-sided (P1 #5).
* `s3gm_n{65,164,1965,6554}` -- completes the density figure.
* `op_{noise_0.1,noise_0.3,occlusion_0.25}_{fast,s3gm}` -- the operator matrix.

### Needs Nick
1. **BLOCKING, provenance.** The 3D leakage pair (0.14 shuffled / 0.56 honest)
   cannot be sourced from anything on this host. The archived pre-fix
   evaluations were never transferred, and all three latent-FM JHU evaluations
   here aggregate to 0.468-0.469. The `0.56` is that row's COVERAGE in tab:jhu,
   not its rel-L2 -- so the honest half of the pair contradicts our own table.
   Either send the archived shuffled-split JSONs, or decide to narrow the claim
   to the 2D protocol ablation, which IS measured here. I did not substitute
   0.469: that would mix a current run into a pair whose other half is archival.
2. Authorship + affiliations; acknowledgments; artifact DOI; JHTDB
   redistribution terms.
3. **Optional, ~25-30 GPU-h.** JHU seed replicates for the canonical DMF-Gen
   row. The existing `reg_s*`/`sup_s*` runs are NOT replicates of it (3000 vs
   6000 epochs, no spectral prior, different dropout), so the seed error bar is
   2D-only and the paper now says so. A true replicate is ~12 h of training
   each. Held rather than launched, so it does not push the higher-value evals
   back in a fair-share queue -- say the word and it goes in.


## 23. Fourth instance, and the first one that destroyed data (2026-09-10 20:30)

Caught while checking an unrelated anomaly (S3GM's steep density curve): the
DMF-Gen **clean density sweep no longer existed**. The three operator fleet
runs had each written straight over it, and occlusion --- which ran last --- is
what survived.

**Cause.** The dmfgen branch of `eval_kolm_ensemble.py` hardcodes its output
names for `cond_source=points`:

    sensor_sweep_dmfgen_n<N>.json      # per density
    sensor_sweep_dmfgen.json           # combined roll-up

ignoring `--out-prefix` *and* the operator suffix, and the roll-up additionally
hardcoded `"protocol": "kolm2d_matched_v1"`. So every operator run wrote clean
filenames, and the roll-up claimed the clean protocol while holding
corrupted-sensor numbers. Every other row was fine: the baseline branch already
threaded both.

**Why this one was worse than sec.20/21.** Those were recoverable --- the
per-snapshot caches survived, so re-aggregating reproduced the canonical
numbers exactly. Resume is *disabled* for dmfgen, so there is no per-snapshot
cache, and the clean sweep had to be **recomputed** (job 3126408). First actual
data loss of the campaign.

**What saved it from reaching the paper.** The five per-density files were
stamped with the operator protocol, so `fig_perf_vs_sensors.py` rejected them
and DMF-Gen simply vanished from the panel rather than showing occluded numbers
as clean. The protocol stamp did its job. The roll-up was the dangerous
artifact --- clean stamp, occluded contents --- and it is quarantined under
`Evaluation/QUARANTINE_2026-09-10_dmfgen_sweep_overwrite/` with a README. The
five per-density files were **renamed** to `kolm_fleet_occl0.25_dmfgen_n*.json`
rather than deleted: they are valid occlusion results that were merely
misfiled.

**Fixes.** The legacy hardcoded name is now used only when the operator is
clean; the roll-up carries `PROTOCOL` and an `operator` block.

**The pattern, stated plainly.** Four instances now, all one root cause: a knob
that changes what is measured must be part of every key derived from the
measurement. I have fixed this per-site four times. The real defect is that
output naming is scattered across four branches of one function instead of
going through a single helper that cannot forget the operator, the density, or
the protocol. That refactor is the durable fix and is *not* done --- it is
recorded here as the outstanding item rather than claimed.

**Also verified while investigating** (all clean, no action needed): the seven
canonical 1% rows share identical sensor fingerprints, including the documented
`idx_sum=22128955` at snap 256, so `tab:kolm` is sound; and all 27 baseline
sweep files are still stamped `kolm2d_matched_v1`.
