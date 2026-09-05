# HANDOFF — DMFGen-3D / ICLR 2027 (updated 2026-08-30)

Paper: `Paper/iclr2027/main.tex` (9-page limit excl. refs; last known to overrun to p.11 —
the Priority-1 reframe below is also the pruning opportunity). Deadline ~Sept 18.
This file is the entry point; the authoritative history lives in, reading order:
`BASELINE_AUDIT_2026-08-28.md` → `FLEET_AUDIT_2026-08-29.md` → `PLAN_IMPROVE_2026-08-30.md`
→ `FLEET_SUMMARY_TABLE_2026-08-30.md`. Do not contradict settled findings without new
evidence. The previous HANDOFF (2026-08-28, full code map and history) is in git history
at commit e5ee749 — consult it for architecture/dataset details not repeated here.

Thesis (unchanged, non-negotiable attributes): 3D, ambient (non-latent), end-to-end (no AE),
function space, generative — *ambient pointwise conditional generation removes the 3D
discretization bottleneck of generative field reconstruction, without latent compression.*

Change the name from LOCUS to DMF-Gen-3D

## Where the paper stands (one paragraph)

All 10 baselines are trained, upstream-audited, and canonically evaluated (n=50, fingerprint
snap=29 sensors=39062 idx_sum=37987162596) on the JHU cross-cube protocol. Honest outcome:
our model no longer leads any single JHU error column (SiT-point wins observed channels,
repaired latent FM wins aggregate/unobserved, non-periodic IDW edges Ux). What we own:
few-NFE ambient sampling (2–4 steps vs 32–1000), no-AE/no-voxel scale, member-level
spectral fidelity, honest cost numbers, and a fully characterized (not calibrated)
uncertainty with a known mechanism (~0.09 spread floor, density-invariant distance
profile). An improvement campaign (Senseiver capacity, DeepONet++, CoNFiLD fixes, S3GM
normalized guidance, LDW-FFM blend) is finishing on the origin HPC — results land in
`Save_TrainedModel/` (NOT in git; exists only on that cluster).

## PRIORITY 1 — Recenter the paper on the regime we own (writing + wing/FireBench results)

The JHU cube is our weakest battlefield: a periodic regular grid is exactly where voxel and
patch methods thrive. The paper's actual empty cell is ambient generative sampling on
scattered points where no grid exists. Concretely:

1. **Restructure the narrative** so JHU is the controlled fairness benchmark and the
   generality results carry the claim: SHIFT-WING (unstructured meshes, surface-only
   sensing) and FireBench (realistic operators, noise/occlusion). The asymmetry IS the
   argument: SiT/latent-FM/Gen4Turb/S3GM cannot enter these experiments without
   voxelization/resampling artifacts — state this explicitly and show what we do there.
2. **Assess what wing/FireBench results already exist**: `Save_TrainedModel/wing/`
   (pointcloud_ffm + baseline_latent_fm + baseline_senseiver runs) and
   `Save_TrainedModel/firebench/` (pointcloud_ffm; latent_fm configs exist). Newest configs:
   `config_iclr_wing_v4.yaml`, `config_iclr_firebench_v5clean.yaml`. Eval machinery:
   `src/evaluate_wing.py`, `src/qualitative_wing.py`, `src/qualitative_firebench.py`
   (FireBench figure path verified clean of z-collapse; wing annotated zcollapse-ok).
   Gap analysis first: which figures/tables do sections 4.2/4.3 need that don't exist yet,
   and which baselines can honestly run there (Senseiver and CoNFiLD are point-native and
   fair on the wing; grid methods get an honest "not applicable without resampling" row).
3. **NICK'S RULING (2026-08-30): do NOT remove the JHU material from the main text yet.**
   Whether JHU detail moves to the SI is decided later. For now: reframe emphasis and
   section ordering only; the JHU table and figures stay in the text.

## PRIORITY 3 — Post-hoc uncertainty recalibration (cheap, high value)

Upgrade the calibration chapter from "characterized, not calibrated" to "characterized and
repaired." Mechanism is known (audit + calib sweeps): spread is a density-invariant
function of distance-to-sensor with a hard ~0.09 floor; NFE rescales the whole profile
~1.5×; no (density, NFE) point is calibrated (sp/err runs 0.44→1.56 across densities).

Two routes, in cost order:
1. **Scalar / per-density spread rescaling**: fit a per-density (optionally per-channel)
   multiplier on the TUNE split (cube-3 ODD indices) to hit sp/err=1 or cov90=0.9; evaluate
   frozen on the TEST split (EVEN). Fittable from the existing
   `pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*/Evaluation/calib_sweep_*.json`
   per-snapshot values — no new sampling.
2. **Conformalized calibration** (the stronger claim): per-point conformal quantile scaling,
   weighted by distance-to-nearest-sensor (which we know is the spread's sufficient
   statistic). Needs per-point ensembles — the saved JSONs are summaries, so this needs ONE
   cheap eval job to dump per-point ensemble data at 2–3 densities (generalize the
   fleet-figure `dump_fleet.py` pattern, which already does this for one snapshot, via
   `src/ensemble_eval.py` machinery). Fit on TUNE, report TEST coverage curves
   before/after.
Honest-reporting requirements: recalibration is post-hoc and stated as such; the floor
remains the residual; offer the same wrapper to the generative baselines (latent FM at
minimum) or state why not — a one-sided repair would be unfair.

## Secondary (mention-only; pick up after 1 and 3)

- **Pareto figure**: accuracy vs inference s/field vs train GPU-h (+ sensor-density axis).
  All numbers already in `FLEET_SUMMARY_TABLE_2026-08-30.md`; the LDW result supplies the
  punchline "generative pays off exactly where the problem is ill-posed" (we win 0.1–1%
  density, interpolation wins 10%).
- **Benchmark as contribution**: fingerprinted protocol + 10 audited baselines +
  per-channel reporting + the fleet-wide unobserved-channel identifiability result,
  released as an artifact.
- **Member-level physics metrics**: spectra/PDFs on single members (ensemble-mean L2
  flatters chunk-incoherent methods — SiT's 0.143 averages away speckle its samples carry).
- **Training-seed replicates**: 2–3 seeds of N29 (checkpoint noise closed at σ≤0.0005;
  seed noise is the open rigor item).

## Paper corrections already owed (apply during the reframe)

1. SiT arm is **SiT-point** (pointnet tokenizer), not patchify; exhibit = token-budget wall.
2. `main.tex:190` and `:426`: spread is NOT "flat in sensor distance" — it responds to
   distance but never sharpens below the ~0.09 floor.
3. Calibration claims must carry operating point (density/NFE/K/checkpoint); NFE 4→16
   moves sp/err +0.22 and K 8→32 moves cov90 +0.11 — both exceed the claimed +0.097 gap.
4. "Beats interpolation": corrected non-periodic IDW Ux 0.174 edges our 0.177 — re-scope
   to NN or include IDW honestly.
5. Retrieval description: `Model.py` top-K is Euclidean KeOps with validity mask only;
   sensor-importance enters softmax logits only (`main.tex:123/:244` as written are false).
6. Quote only canonical n=50 JSONs — never K=1 training diagnostics.
7. All numbers per-channel primary; aggregate flagged; budget disclosures per row
   (steps/wall as logged; excesses noted in FLEET_SUMMARY footnotes).

## Operational state (origin HPC only — do not duplicate)

- Running/queued there (2026-08-30 evening): Senseiver-256 training (128/512 done,
  patience early-stop verified working), DeepONet++ p384 (p768/p640 early-stopped; p768
  canonical: Ux 0.512/Uz 0.541 vs vanilla 0.593/0.652, conditioning-responsive now),
  CoNFiLD Stage-B chain (codec plateau confirmed — continuation bought nothing), S3GM
  normalized-guidance final eval, LDW-FFM 10% pass. Chained evals write into
  `Save_TrainedModel/JHU/...`; the origin session commits/pushes table+figure updates as
  they land — `git pull` before finalizing any number in the text.
- Fleet figure: `Save_TrainedModel/JHU/fleet_figure/fleet_reconstruction_snap29.{png,pdf}`
  (+ per-method npz bundle). Weekly deck: `weekly_update_2026-08-30.pptx`.
- Standing rules: canonical fingerprint + compute-node-only evals; TUNE=odd/TEST=even
  split discipline for anything fitted; never z-collapse 3-D fields
  (`src/check_no_zcollapse.py` must exit 0); upstream-faithful baseline rows frozen —
  improvements are labelled arms; no heavy compute on login nodes.

## Progress log — 2026-08-30 (Engaging session)

DONE (this checkout, committed):
- **P1 reframe applied to `Paper/iclr2027/main.tex`**: experiments reordered
  scaling → wing → FireBench → JHU; JHU recast as the controlled fairness
  benchmark (table + figures kept in main text per Nick's ruling); wing promoted
  to its own subsection with the explicit inadmissibility ("n/a without
  resampling") framing; abstract, contribution bullets, and conclusion rewritten
  to fleet-honest claims. LOCUS → DMF-Gen-3D (single macro).
- **JHU table replaced** with the canonical FLEET_SUMMARY numbers, per-channel
  primary, all fleet rows incl. classical floors; S3GM row todo'd pending the
  finalizing eval. Old paired-CI/TOST text (computed vs. pre-repair latent FM)
  removed; recompute todo'd against canonical n=50 JSONs.
- **All 7 owed corrections applied**: SiT-point naming + token-wall appendix ¶;
  spread-vs-distance floor text (main + fig caption); operating-point discipline
  in Metrics ¶; "beats interpolation" re-scoped (IDW 0.174 vs 0.176 stated
  honestly); retrieval description rewritten to match Model.py (Euclidean k-NN
  selection, importance bias in softmax logits only — verified in code,
  `Model.py` `_knn_search_keops` / logit bias at ~1409); log-uniform→uniform-on-
  integers wording; budget disclosures per row + appendix baseline paragraphs
  (LFM repair+6.3× budget, CoNFiLD C≡P control, FNO3D, DeepONet, S3GM, classical).
- **P1.2 gap analysis**: `GAP_ANALYSIS_WING_FIREBENCH_2026-08-30.md` — needed
  vs. existing per section, honest-participation matrix, and the four
  new-training candidates that need Nick's sign-off (CoNFiLD wing, SiT wing,
  FireBench voxel row, pathway-ablation retrains).
- **P3 machinery written + smoke-tested on synthetic data**:
  `src/recalibrate_spread.py` (route 1: fits per-density/channel spread
  multiplier on TUNE odd from existing calib_sweep/canonical JSONs — works on
  latent-FM payloads too, no new sampling), `src/dump_calib_points.py` +
  `src/dump_calib_points.sh` (route 2: ONE eval job, 3 densities × 50 snaps,
  fingerprint-gated, per-point npz), `src/conformal_recalib.py` (distance-binned
  split-conformal quantiles fit on TUNE, TEST coverage before/after; synthetic
  test repaired 0.45→0.90 cov90). Paper carries the recalibration subsection
  (§ From characterized to repaired) with numbers todo'd.

## Progress log — 2026-08-31 (Engaging session): datasets landed, campaign launched HERE

Data arrived and was verified (`src/engaging/verify_datasets.py`):
- `Dataset/JHU_TurbulenceDataset.h5` (18.4 GB): **single-cutout, 617 CONSECUTIVE
  frames** of isotropic1024coarse (cube 125^3, start_ijk [228,51,563]) — this is
  the temporally-blocked same-region protocol dataset, NOT the 4-cube cross-cube
  file. Canonical cross-cube work (P3 canonical fits, fleet numbers) still lives
  on origin.
- FireBench u10 + u12 in `~/orcd/scratch/firebench3d/firebench3d/` (4.25 GB each):
  exactly the paper protocol ([1,60,3677184,1,1,5], 152x126x192, u,v,w,theta,rho_f).
  CAVEAT: the u12 file's `source` attr wrongly says "u10/ramp0" (stale label from
  the extraction script); the data itself is clearly the higher-wind case
  (mean u ~8.4 vs ~7.1 m/s). Flagged, proceeding on data.

Cluster facts (MIT Engaging): account `mit_general`; `mit_normal_gpu` (6h cap;
h200:8 x13 nodes, h100:4 x1, l40s x53), `mit_preemptable` (2d, preemptable).
All three trainers resume (--RELOAD/--reload), so trainings run as 6h afterany
chains (`src/engaging/submit_chain.sh`). Python: system anaconda 3.11 + user
site (torch 2.7.1+cu126 present; pykeops 2.3 pip-installed this session; jobs
`module load cuda/12.4.0` for the KeOps JIT). **Canonical-fingerprint caution:
the sensor draw is H100-SXM-bound; trainings are SKU-free but any canonical-
operating-point eval here must target the h100 node (gres=gpu:h100:1) — the
fingerprint gate will verify/abort either way.**

LAUNCHED (2026-08-31 ~00:4x, all account mit_general):
- 21631452 fb_merge (CPU): u10+u12 -> `~/orcd/scratch/firebench3d/FireBench_u10u12_merged.h5`
  via `src/engaging/merge_firebench_cases.py` (coordinate-equality checked, case
  order u10 then u12, attrs record case_n_t=[60,60]).
- 21631453-57 jhu_tmp_eng x5: **temporal same-region companion retrain** (paper
  pending item vi): spec02 architecture verbatim, block split JHU_SPLIT_GAP=100
  (363 train / 154 val; residual frame correlation ~0.67 at the gap — disclose),
  epochs 2500 ~= 48k steps (budget-matched to N29), labelled **DemoN33**,
  save_dir `Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_temporal_spec02`.
- 21631458-61 fb_v5c_eng x4 (afterok merge): ours on FireBench, v5clean config,
  DemoN31.
- 21631462-64 fb_bl lfm stage1 x3 (afterok merge); **stage 2 chain must be
  submitted after stage 1 completes** (BL=lfm LFM_STAGE=2).
- 21631465-67 fb_bl det (Senseiver) x3 (afterok merge).
Launchers in `src/engaging/`; per-job configs are sed-generated copies
(`Save_config/*_eng.yaml`) with only data/save_dir/epochs swapped.

NEXT here: watch first segments for import/config errors (monitor armed);
submit LFM stage 2 after stage 1; on training completion run the FireBench
operator matrix (n>=20) + temporal-companion eval; SHIFT-WING data arrives
later per Nick.

NEXT (origin HPC): sbatch `src/dump_calib_points.sh`; run
`recalibrate_spread.py` on existing `calib_sweep_*.json` (login-OK, JSON only);
fill recalibration todos; wing/FireBench evals per the gap doc; land S3GM +
improvement-arm rows. Open editorial items are the 24 \todo{}s in main.tex
(grep todo). Page budget not yet re-measured after the reframe (no LaTeX on
Engaging login node) — compile on origin before pruning decisions.

## Progress log — 2026-08-31 (evening): CoNFiLD stage-1 latent-dim sweep launched

Nick-approved sweep testing whether the C arm's ~0.72 held-out codec ceiling is
latent-capacity-bound (384→1024 precedent: oracle 0.535→0.441; Stage-B plateau
continuation bought nothing → capacity, not optimization). Stage 1 ONLY; stage 2
gated on 2K/4K materially beating 1K on the frozen-decoder oracle.

Four arms, everything except {latent_dim, hidden_features, image_size-relative
attention string} held to arm-C protocol (seed 42, layers 15, groups 48, batch 8,
16384 pts/item, lrs 1e-4/1e-5, train_ratio 0.75, 48600 s stage-1 budget):

| arm | H | D | decoder | latent table (train-only) | prior | infer total |
|---|---|---|---|---|---|---|
| sweep1024  | 256 | 1024 | 5,183,236 | 7,372,800 | 1,441,217 | 6,624,453 (+1.82%) |
| sweep2048  | 256 | 2048 | 9,377,540 | 14,745,600 | 1,441,217 | +66% — exploratory only, never→stage 2 |
| sweep4096  | 256 | 4096 | 17,766,148 | 29,491,200 | 1,441,217 | +195% — exploratory only |
| strict2048 | 144 | 2048 | 5,032,948 | 14,745,600 | 1,441,217 | **6,474,165 (−0.49%)** — the promotable arm |

Smoke test `src/check_confild_sweep_params.py` (job 21693407) PASSED all counts
to the digit + forward shapes + attention_ds=[32,64,128] in every arm (the
attention string is image_size-relative: "32,16,8"@1024 ≡ "64,32,16"@2048 ≡
"128,64,32"@4096 — an unscaled string at 2048+ silently loses ALL attention).

LAUNCHED — **superseded 2026-09-01, see "partition switch" below**: originally
h200/mit_normal_gpu, 3×6h chained segments/arm at wallclock_budget_s=16200
(jobs 21693711-22, evals 21693726-29). All 16 cancelled unstarted.

### Partition switch to mit_preemptable (2026-09-01)

After ~24 h nothing had started: all 13 h200 nodes mixed/allocated, our jobs at
priority 258,532 behind other work; estimates had slipped to Sep 3. The real
cost was not the first start but the **per-segment requeue** — a 3-segment chain
pays the (then ~1 day) queue wait three times. mit_preemptable (2-day limit)
starts sooner AND fits the whole 48600 s budget in ONE job, and preemption is
cheap because the trainer resumes from last.pt.

Budget correctness under preemption (the trap): `wallclock_budget_s` is measured
from PROCESS start (`confild_upstream_training.py:486`) and never accumulates
across resumes, so a naive resume would grant a fresh 48600 s and overshoot the
protocol. The launcher now sums wall-clock already consumed (per-process maxima
of `elapsed_seconds` in `history.jsonl`, which resets each process) and passes
only the remainder; it exits 0 when the budget is spent. Logic unit-tested
against synthetic 1/2/3-segment histories + empty + malformed-line cases.
Configs now carry the true total 48600 (not 16200).

SUPERSEDED AGAIN by the environment incident below (jobs 21744632-35 failed on
import in 29 s; relaunched as 21761937-40).

## Progress log — 2026-09-01: ENVIRONMENT INCIDENT — the whole Engaging campaign was broken

**Nothing in this campaign had ever executed on Engaging.** Everything sat
PENDING for two days, so three independent, campaign-fatal defects went unseen
until the preemptable switch finally got a job onto a node. ALL 28 main-campaign
jobs and all 4 sweep arms would have produced nothing.

| # | Defect | Killed | Visible to |
|---|---|---|---|
| 1 | scipy 1.16.0 vs numpy 1.24.4 — `numpy.exceptions` (added in numpy 1.25) missing | every baseline trainer, on import | `pip check` (login-safe!) |
| 2 | `neuralop` never installed (`Model.py:10` imports it at top level) | every pointcloud trainer, on import | any import test |
| 3 | KeOps could not link `-lnvrtc` | every KeOps reduction, at RUNTIME | only a real GPU reduction |

Defect 1 was masked: `model_baseline.py:4112` wraps imports in
`try/except ImportError` whose fallback is a relative import, and `src/` has no
`__init__.py`, so the surfaced error was the misleading "attempted relative
import with no known parent package" — pointing at packaging, not at scipy.

Defect 3 is the instructive one. `ld` resolves `-l` flags via **LIBRARY_PATH**,
not LD_LIBRARY_PATH, and the `cuda/12.4.0` modulefile only prepends the latter.
`libnvrtc.so` was present the whole time, just never on the linker's search
path, so KeOps' JIT build failed, cached the failure, and every reduction raised
`OSError: nvrtc_jit.so: cannot open shared object file`. **Imports all pass** —
this is invisible to import tests and sits under `Model.py`'s kNN search, so
jobs would have staged 6 GB, initialised, then died.

FIX — dedicated venv `~/envs/phycoflow-env` + `source ~/envs/phycoflow`
(warp/warp-env convention; miniforge 25.11.0-0, py3.12.12, isolated so the
broken `~/.local` python3.11 packages cannot leak in). Pins torch 2.7.1+cu126,
numpy 1.26.4, scipy 1.17.1, pykeops 2.3, neuraloperator 2.0.0 — frozen in
`requirements_engaging.txt`. The activation script loads cuda ITSELF and exports
LIBRARY_PATH (an earlier version keyed off an already-set CUDA_HOME and silently
no-opped when sourced before cuda — a fix that quietly does nothing is worse
than none). All six `src/engaging/` launchers now source it.

GUARD — `src/engaging/check_env.sh`, two tiers:
- `bash check_env.sh` — LOGIN-SAFE: `pip check` + metadata, no imports. This
  catches defect 1 where we actually launch from. **Engaging login nodes cannot
  import numpy/torch at all** (the process is interrupted loading the compiled
  core; affects `~/envs/warp-env` too — a site restriction, not our env), so
  never conclude the env is broken from a login-node import failure.
- `sbatch check_env.sh` — imports of every campaign entry point, CUDA, and a
  real KeOps `argKmin` cross-checked against `torch.cdist` (catches defect 3).
Verified green 2026-09-01 (job 21761553, L40S): all imports, KeOps JIT OK, no
`-lnvrtc` errors. Run this BEFORE any relaunch.

**sbatch spools the script at submission**, so fixing a launcher does NOT reach
already-queued jobs — the same trap as the 2026-08-31 directory move. Every
affected job had to be cancelled and resubmitted.

### RELAUNCHED 2026-09-01 (all on the venv)

CoNFiLD sweep, mit_preemptable, one job per arm (`--time=14:00:00`, gpu:h200:1),
resume-aware budget so the 48600 s total survives any preemption; optional
`SEGMENT_CAP` caps a segment for 6 h partitions:
- sweep1024 21761937, sweep2048 21761938, sweep4096 21761939,
  strict2048 21761940; oracle evals 21762065/68/69/70 armed on each.

Main campaign, mit_normal_gpu, chains as originally designed:
- jhu_temporal DemoN33 21770405-09 (5) | fb_v5clean DemoN31 21770410-13 (4)
- fb LFM stage1 DemoN35 21770414-16 (3) | fb Senseiver DemoN36 21770417-20 (3)
- fb LFM stage2 21770421-23 (3, FIRST_DEP=afterany:21770416)
- seedrep 1379 21770424-28 (5) | seedrep 2718 21770429-33 (5)

### QOS ceilings (measured; these now bound the campaign, not the code)

`mit_normal_gpu gres/gpu=2` and `mit_preemptable gres/gpu=4` **per user**, and
the unrelated `v70d/v72d` arrays hold 3 preemptable slots. So the main campaign
progresses 2 GPUs at a time and the sweep contends with other work for the rest.
Plan schedules against this, not against queue depth.
- **If an arm is PREEMPTED, resubmit the identical command** — it resumes and
  finishes only the remaining budget. Monitor is armed and prints the exact
  resubmit line on preemption.
- Oracle evals armed on each job (afterany), kept on mit_normal_gpu because
  short 3 h jobs backfill well: 21744661-64. They now ALSO guard on budget
  consumed >= 48600-300 s and exit 4 rather than score an under-trained arm
  (afterany can fire on a preempted job; comparing arms at different spent
  budgets would silently corrupt the sweep).
  `src/engaging/eval_confild_stage1_engaging.sh` runs
  `evaluate_confild_stage1.py` (un-gated, SKU-independent) on last.pt (primary,
  budget-matched) + best.pt, snaps 150 151 153 162, settings identical across
  arms; JSONs at `<run>/Evaluation/stage1_oracle_{last,best}/stage1_auto_decode.json`.
- Upstream CoNFiLD checkout cloned to
  `/orcd/scratch/orcd/002/ntricard/baselines/CoNFiLD` @ 449835e (required; the
  `/projects/ammoniacomb` path does not exist on Engaging).
- Configs: `Save_config/config_baseline_CoNFiLD_xcube_{sweep1024,sweep2048,sweep4096,strict2048}.yaml`
  (per-arm save_roots under `Save_TrainedModel/JHU/baseline_confild/` — required,
  resume/stage-2 discovery globs per save_root). demo_num stays 23.
- Free corroboration signal: `heldout_codec_rel_l2_zscore` per figure tick in
  each run's `history.jsonl`.
- Engaging-vs-origin caveat: wall-clock budget on h200 buys more epochs/hour
  than origin h100 — within-sweep comparison is controlled (incl. the re-trained
  1024 arm); comparison to origin C numbers is indicative only.

## RESULTS — CoNFiLD capacity study (2026-09-02..05, Engaging)

Parameter budget was explicitly LIFTED for this study (Nick, 2026-09-02): the
±10% matched question was already answered (C = agg 0.871), so the question
became CoNFiLD's UNCONSTRAINED ceiling.

### 1. Stage-1 codec vs latent dimension (oracle, runtime-matched 48600 s)

| latent | decoder | oracle mean | max-channel |
|---|---|---|---|
| 1024 | 5,183,236 | 0.3333 | 0.3991 |
| 2048 | 9,377,540 | 0.2948 | 0.3599 |
| 4096 | 17,766,148 | 0.2702 | 0.3298 |
| 8192 | 34,543,364 | **0.2462** | 0.3024 |

Monotonic, ~8-9% per doubling, NO knee through 8192. Latent capacity is nearly
FREE in compute at stage 1: every arm trained at ~31-32 s/epoch, because the
FiLM projection is per ITEM (7,200) not per point (1.95M). Decoder WIDTH is
per-point and costs ~4x per doubling — latent is the efficient axis.

**Do not use the in-training `heldout_codec_rel_l2_zscore` proxy as a result.**
Its 400-step latent fit systematically UNDER-fits large latents and twice
inverted the ranking vs the 3000-step/3-restart oracle (it reported 4096 as
WORSE than 2048; the oracle shows it better). Trajectories only.

Context: the latent-FM ConvAE ceiling is 0.0296 (BASELINE_AUDIT item 9), so
CoNFiLD's best codec is still ~10x worse. That gap is its defining compressive
bottleneck, not a defect.

### 2. Stage-2 prior capacity — THE lever, and it SATURATES

Stage 1 held fixed (sweep2048); only prior num_channels varied. The UNet is
1-D conv, so size depends on num_channels/channel_mult, NOT model_image_size —
every earlier arm had unknowingly run the SAME 1,441,217-param prior while the
latent it models grew 8x.

| prior | params | steps | Ux | Uy* | Uz | p* | agg | CRPS | cov90 |
|---|---|---|---|---|---|---|---|---|---|
| ch32 | 1.4M | 630k | 0.688 | 1.030 | 0.727 | 2.818 | 1.316 | 0.782 | 0.391 |
| ch64 | 5.7M | 471k | 0.448 | 1.152 | 0.485 | 1.347 | 0.858 | 0.423 | 0.605 |
| ch128 | 22.9M | 204k | 0.434 | 1.151 | 0.474 | 1.138 | **0.799** | 0.404 | 0.583 |
| ch256 | 91.6M | 105k | 0.440 | 1.108 | 0.484 | 1.188 | 0.805 | 0.403 | 0.596 |
| P (fleet) | 118.9M | — | 0.409 | 0.751 | 0.374 | 1.007 | 0.635 | 0.371 | 0.57 |

1.4M -> 22.9M bought 0.517 aggregate; the next 4x bought NOTHING (0.799 ->
0.805, within noise). **CoNFiLD's ceiling here is not a parameter-count limit.**

**Paper-relevant claim:** ch256 (91.6M) still falls ~30% short of P (118.9M) at
comparable prior scale. C and P share a bit-identical stage 1 and differ ONLY
in the prior, so P's advantage is ARCHITECTURAL, not capacity. This is stronger
than the parameter-matched result alone and required the scale-up to establish.

### 3. What is actually broken: the unobserved channels

Uy sits at 1.03-1.15 across a 64x prior-capacity range and NEVER improves,
while p falls steadily (2.818 -> 1.138). Meanwhile the codec represents both
well (Uy 0.326 / p 0.301 at latent 1024). So the decoder can express them; the
DPS sampler cannot infer them from sparse Ux/Uz. Remaining untested lever is
the GUIDANCE (dps_scale, steps, conditioning method) — eval-time only, no
retraining. cov90 rising 0.391 -> ~0.59 with prior size says the small prior
was overconfident, not merely inaccurate.

### 4. Combined codec+prior arms (in flight)

sweep8192 (best codec) x ch128/ch256. NOTE a real confound: the prior is 1-D
conv over the latent sequence, so latent 8192 makes every diffusion step ~4x
more expensive than at 2048 — these arms get far fewer steps in the same
budget (tracking ~55k and ~25k vs 204k/105k). Quote step counts beside any
result; a weak outcome may be step starvation, not a failure to compound.
