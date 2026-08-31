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
