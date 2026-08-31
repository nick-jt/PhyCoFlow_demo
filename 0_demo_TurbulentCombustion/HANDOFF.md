# HANDOFF — LOCUS / ICLR 2027 (updated 2026-08-30)

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
