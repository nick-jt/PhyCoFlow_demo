# Fleet audit — 2026-08-29 (full 10-agent pass)

Intended as §34 of BASELINE_AUDIT_2026-08-28.md (append there once the write guard is
lifted; kept as a standalone file on branch `eval-infra-audit` meanwhile).

One agent per baseline code, four questions each: upstream fidelity / fair JHU adaptation /
results (runtime, memory, error, spread, VISUALS) / eval-script quality. No training job was
stopped: no baseline showed training-loss pathology.

## Verdicts (one line each)

| Baseline | Fidelity | Fairness (params vs 6,506,253) | Key result finding |
|---|---|---|---|
| Senseiver | PASS (PE + budget fixed; clip-removal verified applied, clipped_frac 0.0) | −1.33% | Slow grind, not hard plateau (best ep6280); visuals = severe low-pass + Uy/p climatology (bottleneck story) |
| CoNFiLD C/F/P | PASS (verbatim upstream decoder; norm fix live in all 3 runs) | +1.8% / +1.7% / P waived (declared) | **C≡P stage-1 bit-identity is BY DESIGN** (stage-1 configs byte-identical; P differs only in stage-2 prior) = strongest replicate control; F's washed-out figures are the honest 384-bottleneck consequence |
| S3GM | PASS (verbatim port @2343293; z-as-time caveat stands) | −2.4% | **Sampler "divergence" fully attributed to old baked-in α=5/β=0.4** (run predates config fix; alpha_dps=5.0 stamped in its own metrics; error spikes exactly at sensors). Training healthy 0.69→0.012. All periodic metrics/figures from DemoN93/94 are divergence documentation ONLY — never quote Ux=2.2e6 as a model result. Real numbers: finalize 17038277 `--arm jhu_tuned`. |
| Latent FM | PASS (conditioning fix live via lfm_fixes monkey-patch, verified 3 ways) | +2.66% (AE+UNet) | Canonical eval driver now EXISTS (eval_latentfm_ensemble.py, fingerprint gate proven end-to-end); disclose ~6.3× step budget excess; visuals = mid-scale texture retained, dissipation range smoothed (the latent-methods claim, honestly present) |
| Gen4Turb | PASS (upstream code untouched; anneal arm = 1 labelled scheduler.step line) | −5.4% | Best ckpt moved to ep3081 (stale-premise obsolete); table confirmed ingesting STRICT-mask JSON; monitor figs genuine z=60 slices |
| SiT | PASS (transport lib byte-identical; declared Euler-32 deviation) | +1.01% | **Naming correction: the trained arm is SiT-point (pointnet tokenizer), NOT patchify** — exhibit is the token-budget/chunk-coherence wall (speckle, no seams). Val still descending — run continues. |
| FNO3D | PASS (SpectralConvOddSafe = upstream HEAD fix, verified 3.5e-7) | +3.4% | **p=1.506 anomaly resolved: NOT FNO-specific** — framework-wide unobserved-channel identifiability limit on a K=1 diagnostic (our N29 = same failure, p≈0.98–1.23 band); no Gibbs ringing; domain padding = fairness footnote only. Width-OOM claim (17004486) has NO preserved traceback — do not cite as memory-wall evidence; §18c citation needs correcting. |
| DeepONet | PASS (upstream-exact branch/trunk; set-encoder = declared adaptation) | −0.008% | Visuals harsher than expected: near-CONSTANT predictions (lowest mode only) — the pre-registered structural-bottleneck finding, not a defect to tune away |
| Classical | PASS except **FLOOR BUG: on-disk JSON used periodic KD-tree/IDW (wrong)** | n/a | Corrected non-periodic re-run = job 17040032 (same paths/tags; periodic originals preserved as *_PERIODIC_WRONG.json). Non-periodic IDW Ux ≈0.171 BEATS ours (0.177): **"beats interpolation" claim must be re-scoped to NN, or IDW included honestly.** |
| Eval infra | — | — | All 7 canonical drivers now hard-abort on login node + fingerprint mismatch; 10 legacy launchers gated behind ALLOW_LEGACY_EVAL=1; 2 GENUINE z-collapse paths found & fixed (helpers.save_smooth_mask_plot, evaluate_coherence.save_worst_direction_spatial_map) — likely the collapsed figures Nick saw; qualitative_firebench clean. |

## Armed/launched jobs (all account f2pde, compute nodes)

- 17038277 s3gm_final (afterok:17014298) — jhu_tuned + upstream arms, final-ckpt stability sweep
- 17039914 g4t_anwin (afterany:16994805) — Gen4Turb window scan + canonical headline evals
- 17039932 fno3d_eval (afterany:17001807) — best+last × NFE 4/16
- 17039980 dpn_eval_bl (afterany:17017842) — best+last, shell-level fingerprint gate deletes bad JSONs
- 17040014 dpn_step48k (RUNNING watcher) — snapshots step48k_matched.pt at 48k steps
- 17040017/18/19 cnf_eval{C,F,P} (afterany per arm) — best+last, canonical, md5-logged
- 17040032 classical_np (RUNNING) — non-periodic floors + sweep + patched figures
- 17040070 lfm_canon (afterany:16997534) — best/last/window ckpt, K=8 NFE=4
- Pre-existing verified armed: 17001659/60 (Senseiver best+budget), 17002675/76 (SiT best+last arrays), 17003876 (ckptvar eval), 17014342 (s3gm arch)
- "Stalled" jobs adjudicated healthy (CPU pegged, outputs appearing): 17023433 calib_hi16, 17037431 eval_k32

## Paper-facing consequences collected

1. Classical floors: KD-tree 0.290→~0.208, IDW 0.235→~0.171 (Ux). Re-scope "beats
   interpolation" after 17040032 lands.
2. Unobserved-channel (Uy,p) errors ≥1.0 are a FRAMEWORK-WIDE identifiability result at K=1
   diagnostics — report from canonical K=8 evals only, and as a shared limit, not per-method
   failure.
3. SiT arm must be renamed SiT-point; exhibit = token-budget wall (speckle figures ep3200).
4. Budget disclosures per row: Senseiver ~69k steps, S3GM ~105k, latent FM ~209k(+95k stage1),
   Gen4Turb 105k, SiT 38.5k (wall-matched), DeepONet ~81k (48k ckpt captured by watcher),
   FNO3D ~80k — all ≥ our 48k except SiT (wall-matched arm); state per audit §16.
5. C≡P bit-identity = replicate control; state explicitly in the CoNFiLD row caption.
6. "log-uniform 0.1–1%" wording is wrong — the train draw is UNIFORM on integers (all models).
7. Sensor-fingerprint gate + compute-node abort now enforced in every canonical driver
   (was print-only or absent everywhere, including ensemble_eval.py itself).
8. Table hazards fixed/flagged: dead lfm_shipped glob removed; DeepONet smoke-JSON and
   budget-vs-best sort hazards; rows should require n=50 snapshots before the paper table is cut.
9. Gen4Turb tracked-file hygiene (Par.pkl, models/best_model.pt overwritten in upstream clone)
   — restore proposal recorded, do NOT git-checkout without preserving the scored artifact.
10. Two z-collapse fixes must reach main before ANY figure regeneration
    (helpers.save_smooth_mask_plot, evaluate_coherence.save_worst_direction_spatial_map).

## How to apply the code edits (write guard blocked direct edits)

Everything lives on branch `eval-infra-audit`
(worktree `.claude/worktrees/eval-infra-audit/`):
- 22 infra files (7 hardened drivers, 10 gated legacy launchers, table+footnotes,
  z-collapse fixes, guard silencer) — consolidated diff vs main's current uncommitted
  state: `eval_infra_audit.patch` at the worktree root.
- 2 latent FM files (eval_latentfm_ensemble.py, eval_latentfm_canonical.sh); execution
  copies in ~/.claude/jobs/3ac3fd02/tmp/ are what armed job 17040070 actually runs.
- CAUTION: `confild_eval_unified.py` was ALSO edited directly in the main checkout by the
  CoNFiLD agent (backup: confild_eval_unified.py.orig_20260829) — re-diff before copying
  the worktree version over it.
- Remaining un-applied micro-patches (proposed, small): ensemble_eval.py `"backbone"` stamp
  in the JSON payload; eval_senseiver_iclr.py hard-fail on CPU fallback (superseded by the
  infra guard if merged); baseline_classical_jhu.py default flip to non-periodic + figs
  boxsize=None (the re-run used explicit flags, so numbers are fixed regardless).

## RULING 2026-08-29 (Nick): these are the FINAL baseline runs

Compute on this HPC is about to run out. Standing consequences:
- NO new training launches, NO relaunches, NO re-run-based proposals (FNO domain-padding A/B,
  width-probe re-run, Senseiver 48k-checkpoint extra arm, LFM window sweep beyond the armed
  job) — all DEFERRED indefinitely; the paper is written from what the current runs +
  already-armed chained evals produce.
- The armed canonical evals ARE the paper numbers; if one fails it gets one cheap retry on
  the existing checkpoint, nothing more.
- Budget rows are reported exactly as run (steps/wall as logged); no equalization re-runs.
