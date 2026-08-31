# Gap analysis — wing (§4.2) and FireBench (§4.3) — 2026-08-30

Per HANDOFF Priority 1.2: what the reframed sections need that does not exist
yet, and which baselines can honestly run there. Paper reframe itself is done
(see main.tex: experiments now ordered scaling → wing → FireBench → JHU;
JHU table/figures kept in main text per Nick's ruling).

## Inventory (what exists)

**Wing** (origin `Save_TrainedModel/wing/`): trained runs for pointcloud_ffm,
baseline_latent_fm, baseline_senseiver. Newest config `config_iclr_wing_v4.yaml`.
Eval: `src/evaluate_wing.py` (rel-L2/CRPS/cov90 per case, JSON out, K/taps/shear
/noise args), `src/qualitative_wing.py` (zcollapse-ok annotated),
`src/replot_wing.py`, `src/eval_wing_plots.sh`. Also scripted but training
status unknown from this checkout: SiT wing (`config_baseline_SiT_wing.yaml`,
`src/train_sit_wing.sh`), gappy POD wing (`src/baseline_gappy_pod_wing.py`,
`src/run_gappy_pod_wing.sh`). Paper figures qual_wing.pdf / uncertainty_wing.pdf
exist. **No CoNFiLD wing config exists** (only `config_baseline_CoNFiLD_xcube.yaml`).

**FireBench** (origin `Save_TrainedModel/firebench/`): pointcloud_ffm trained
(v5clean); latent_fm configs exist. Eval: `src/eval_firebench_ops.sh`/`ops2.sh`
(operator matrix), `src/qualitative_firebench.py` (audit-verified free of
z-collapse), `src/replot_firebench.py`. Paper already carries tab:ops,
tab:robustclean, tab:crossvar and qual_firebench.pdf with latent FM + Senseiver
numbers; `figures/uncertainty_firebench.pdf` exists but is NOT referenced in
main.tex (free asset).

## §4.2 wing — needed vs. existing

| Item | Status | Action |
|---|---|---|
| Wing baseline table (new `tab:wing`) | MISSING — section currently has ours-only prose numbers | Build from evaluate_wing.py JSONs: rows ours / Senseiver / latent FM / gappy POD (+SiT-point if trained); explicit "n/a without resampling" rows for Gen4Turb, S3GM, FNO3D |
| Senseiver wing row | Run trained; eval JSON status unknown | Run evaluate_wing-equivalent on origin (cheap eval) |
| Latent FM wing row | Run trained; **honesty label needed** — its ConvAE is grid-latent, so wing participation implies resampling; label the resampling in the row | Eval + caption label |
| CoNFiLD wing row | NO config/run. Point-native ⇒ fair and desirable per HANDOFF | Needs config + Stage1/2 training — new training, requires Nick's sign-off (2026-08-29 "final runs" ruling vs. 2026-08-30 compute-lifted campaign covers JHU arms only) |
| Gappy POD wing row | Script exists; run status unknown | Cheap CPU job |
| SiT-point wing row | Config+launcher exist; training status unknown | Check origin; if untrained, same sign-off question as CoNFiLD |
| Pathway ablation (global-only / local-only / importance-bias off) | MISSING; paper commits to it as hypothesis test | Eval-time ablation if Model.py flags permit (importance bias: zero `sensor_importance_scale` at eval — verify; pathway masking: check forward flags); else labelled retrains — sign-off |
| Tap-count sweep | MISSING | Cheap: evaluate_wing.py --n-taps sweep on existing ckpt |
| Slice gallery | qual_wing.pdf exists (1 case) | Extend qualitative_wing.py to a small gallery of held-out geometries |
| Wing rank-hist/coverage curve | uncertainty_wing.pdf has spread-reliability | Optional; low priority |

## §4.3 FireBench — needed vs. existing

| Item | Status | Action |
|---|---|---|
| tab:ops / tab:robustclean / tab:crossvar | EXIST in paper | **Verify provenance**: confirm which checkpoints/runs produced the LFM+Senseiver numbers (HANDOFF lists only pointcloud_ffm as trained in `Save_TrainedModel/firebench/`; the numbers likely predate the reorg). n=8 snapshots — consider n≥20 rerun of the matrix (cheap eval) |
| qual_firebench.pdf | EXISTS, z-collapse-clean path | Optionally regenerate for consistency with final ckpt |
| uncertainty_firebench.pdf | EXISTS, unreferenced | Reference from §4.3 or appendix (free win) |
| Voxel-diffusion row (route completeness) | MISSING | FireBench crop is 152×126×192 ⇒ grid methods CAN enter; a Gen4Turb-class row would need new training — sign-off required; alternative: state route coverage via the JHU section and skip |
| Further OOD operators (ball occlusion, out-of-range density, structured arrays) | MISSING (paper todo) | Eval-only on existing ckpt (measurement_ops supports occlusion kinds) — cheap |

## Honest-participation matrix (who can enter, and how)

- Point-native, fair on wing: ours, Senseiver, CoNFiLD-class, gappy POD
  (sensor-native), SiT-point (token-native).
- Grid-native, excluded on wing without resampling: Gen4Turb, S3GM, FNO3D,
  latent FM's ConvAE (participates only via resampling — keep the row but
  label it, per main.tex §4.2 text).
- FireBench: everyone can enter in principle (regular crop); present rows are
  LFM + Senseiver.

## Decision needed from Nick

New-training candidates (all outside the 2026-08-30 campaign list): CoNFiLD
wing, SiT-point wing (if untrained), FireBench voxel-diffusion row, pathway
ablation retrains (if eval-time masking is insufficient). Everything else
above is eval-only on existing checkpoints and within standing rules.
