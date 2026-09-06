# FireBench Operator-Matrix Frame Audit — 2026-09-05

**Question.** Paper Table 2 (main.tex L220–231: ours N18 vs latent-FM vs Senseiver
under clean / noise 0.1 / noise 0.3 / slab-25% / dropout-vw) was produced by SLURM
jobs 16726561 (timed out mid-N11) and 16764760 (completed; source of all baseline
cells). Ours' config uses `train_ratio 0.9`; the baseline configs use `0.75`. The
launcher passed the **same** `--snapshot-index` values to both sides. Were the
models scored on the same absolute frames?

**Answer: no.** The scored frame sets are disjoint (ours: abs frames 109–119,
baselines: 90–100), and the content differs materially (fire age differs by 38 s;
per-frame field variance differs by up to 1.7x). Details, evidence, and a matched
re-eval script below.

Reproduce every number here with:
`~/venvs/jhtdb_env/bin/python src/audit_fb_frames.py` (CPU-only, login node OK).

---

## (a) Exact split parameters per model (read from run artifacts, not memory)

| Model | Run dir | Source of record | train_ratio | split mode / gap |
|---|---|---|---|---|
| Ours (N18) | `Save_TrainedModel/firebench/pointcloud_ffm/iclr_firebench_v4_DemoN18_20260819_083221` | `args.json`: `train_ratio = 0.9`, `seed = 42`, `data = .../FireBench_u10u12_merged.h5` (matches `Save_config/config_iclr_firebench_v4.yaml` L27) | **0.9** | block / 10 |
| Latent-FM (LFM) | `Save_TrainedModel/firebench/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN35_20260823_234826` | `run_config.yaml` L15: `train_ratio: 0.75` (matches `Save_config/config_baseline_Gen_firebench.yaml` L33) | **0.75** | block / 10 |
| Senseiver (SEN) | `Save_TrainedModel/firebench/baseline_senseiver/Baseline_senseiver_Stage1_DemoN36_20260823_153857` | `run_config.yaml` L15: `train_ratio: 0.75` (matches `Save_config/config_baseline_Det_firebench.yaml` L36) | **0.75** | block / 10 |

`JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10` is exported by all three train launchers
(`src/train_iclr_firebench_v4.sh`, `src/train_baseline_lfm_firebench.sh`,
`src/train_baseline_det_firebench.sh`, each L15) and by both eval launchers
(`src/eval_firebench_ops.sh`, `src/eval_firebench_ops2.sh`). Same H5 for all:
`/projects/ammoniacomb/generative_reconstruction/firebench3d/FireBench_u10u12_merged.h5`,
120 frames, attrs `cases = u10/ramp0 [0:60] + u12/ramp0 [60:120]`, i.e. frames
0–59 are the u10 sim (t = 30–148 s) and frames 60–119 the u12 sim (t = 30–148 s).
`time_stride = 1` everywhere.

## (b) Val-index → absolute-frame maps, both boundary semantics

Both dataset classes (`src/helpers.py::TurbulentCombustionH5Dataset` and
`src/helpers_baseline.py::TurbulentCombustionH5Dataset`) use the identical block
logic: `train = all[:n_train]`, `val = all[n_train+gap:]`, gap frames discarded,
`n_train = max(1, N - n_val - gap)`. They differ between trees only in the val-size
boundary:

- **$MAIN (as of 2026-09-05, and at eval time): `int` semantics** —
  `n_val = max(1, int(N*(1-ratio)))` (helpers.py L239, helpers_baseline.py L159).
- **Worktree fix: `round` semantics** — `n_val = max(1, int(round(N*(1-ratio))))`
  (helpers.py L242, helpers_baseline.py L162). Not present in $MAIN.

With N=120, gap=10:

| Split | semantics | n_val | train | gap | val | val idx i → abs frame |
|---|---|---|---|---|---|---|
| ours 0.9 | int (as-run) | 11 | 0–98 | 99–108 | **109–119** | `109 + i` |
| ours 0.9 | round | 12 | 0–97 | 98–107 | 108–119 | `108 + i` |
| baseline 0.75 | int (as-run) | 30 | 0–79 | 80–89 | **90–119** | `90 + i` |
| baseline 0.75 | round | 30 | 0–79 | 80–89 | 90–119 | `90 + i` (identical — 0.25 is float-exact) |

**Which semantics did the original eval jobs run under? `int`, proven two ways:**

1. $MAIN's helpers files still carry `int()` today; the jobs ran from `$MAIN/src`
   on 2026-08-25 and the `round()` fix exists only in this worktree.
2. Runtime fingerprint: `ensemble_eval.py` draws
   `default_rng(seed=0).choice(len(val_dataset), 8, replace=False)`. The
   16764760 log records `matched snapshots: 3 9 10 5 2 1 0 4`.
   `choice(11)` reproduces exactly `[3 9 10 5 2 1 0 4]`; `choice(12)` (round
   semantics) gives `[4 10 11 6 8 2 0 3]`. Only n_val=11 (int) matches.

Note the `int`/`round` discrepancy only affects **ours'** split (0.1 is not
float-exact: `120*0.09999... = 11.999...` → int 11 vs round 12); the baselines'
0.75 split is identical under both, so the remap in (c)/deliverable 2 is
semantics-proof on the baseline side.

## (c) Frames actually scored in the operator-matrix jobs

Job 16726561 (`eval_firebench_ops.sh`) hit its 12 h wall during ours' N11 dropvw
pass; job 16764760 (`eval_firebench_ops2.sh`) finished N11 dropvw and ran **all**
baseline cells (log ends `ALL DONE`). Both used val indices
`SNAPS = 3 9 10 5 2 1 0 4` (chosen by ensemble_eval on N18's val split, read back
from `ops_clean_K8.json`, logged as `matched snapshots:`). Independent
confirmation: the baseline run dirs contain `offline_eval_val_0000..0005, 0009,
0010` output dirs timestamped inside the 16764760 window (2026-08-25 06:27–06:46).

Same val index, different absolute frame:

| val idx | ours abs frame | ours u12 t [s] | baseline abs frame | baseline u12 t [s] |
|---|---|---|---|---|
| 0 | 109 | 128 | 90 | 90 |
| 1 | 110 | 130 | 91 | 92 |
| 2 | 111 | 132 | 92 | 94 |
| 3 | 112 | 134 | 93 | 96 |
| 4 | 113 | 136 | 94 | 98 |
| 5 | 114 | 138 | 95 | 100 |
| 9 | 118 | 146 | 99 | 108 |
| 10 | 119 | 148 | 100 | 110 |

## (d) Overlap / contamination

- **Scored-frame intersection: NONE.** Ours {109–114, 118, 119}; baselines
  {90–95, 99, 100}.
- **Subset check (as claimed): TRUE.** Ours' val (109–119) is a subset of the
  baselines' val (90–119), so a frame-matched baseline re-eval needs no
  retraining — remap `baseline --snapshot-index = abs_frame - 90`.
- **Baseline-scored frames inside OURS' train block: 90–95** (6 of 8); frames
  99–100 fall in ours' gap. **Ours-scored frames inside baselines' train: none.**
  Important nuance: this is *not* leakage into any reported number — each model
  was scored strictly outside its own training+gap region. The defect is frame
  identity, not train/test contamination.
- Accidental symmetry: both models were scored 11–21 frames past their own train
  end (identical distances), so *extrapolation distance from own training data*
  was matched; *physical content* was not.

## (e) Severity: how wrong can Table 2 be?

Same simulation (u12), same decorrelated-tail construction — but **not** the same
distribution of content. The fire is substantially more developed at ours' frames:

- Per-frame field std (every-64th-point subsample), mean over scored frames,
  ratio ours-frames / baseline-frames: **T 1.71x, CO 1.46x, U_1 1.30x**,
  CH4 1.03x, p 0.97x. Fire age differs by 36–38 s on a 118 s simulation.

Per-frame metric variability from the archived outputs (N18
`ops_*_K8.json` per-snapshot entries; LFM/SEN `ops_*_snap*.json` aggregates):

| op | ours relL2 mean±std (slope/idx) | LFM (slope/idx) | SEN (slope/idx) |
|---|---|---|---|
| clean | 0.456 ± 0.010 (+0.0028) | 0.724 ± 0.015 (−0.0040) | 0.779 ± 0.019 (+0.0057) |
| noise01 | 0.458 ± 0.010 (+0.0028) | 0.722 ± 0.018 (−0.0037) | 0.779 ± 0.020 (+0.0057) |
| noise03 | 0.461 ± 0.010 (+0.0028) | 0.724 ± 0.016 (−0.0036) | 0.779 ± 0.020 (+0.0058) |
| slab25 | 0.511 ± 0.020 (+0.0024) | 0.720 ± 0.016 (−0.0023) | 0.780 ± 0.019 (+0.0054) |
| dropvw | 0.603 ± 0.020 (+0.0056) | 0.719 ± 0.007 (+0.0013) | 0.908 ± 0.005 (+0.0014) |

Within-tail per-frame spread is 1.5–13% of the mean, i.e. cell std 0.005–0.020.
But the frame trends matter more: the matched frames sit ~19 val indices later
than the frames the baselines were scored on. Crude linear extrapolation of the
observed slopes over Δidx = 19 (unreliable — 2x beyond the observed window, but
the best available bound) suggests baseline cells could shift by roughly
**−0.08 (LFM, most ops) to +0.11 (SEN, most ops)** on the matched frames.

Consequences:

1. **Ordering (ours best in every cell): almost certainly robust.** The smallest
   current margin outside dropvw is ~0.21 relL2 (~10–20 per-frame sigma); even
   adverse extrapolation leaves ≥ 0.14.
2. **Two specific claims are at risk:**
   - *LFM dropvw* margin is 0.116 (0.603 vs 0.719). Applying the most
     LFM-favorable slope seen anywhere (−0.0040/idx) over Δ19 would put LFM at
     ~0.64, shrinking the margin to ~0.04 ≈ 2 per-frame sigma. Probably still a
     win, but no longer comfortably.
   - The intro headline "*its worst case (0.60) beats the best baseline's clean
     case (0.72)*" (main.tex L70): matched-frame LFM-clean could plausibly land
     near 0.65, leaving a ~0.05 margin — direction survives, "beats" becomes thin.
3. **Exact cell values: wrong at the 0.01–0.11 level.** Any baseline number
   quoted to three digits from these jobs is not defensible as a matched
   comparison.

**Verdict: MODERATE-HIGH.** Not same-distribution-indistinguishable (content
differs materially and baseline metrics demonstrably trend with frame index), but
not ordering-flipping either. The table's qualitative story survives; its printed
baseline numbers do not.

**Independent pre-existing problem:** both source jobs are already quarantined
(headers of `src/eval_firebench_ops.sh` / `ops2.sh`, baseline audit 2026-08-29)
for using the **legacy non-canonical sensor draw**. Table 2's baseline cells
therefore have two strikes: legacy sensor protocol AND frame mismatch. The
matched re-eval below deliberately keeps the legacy protocol so that only the
frame identity changes relative to 16726561/16764760 (ours' cells are not rerun);
migrating the whole matrix to the canonical draw is a separate job and would also
touch ours' cells.

## Matched re-eval

`src/eval_firebench_ops_matched.sh` (this worktree; wall 3:30, gpu-h100/f2pde;
the original 80 baseline evals took ~20 min on one H100). It rescores only the
5-op x 2-baseline matrix on ours' absolute frames via
`--snapshot-index = abs_frame − 90` ∈ {19, 20, 21, 22, 23, 24, 28, 29}, keeps
`ENSEMBLE_OP_SEED = 1000 + ours_val_idx` so each baseline sees the operator
realization ours saw on that frame, and writes
`ops_<op>_matched_snap<ours_idx>.json` beside (not over) the legacy files.
The baseline eval CLI accepts arbitrary per-run val indices
(`evaluate_Gen/Det_Baseline.py --snapshot-index`, consumed as `dataset[idx]` on
the 30-frame val split — indices 19–29 verified in range). Not submitted;
parent session to submit.

## Recommended wording (conservative)

Severity is not "low", so do **not** ship Table 2 as-is on a bare footnote.
Recommended course:

1. **Hold the baseline columns of Table 2 (and the L70 headline number) until
   `eval_firebench_ops_matched.sh` lands.** The re-eval is ~20 GPU-minutes; there
   is no schedule argument for printing known-mismatched numbers.
2. If a draft must circulate before the re-eval, the honest footnote is:
   "Baseline cells were evaluated on validation frames drawn from an earlier
   window of the same held-out tail (u12, t = 90–110 s) than our model
   (t = 128–148 s) due to differing train/val ratios; a frame-matched
   re-evaluation is in progress and baseline values may shift by up to ~0.1
   relL2. Orderings are expected to be unaffected." Do not present the current
   baseline values without this caveat, and do not cite the 0.60-vs-0.72
   worst-case-beats-clean claim until re-checked on matched frames.
3. After the matched re-eval, rebuild the table from
   `ops_<op>_matched_snap*.json` and re-derive the L70 claim. Remember the
   legacy-sensor-draw quarantine still applies to the whole matrix (ours
   included) — frame-matching fixes comparability *within* the table, not the
   table's canonical-protocol status.

---
*Audit artifacts: `src/audit_fb_frames.py` (reproduces everything above),
`src/eval_firebench_ops_matched.sh` (re-eval, not submitted). Evidence files
referenced: `$MAIN/src/eval_firebench_ops_16764760.log` L35,
`$MAIN/src/slurm-16726561.out` (timeout), run-dir `args.json` / `run_config.yaml`
as cited in (a).*
