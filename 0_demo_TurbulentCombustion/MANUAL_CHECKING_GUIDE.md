# Manual checking guide — PoF-2026 benchmark code

Prepared 2026-09-22 for a line-by-line audit of (A) every baseline against the
upstream repository it was taken from, (B) our fork against the PhyCoFlow
repository we forked from, and (C) every experiment YAML against its upstream
ancestor. Nothing in the repo was modified to produce this guide; the only
side effects are a read-only `upstream` git remote and the workspace below.

Repo under audit: `/work/hdd/bilr/ntricard/PhyCoFlow_demo` at `8aa01c9` (main, after the 2026-09-22 pull of the Engaging campaign; Sections A–C were written at `41203e0` and re-checked — the pull touched none of the source or config files they cite. Section D covers what the pull added).
Paths written `src/...` mean `0_demo_TurbulentCombustion/src/...`.

## 0. Workspace (already set up)

```
/work/hdd/bilr/ntricard/manual_check/
├── MANUAL_CHECKING_GUIDE.md          this file (sections 0, 1, A, B, C, D); GUIDE_A/B/C/D_*.md are the same sections as separate files
├── cmp.sh                            side-by-side helpers (below)
├── clone_upstream_baselines.sh       re-run to fetch fresh upstream clones
├── phycoflow_fork_point/             git worktree: upstream at the fork point 828c3c6
├── phycoflow_upstream_head/          git worktree: upstream/main f163ad7 (2026-08-17)
├── recovered_docs/                   BASELINE_AUDIT_2026-08-28.md, FLEET_AUDIT_2026-08-29.md, PLAN_IMPROVE_2026-08-30.md,
│                                     HANDOFF_pre_2026-09-08.md, README_upstream_at_fork_point.md (all deleted from the tree in 464c72e)
└── upstream_baselines/               clones at the commit our code records (= each repo's HEAD today)
    ├── senseiver/                    OrchardLANL/Senseiver            e443eb0 (2024-11-21)
    ├── confild/                      jx-wang-s-group/CoNFiLD          449835e (2025-02-19)
    ├── s3gm/                         lzy12301/S3GM                    2343293 (2025-04-09)
    ├── gen4turbulence/               vivekoommen/Gen4Turbulence       8af0a7b (2026-02-06)
    ├── sit/                          willisma/SiT                     cbde832 (2025-12-21)
    ├── neuraloperator/               neuraloperator/neuraloperator    00b7d86 (2026-08-06)
    ├── deepxde/                      lululxvi/deepxde                 99b6620 (2026-08-18)
    ├── deeponet_lu/                  lululxvi/deeponet                8d62345 (2022-06-25)
    ├── rectifiedflow/                gnobitab/RectifiedFlow           5a1fd4d (2024-06-20)
    ├── sparse_reconstruction_shu/    BaratiLab/Diffusion-based-Fluid-Super-resolution a802a4b (Shu et al. code)
    └── phycoflow_upstream/           cosmos2w/PhyCoFlow_demo, standalone clone at 828c3c6
```

To confirm the clones are current before you start:

```bash
cd /work/hdd/bilr/ntricard/manual_check && bash clone_upstream_baselines.sh
```

Every clone is checked out at the commit our code records (`PINS` in the
clone script, with the file:line where each was found). As of 2026-09-22
every one of those commits is also that repo's HEAD, so there is no upstream
drift to account for: a mismatch is ours. Also on disk:
`/work/hdd/bilr/ntricard/datasets/baselines/{CoNFiLD,Senseiver_OrchardLANL,Gen4Turbulence}`
are symlinks into `upstream_baselines/` so the hard-coded paths in
`src/confild_*.py` and `src/gen4turb_eval.py` resolve.

### `cmp.sh`

```bash
./cmp.sh hunks  src/Model.py              # every hunk of OUR change to a shared file, with line numbers
./cmp.sh fork   src/Model.py              # ours vs fork point (unified diff in less)
./cmp.sh uphead src/Model.py              # ours vs upstream/main HEAD
./cmp.sh upstream-delta src/Model.py      # what upstream changed after we forked
./cmp.sh files  <ours> <theirs>           # any two files, e.g. our senseiver adapter vs the clone
VIEW=vimdiff ./cmp.sh fork src/Model.py   # interactive; ours on the right
```

Equivalent raw git (from the repo root):

```bash
git fetch upstream
git merge-base HEAD upstream/main                      # -> 828c3c6
git diff 828c3c6 HEAD -- 0_demo_TurbulentCombustion/src/Model.py
git log --format='%h %ad %s' --date=short 828c3c6..HEAD -- <path>   # our commit trail for a file
git log -L<start>,<end>:<path> 828c3c6..HEAD             # history of a line range
git show 828c3c6:<path>                                  # the upstream file as merged
```

## 0.1 Suggested order for one evening

1. **Section B first** (fork diff, ~5 k diff lines over 14 files). It is the
   method we are benchmarking against ourselves (DMF-Gen = upstream
   PointCloudFFM), so every **BEHAVIOUR CHANGE** flag there is the highest-value
   check. `src/Model.py` and `src/helpers.py` carry most of it.
2. **Section A**, in this order of risk: CoNFiLD (three arms, capacity claim
   most likely to be challenged), S3GM (declared 3D deviation), Senseiver
   (grad-norm and loss-reduction question), DeepONet (set-encoder adaptation),
   Gen4Turb, SiT, Geo-FNO/FNO3D, classical anchors, LFM.
3. **Section C** last, reading only the tables for the regimes you care about
   and the 20 items in C.4.

Keep a running list of anything you'd call a discrepancy; the audit is only
useful if it ends in a list of (file:line, what, verdict).

## 0.2 Reading the sections

- Every `file:line` in Sections A and B was checked against HEAD `41203e0`
  when written; upstream line numbers quoted from our comments are what our
  code *claims*, and are the thing you are verifying.
- "REASON NOT DETERMINED — ask Nick" marks hunks whose motivation could not
  be recovered from commit messages, comments, or the handoff docs; those are
  yours to remember.
- Section C tables give sensor counts per observed field and as a fraction of
  the points of the file the model was **trained** on; C.3.6 gives what the
  evaluation actually used, which is what the paper reports.

---
## Contents

- §0 Workspace and tools · §0.1 suggested order · §0.2 reading the sections
- §1 Findings to read first (verified)
- §A Baselines against their upstream repositories (A.0 global facts, A.1–A.10 one per method, A.13 not-determined list)
- §B Our fork against cosmos2w/PhyCoFlow_demo (B.2 behaviour-change table, B.3 hunk-by-hunk per file, B.6 what upstream did since)
- §C Experiment YAMLs (C.1 inventory, C.2 diffs vs upstream, C.3 per-regime tables, C.4 twenty items to double-check)
- §D The 2026-09-22 pull: CoNFiLD capacity sweep, FireBench, JHU seed replicates / temporal companion, code changes, D.6 thirteen items

---

## 1. Findings to read before you start

These surfaced while preparing the guide. Each was verified directly against
the files (not taken from the section writers' reports alone). None was
changed in the repo. They are ordered by how much they could affect a claim
in the paper.

### 1.1 Every reported Senseiver row ran the *pre-audit* architecture

The upstream-exact Senseiver (per-axis positional frequencies, residual MLPs
after both cross-attentions, tied encoder layers, bare-Linear readout, plain
Adam, no gradient clipping — `src/model_baseline.py:4151-4172`) is selected
only by `upstream_layout: true`, `max_freq: null`, `optimizer: upstream`,
`grad_clip: null`. **No YAML under `Save_config/` sets any of these**
(`grep -rn upstream_layout Save_config/` is empty; the as-run snapshot
`Save_config/kolmogorov2d/det_baseline/Baseline_senseiver_Stage1_DemoN61_*.yaml`
confirms it for the Kolmogorov row). The Kolmogorov, cylinder, surface and
u-only Senseiver rows therefore used the pre-audit layout with AdamW + cosine,
grad-clip 1.0 (which the code's own measurement says binds on 100 % of steps,
`model_baseline.py:5465-5471`), scalar `max_freq 64`, and `ff_mult 4`. The
JHU row's config (`config_baseline_Senseiver_iclr.yaml`) is not on this host,
so it cannot be confirmed either way here; `src/check_xattn.py:22-31` shows
what it contained (`upstream_layout True`). Neither the paper nor the handoff
docs mention this. See §A.2.

### 1.2 The 2D S3GM rows carry ten pre-audit deviations from upstream

The audit's upstream-faithful S3GM lives in `src/s3gm3d.py` (JHU only). The
2D rows ("S3GM (PC-DPS)" in the Kolmogorov and cylinder tables) run the
fork-parent's 2D path in `model_baseline.py` with `sigma_max 7.0` (upstream
20), grad-clip 0.5 + spike skip, warmup + cosine schedule, fixed `alpha_obs
1.0`, 5 Langevin corrector steps, no overlap-consistency term, etc. The paper
discusses only the 3D adaptation. See §A.9 (2D checklist).

### 1.3 Sensor draws are not reproducible with upstream's helper

`src/helpers.py:584` (commit 7a3f03e) draws the per-field sensor *count* with
`torch.randint(...)` on the CPU generator; upstream (and our own
`helpers_baseline.py:1487`) pass `device=device`. Under the same
`torch.manual_seed` the CUDA Philox offset therefore differs before
`randperm`, so our sensor indices are not upstream's for the same seed.
Everything in the campaign is internally consistent (every method goes
through `helpers.build_sparse_condition`; the fingerprint in
`ensemble_eval.py:41` is of our draw), and `HPC_MIGRATION_AND_REMAINING_WORK.md:208`
records the non-equivalence — but it is not in the paper's protocol text.
See §B.2 #1, §B.3.2 H8.

### 1.4 DMF-Gen as run differs from the fork point in five active ways

Active in every PoF DMF-Gen config: logit-normal `t` sampling
(`Model.py:2081-2088`), EMA weights evaluated instead of raw
(`train_pointcloud_ffm.py:332-381`, `ensemble_eval.py:125-133`), bf16
autocast + `torch.compile`, an inference-only ODE step cache claimed exact
(`Model.py:1521-1553`; gate `src/test_ode_cache.py`), and the flash-eligible
cross-attention rewrite (`Model.py:994-1026`, identical except the
zero-valid-sensor edge case). The paper says DMF-Gen is evaluated "as
published" (`main.tex:514-516`). See §B.2 for the full 15-row table with
active/inactive flags.

### 1.5 Thirteen launcher-referenced YAMLs are still not in git after the 2026-09-22 pull

Including the configs for the JHU FNO3D, DeepONet, DeepONet++, Senseiver,
S3GM, SiT-matched and CoNFiLD P/F rows, and the k32/k64 DMF-Gen ablations.
`Save_TrainedModel/JHU/*` now holds only the Engaging CoNFiLD sweep and DMF-Gen
seed-replicate run dirs (Section D); the JHU Senseiver/FNO3D/DeepONet/S3GM/SiT
run dirs are still absent (the restored `HANDOFF.md` says they exist only on the
origin cluster). Full list in §A.0.3, §C.4-10 and §D.1.

### 1.6 From the 2026-09-22 pull (verified; details in §D)

- The CoNFiLD "prior sweep at fixed stage 1" (HANDOFF.md:389-395, commit
  bcd98e5) is not fixed at its ch32 point: that row is byte-for-byte the
  `strict_ld2048_hf144` arm (decoder width 144), while ch64/128/256 sit on the
  width-256 `sweep_ld2048` codec. The clean prior-only comparison in the files
  is 0.858 → 0.799 → 0.805.
- The PoF paper's footnote on a "strict 2048-point conditioning" CoNFiLD arm
  at 0.941 (`main.tex:1103-1106`) matches nothing in the pull (Engaging strict
  = 1.316 canonical, stamped) and misdescribes the arm (2048 is the latent
  size; conditioning is the standard 19 531 + 19 531 sensors).
- FireBench: DMF-Gen trains at `train_ratio 0.9` (98/12 frames), Senseiver and
  latent FM at 0.75 (80/30). §C.3.5 said 0.9 for all three; corrected. No
  FireBench row has been evaluated yet.
- The DMF-Gen seed replicates (1379, 2718) trained to 6000/6000 epochs but were
  never scored; only the DemoN33 temporal companion has canonical JSONs
  (agg 0.603 vs N29's 0.593; CRPS 0.40 vs 0.29).
- Tracked `confild_sweep2048_s2_eng.yaml` / `confild_sweep8192_s2_eng.yaml` are
  the last overwrite of a shared job copy, not the config of the runs that
  cite them; trust each run dir's `run_config.yaml` for `num_channels`.

### 1.7 Smaller items (verified)

- `model_baseline.py:6122-6125`: the `@torch.no_grad()` that decorated
  `visualize_reconstruction_latentfm` now decorates the inserted helper
  `_apply_eval_measurement_ops`; the visualizer at line 6160 is undecorated.
  Benign (`LatentFlowMatching.sample` is itself `no_grad`), but a real defect.
- SiT blocks use `qk_norm=True` (`model_baseline.py:2669`); upstream SiT does
  not. 2D SiT rows train with Huber `huber_beta 0.1`, the JHU row with MSE
  (`huber_beta 0.0`). Both are fork-parent choices, declared in the audit as
  "class (iii), unchanged". See §A.10.
- The "Geo-FNO" rows are a plain FNO on a value + mask raster; no learned
  deformation is wired in (`geofno_variant: "fno"` everywhere). The paper's
  description is accurate; the name is arguable. See §A.3.
- `gather_topk` went 32 (upstream) → 64 (ICLR-main) → 16 (every PoF run);
  the paper never states k. See §C.4-2.
- Cylinder grid-trained models train at 0.1–1 % of 80 000 grid points but
  are evaluated at 238 sensors = 0.30 % of their grid, to be count-matched
  with the mesh rows. See §C.4-3.
- Kolmogorov Geo-FNO seed replicates still use the 200 k-step budget while
  the reported row is the 50 k rerun. See §C.4-7.
- Upstream's 645-line `README.md` was deleted in 464c72e and its content
  went nowhere; a copy is in `recovered_docs/README_upstream_at_fork_point.md`.
- Two hunks have no recoverable reason: `helpers.py:342`
  `weights_only=True → False` (817654b) and the removed sensor overlay in
  `helpers_baseline._save_car_surface_field_plot` (c93c08a). See §B.

### 1.8 Good news (verified)

- Every upstream commit our code records (Senseiver e443eb0, S3GM 2343293,
  CoNFiLD 449835e, deepxde 99b6620, SiT cbde832, Gen4Turb 8af0a7b,
  neuraloperator 00b7d86) is **still that repo's HEAD** as of today. The
  clones in `upstream_baselines/` are exactly the code the ports were written
  against; upstream line numbers quoted in our comments should resolve.
- `src/sit_transport/` is byte-identical to the fork parent; the SiT and S3GM
  model classes have no hunks since the fork. Any deviation there is Jason's
  port, not ours.
- All 76 run-time YAML snapshots match their tracked source except one
  (the first fullbudget Geo-FNO launch, relaunched correctly; §C.4-8).
- Recomputed sensor fractions, step budgets and split sizes agree with the
  paper's protocol section for every 2D paper row.

---

# Section A — Baselines against their upstream repositories

Prepared 2026-09-22 from a read-only pass over
`/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion` (git `main`, HEAD 41203e0; still valid at 8aa01c9 — the pull did not touch the files cited here).
All `file:line` references below are to that checkout and were verified against the files as they
are on disk today. Every upstream repo is now cloned under `upstream_baselines/` at the recorded commit (see §0); every "UPSTREAM" column is what OUR
code/docs say to compare against, and the clone is where you open it. Where our comments cite an upstream `file:line`, that line number
was written by whoever did the original port and is only as good as that person's clone — treat it
as a locator, not as gospel.

---

## A.0 Read this first — global facts that affect every baseline

### A.0.1 Where the recorded upstream commits live (and which baselines have none)

| Baseline | Upstream repo | Commit recorded by us? | Where recorded (our file:line) |
|---|---|---|---|
| Senseiver | github.com/OrchardLANL/Senseiver | **yes, `e443eb0`** | `src/model_baseline.py:4154-4156` ("Upstream reference kept on disk at /work/hdd/bilr/ntricard/datasets/baselines/Senseiver_OrchardLANL (github.com/OrchardLANL/Senseiver @ e443eb0, which is still repo HEAD)") |
| S3GM | github.com/lzy12301/S3GM | **yes, full hash `2343293bf627e7b4afc0d63b11ad37ece0919653`** | `src/s3gm3d.py:3` |
| CoNFiLD | github.com/jx-wang-s-group/CoNFiLD | **yes, `449835e`** (2025-02-19; "last CODE change `1ae4f81`, 2024-11-08") | `Save_config/config_baseline_CoNFiLD_xcube.yaml:2-4` |
| DeepONet | lululxvi/deepxde and lululxvi/deeponet | **yes, both full hashes**: deepxde `99b6620386d18cefb1549dddb7b7fe468cfad607`, deeponet `8d62345afd39e1df9c2c8c8d0e7c41882b06a9bf` | `src/deeponet_baseline.py:9-16` |
| neuraloperator (Geo-FNO 2D, FNO3D) | github.com/neuraloperator/neuraloperator | **pip release 2.0.0** (installed in `~/venvs/jhtdb_env`, verified today); fix taken from upstream commit **`00b7d86`** | `src/fno3d_backbone.py:17-19, 62-75` |
| SiT | github.com/willisma/SiT (Ma et al.) | **recorded only in the (deleted) audit doc: `cbde832`** — see A.0.4. Nothing in `src/` records it. | `BASELINE_AUDIT_2026-08-28.md` §3 (git-only, commit e5ee749) |
| Gen4Turb | github.com/vivekoommen/Gen4Turbulence | **recorded only in the deleted audit doc: `8af0a7b`**. Nothing in `src/` records it. | `BASELINE_AUDIT_2026-08-28.md` §3 |
| Latent FM (LFM) | none — fork-parent's own implementation | n/a | — |
| MLP-RBF | none — our own control (fork-parent DMF-Gen backbone) | n/a | — |
| Classical anchors | none — written from scratch (scipy cKDTree + numpy) | n/a | — |
| RectifiedFlow (gnobitab) | idea citation only | no | `src/model_baseline.py:1572`, `src/Model.py:2024`, `src/phycoflow_pointcloud/models/portable_core.py:2354` — all three sit inside the **DMF-Gen `PointCloudFFM` docstring** ("Core 1-RF idea: ..."). No code was taken from that repo; see A.12. |

Run-directory metadata does **not** record commits: `run_metadata.json` (written by
`src/train_Det_Baseline.py:115` and `src/train_Gen_Baseline.py:139`) contains only
`config_path, baseline_model, training_stage, device, cuda_device_name, started_at`
(checked on `Save_TrainedModel/kolmogorov2d/baseline_sit_p4/Baseline_sit_Stage1_DemoN66_20260908_132003/run_metadata.json`).
`deeponet_baseline.py:9-10` says "commits recorded in the run metadata" — that is the module
docstring itself, not a JSON field.

### A.0.2 Provenance layers: fork parent vs. our fork

The repo is a fork of `cosmos2w/PhyCoFlow_demo` (remote `upstream` is fetched; merge-base
`828c3c6`, 2026-06-05). The **fork parent already contained** `model_baseline.py` with S3GM, SiT,
Senseiver (pre-audit layout), LFM, MLP-RBF, Geo-FNO adapters, and the vendored `sit_transport/`
(first parent commit `556e5b6` 2026-04-21 "Updated Generative Baselines", author cosmos2w).

`git diff 828c3c6 HEAD -- 0_demo_TurbulentCombustion/src/{model_baseline,helpers_baseline,train_Gen_Baseline,train_Det_Baseline}.py`
(or `./cmp.sh fork src/model_baseline.py`) shows what **we** changed:

* `src/sit_transport/*` — **byte-identical** to the fork parent (no hunks). Section B (fork diff) will not show anything here; any deviation from Ma et al. was made by cosmos2w.
* SiT model classes `model_baseline.py:2617-3092` — **no hunks** (unchanged since fork parent).
* S3GM model classes `model_baseline.py:3108-4088` — **no hunks** (unchanged since fork parent).
* Senseiver classes `4175-4518` — **heavily rewritten by us** (the "upstream_layout" restoration).
* LFM: `ConvAE3D`, `LatentFMUNet3D`, `PointNetSensorEncoder`, `spatial_dim` branches — added by us; 2D `ConvAE/LatentFMUNet/LatentFlowMatching` are fork-parent code.
* `CoNFiLDAdapter` 8215-8270, `run_epoch_senseiver` grad-norm logging, `LatentFMAdapter` 3D paths, DPS/visualizer changes — ours.

So for "word for word vs upstream", the SiT and S3GM **model** code must be diffed against Ma et al. / Li et al., and the diff you see will be cosmos2w's port, not ours.

### A.0.3 Files that the launchers reference but that are NOT on disk (you cannot check what you cannot open)

Verified with a loop over every `Save_config/*.yaml` string in `src/*.sh`, `src/*.slurm`:

```
MISSING: Save_config/config_baseline_Senseiver_iclr.yaml        <- JHU Senseiver headline row (train_senseiver_iclr.sh:38, verify_seedcheck.sh:19)
MISSING: Save_config/config_baseline_Senseiver_iclr_smoke.yaml
MISSING: Save_config/config_baseline_S3GM_xcube.yaml            <- JHU S3GM row (s3gm_train.sh:22)
MISSING: Save_config/config_baseline_DeepONet_iclr.yaml         <- JHU DeepONet row (train_deeponet.py:42 default, train_deeponet.sh)
MISSING: Save_config/config_baseline_DeepONetPP_iclr.yaml       <- DeepONet++ (train_deeponetpp.sh)
MISSING: Save_config/config_baseline_fno3d_xcube.yaml           <- JHU FNO3D row (train_fno3d_matched.sh:10)
MISSING: Save_config/config_baseline_SiT_xcube_matched.yaml     <- JHU SiT-point matched arm (train_sit_xcube_matched.sh)
MISSING: Save_config/config_baseline_CoNFiLD_figsmoke.yaml
```
Also missing: CoNFiLD arm **P** (`unified_published_prior`) and arm **F** (`unified_faithful384`) configs — only arm **C** (`config_baseline_CoNFiLD_xcube.yaml`) is on disk (paths in `src/confild_stageA_fit.py:40-44`).
`Save_TrainedModel/JHU/*` directories are all **empty** on this host (the JHU run dirs, including their `run_config.yaml`, were not transferred — see `DATA_TRANSFER_MANIFEST.md:30`).
Consequence: for the JHU rows of Senseiver, S3GM, DeepONet(++), FNO3D, SiT-matched, CoNFiLD P/F you can check the *code*, but the exact hyper-parameters as run must come from the MIT Engaging host or from git history (`git log --all -- '*Senseiver_iclr*'` finds nothing — they were never committed).

`/work/hdd/bilr/ntricard/datasets/baselines/{Senseiver_OrchardLANL,CoNFiLD,Gen4Turbulence}` are now **symlinks** to the pinned clones in `upstream_baselines/` (created 2026-09-22 so the hard-coded paths in the CoNFiLD scripts and `gen4turb_eval.py` resolve). Note the Gen4Turbulence symlink points at a *pristine* clone: Nick's modified clone with `train_model_uxuz.py`, the anneal edit and `Par.pkl` was never transferred from Kestrel/Engaging.

### A.0.4 Audit documents that exist only in git history

`BASELINE_AUDIT_2026-08-28.md` (1913 lines), `FLEET_AUDIT_2026-08-29.md`, `PLAN_IMPROVE_2026-08-30.md`
were deleted from the tree in commit `464c72e` ("Removed old MD files", 2026-09-08). They are the
"per-method audit notes" the paper promises (`main.tex:363-368, 397`). Extracted copies are in
`recovered_docs/BASELINE_AUDIT_2026-08-28.md` etc. (`git show e5ee749:0_demo_TurbulentCombustion/<name>`).
Section 3 of the baseline audit is the per-baseline verdict table; §15 (S3GM), §13 (FNO), §12 (CoNFiLD),
§23-24 (Senseiver/S3GM), §27 (DeepONet), §8 (Gen4Turb) contain the item-by-item upstream diffs.
Quote from `FLEET_AUDIT_2026-08-29.md` "Verdicts": Senseiver PASS, CoNFiLD PASS ("verbatim upstream decoder"),
S3GM PASS ("verbatim port @2343293; z-as-time caveat stands"), Latent FM PASS ("conditioning fix live via lfm_fixes monkey-patch"),
Gen4Turb PASS ("upstream code untouched; anneal arm = 1 labelled scheduler.step line"), SiT PASS ("transport lib byte-identical; declared Euler-32 deviation"),
FNO3D PASS ("SpectralConvOddSafe = upstream HEAD fix"), DeepONet PASS ("upstream-exact branch/trunk; set-encoder = declared adaptation").

### A.0.5 Monkey-patch modules (code that changes behaviour without being in the class you read)

| Module | What it patches | Gated by | Who imports it |
|---|---|---|---|
| `src/lfm_fixes.py` (244 lines) | `LatentFlowMatching._encode_condition` → `cond_mode: image_norm` (masked-mean pooling + density mask) and `latent_scale_mode: global\|per_channel` (LDM-style latent rescale) | stage-2 config keys | `src/eval_latentfm_ensemble.py:187-207` auto-imports when the run config needs it; `src/dump_fields_baseline.py:174-176` |
| `src/sen_sweep_fixes.py` (292 lines) | Senseiver fixed-mask TUNE validation + patience early-stop (PATCH A); **"Senseiver+local" IDW residual (PATCH B, "ENHANCED VARIANT, not upstream")** | env `SEN_VAL_FIXED`, `SEN_LOCAL_IDW`, ... | `src/train_det_sweep.py` via `train_senseiver_sweep.sh:37-45`; `eval_senseiver_sweep_wrap.py:29` |
| `src/sen_local_xattn.py` (132 lines) | Senseiver local cross-attention variant | env `SEN_LOCAL_XATTN` | `eval_senseiver_sweep_wrap.py:31`, `check_xattn.py` |
| `src/s3gm_norm_guidance.py` (339 lines) | replaces `s3gm3d.s3gm_reconstruct_ensemble` with normalised-gradient guidance (`install()` at :276) | explicit `install()` | the "S3GM (normalized guidance)" row (`main.tex:997`) |

`assemble_baseline_table.py:40-41`: "Senseiver+local / +xattn enhanced variants WITHDRAWN (Nick 2026-08-30: architecture must stay as-is; only hyperparameter arms are fair)." So PATCH B / xattn are not paper rows.

### A.0.6 Shared glue every baseline goes through (not upstream code, but check it once)

* Sensor draw: `src/helpers.py:507` `build_sparse_condition` (CPU `torch.randint` at :584/:649, then `torch.randperm` on device) is the **canonical** one used by every eval driver. `src/helpers_baseline.py:1352` is a copy that passes `device=device` to `randint` (:1487/:1491) and therefore yields a *different* permutation for the same seed; `model_baseline.py` training loops bind the `helpers_baseline` copy (`model_baseline.py:4129`). Comments citing `helpers.py:536` / `helpers_baseline.py:1457` are stale line numbers.
* Grid rasterisation used by S3GM-2D / SiT-patch / Geo-FNO / LFM: `helpers_baseline.py:2196` `compute_pad_size`, `:2208` `pointcloud_to_grid_padded`, `:2233` `grid_to_pointcloud`, `:2259` `build_obs_grid_mask`, `:2304` `build_obs_grid_mask3d`, `:1505` `nearest_fill_grid` (Voronoi fill for `cond_mode: interp`), `:2347` `scatter_sensors_to_nodes`, `:2364` `nearest_sensor_fill_nodes`, `:992` `validate_regular_grid_compatibility`.
* Trainers: `src/train_Gen_Baseline.py:63` `main()` (generative: s3gm/latent_fm/sit/confild) and `src/train_Det_Baseline.py:64` `main()` (senseiver/mlp_rbf/geofno); registry `model_baseline.py:8272-8288`. Wall-clock budget stop + tail archive: env `BASELINE_MAX_HOURS` / `BASELINE_ARCHIVE_N` (`train_Det_Baseline.py:171-177`).
* Metrics: `src/ensemble_eval.py` `ensemble_metrics` (all rows).
* Installed versions today (`~/envs/jhtdb`): `neuraloperator 2.0.0`, `timm 1.0.29` (SiT blocks import `timm.models.vision_transformer.Attention, Mlp, PatchEmbed`); `deepxde` is **not** installed (DeepONet is reimplemented, not imported).

---

## A.1 Classical anchors (nearest neighbor, IDW k=8, gappy POD, training mean)

**Upstream.** None. Written from scratch (audit §3: "Classical anchors — built from scratch; none existed"). Cites Everson & Sirovich (1995) for gappy POD (`main.tex:444`). Nothing to clone; check against the textbook definitions and against `scipy.spatial.cKDTree` semantics.

**Our implementation (all "our own").**

| File | Elements | Lines |
|---|---|---|
| `src/baseline_classical_jhu.py` (681) | `draw_sensors` 158-193; `kd_predict` (NN and IDW) 195-217; `GappyPOD` (method-of-snapshots, basis never materialised) 219-258; `obs_columns` 260-274; `score` 276; `summarize` 288; `main` 325+ (periodic flag :344-349, boxsize :402-410) | |
| `src/baseline_classical_2d.py` (922) | "2D port of baseline_classical_jhu.py (read them side by side ...)" :3-5. `draw_sensors` 173; `kd_predict` 214-239; `GappyPOD` 241-280; `build_coords_box` 408 (periodic rescale); `self_test` 451; `main` 567 | |
| `src/baseline_gappy_pod_wing.py` (118) | joint surface+volume POD, least squares on surface taps (wing dataset; not a PoF row) | |
| Launchers | `run_classical.sh` (JHU), `run_classical_kolm.sh`, `run_classical_kolm_sweep.sh`, `run_classical_cyl.sh`, `run_classical_cyl_surface.sh`, `run_classical_cyl_uonly.sh`, `run_classical_figs.sh`; outputs `Save_TrainedModel/*/baseline_classical/*.json` | |

**Element checklist.**

| Element | Ours | Compare against |
|---|---|---|
| NN interpolation | `baseline_classical_jhu.py:205-207` (`tree.query(k=1)`) / `_2d.py:214+` | definition; `cKDTree(boxsize=...)` doc |
| IDW k=8, weights `1/max(d,1e-12)`, exact-hit override `d<1e-11` | `jhu.py:208-216` | Shepard IDW p=1 (note: **power 1, not 2** — confirm this is what the paper means by "inverse-distance weighting") |
| Gappy POD: POD from TRAIN snapshots, coefficients by lstsq on sensor rows, rank r | `jhu.py:219-258` (`reconstruct` :253-258) | Everson & Sirovich 1995 eqs.; rank 80 chosen "on train cube 2" (audit §3) |
| Training-mean constant = exactly 0 in z-score units | docstring `jhu.py:27-37` | trivial |
| Unobserved channels → 0 (train mean) for NN/IDW; all channels from POD basis | `jhu.py:17-26`, `kd_predict` docstring | policy decision, document it |
| Periodicity | `jhu.py:344-349` help text: "use the WRONG periodic wrap (the cutout spans 12.11% of the 2pi domain and is NOT periodic; audit 2026-08-28 s26)"; `_2d.py:50-54` "DEFAULT IS NON-PERIODIC" | paper `main.tex:446-449` ("an earlier periodic implementation understated these floors ... corrected in audit") |

**Declared deviations.** None from an upstream (there is none). The audit-driven change is periodic → non-periodic default (`FLEET_AUDIT` "FLOOR BUG: on-disk JSON used periodic KD-tree/IDW (wrong)"; corrected JSONs `classical_baselines_main_1pct_nonperiodic_2d.json` etc. in `Save_TrainedModel/kolmogorov2d/baseline_classical/`; Kolmogorov table reports both per./non-per. rows, `tab_kolm_body.tex:10-13`).

---

## A.2 Senseiver (Santos et al., Nat Mach Intell 2023)

**Upstream.** `github.com/OrchardLANL/Senseiver @ e443eb0` ("still repo HEAD" as of 2026-08-28) — `src/model_baseline.py:4154-4156`. Upstream files our comments cite: `positional.py:18-22`, `model.py:24-30, 33-43, 46-55, 74-83, 176-192, 194-197, 215-216, 235-255, 242, 276`, `network_light.py:68, 78`, `s_parser.py:45` (all in `model_baseline.py:4158-4167, 4178-4181, 4217, 4228, 4243, 4270, 4309, 4409, 4431, 4445, 4460, 5456, 5465, 7877, 7926`). The reference clone was at `/work/hdd/bilr/ntricard/datasets/baselines/Senseiver_OrchardLANL` (not on this host; `DATA_TRANSFER_TIER2.md:24`).

**Our implementation — reimplemented in PyTorch (`nn.MultiheadAttention`), not vendored.** Two layouts coexist behind `upstream_layout`:

| File | Element | Lines | Kind |
|---|---|---|---|
| `src/model_baseline.py` | header comment: what `upstream_layout=True` reproduces | 4151-4172 | doc |
| | `SenseiverFourierPositionalEncoding` (per-axis `linspace(1, N_axis/2, num_bands)`; non-persistent `freqs` buffer) | 4175-4213 | reimplemented |
| | `_senseiver_mlp` (LayerNorm, Linear(d,d), GELU, Linear(d,d); ff_mult=1 = upstream) | 4216-4224 | reimplemented |
| | `SenseiverSelfAttentionBlock` (upstream path 4242-4246, 4260-4263; pre-audit path 4247-4255, 4264-4266) | 4227-4266 | reimplemented |
| | `SenseiverCrossAttentionLayer` = Sequential(Residual(CrossAttn), Residual(mlp)) | 4269-4305 | reimplemented |
| | `SenseiverEncoderBlock` (cross + `num_self_attn_layers` self-attn) | 4308-4367 | reimplemented |
| | `Senseiver` (field embedding, kv projection, latent array, weight-tied encoder `layer_1`+`layer_n`, decoder query token, bare-Linear readout) | 4370-4518 | reimplemented + our per-channel-mask adaptation |
| | `run_epoch_senseiver` (MSE mean, grad-norm measurement, optional clip) | 5409-5509 | our glue |
| | `SenseiverAdapter` (per-axis `max_freq: null` → grid shape 7877-7892; `optimizer: upstream` → plain Adam 7926-7930, else AdamW+cosine 7931-7940) | 7868-8004 | our glue |
| | `visualize_reconstruction_deterministic` | 6802-6994 | our glue |
| `src/eval_senseiver_iclr.py` (331) | JHU canonical eval (K=1, CRPS=MAE, Voronoi floor, sensor-count sweep) | whole | our glue |
| `src/diag_senseiver_gradnorm.py` (109) | measures pre-clip grad norm ("Does clip_grad_norm_(1.0) actually bind ...") | whole | diagnostic |
| `src/train_det_sweep.py`, `sen_sweep_fixes.py`, `sen_local_xattn.py`, `eval_senseiver_sweep_wrap.py`, `check_xattn.py`, `check_sweep_params_and_idw.py` | improvement arms (wide bottleneck 64×320, +local IDW, +xattn) | | our own; withdrawn from paper except the capacity arm |

**Element-by-element checklist (open `model_baseline.py` next to upstream `Senseiver/`):**

| # | Element | Ours (`src/model_baseline.py`) | Upstream (as cited by us) | Note |
|---|---|---|---|---|
| 1 | Positional encoding: per-axis top frequency = grid shape, `linspace(1, N/2, bands)`, sin/cos, coords mapped to [-1,1] | 4184-4213 | `positional.py:18-22` | With `max_freq: 64.0` (all on-disk configs) the pre-audit scalar ladder is used; `max_freq: null` selects upstream (7880-7888). **No on-disk config sets `null`.** |
| 2 | MLP width-preserving (LN, Linear(d,d), GELU, Linear(d,d)) | 4216-4224 | `model.py:24-30` | pre-audit `ff_mult=4` MLP at 4249-4255 |
| 3 | Cross-attention = Residual(CrossAttn) + Residual(mlp), LN on q and kv | 4295-4305 | `model.py:33-43` | pre-audit encoder cross-attn has **no** trailing MLP (4361-4364) |
| 4 | Self-attention layer, dropout on residual branch | 4257-4263 | `model.py:46-55, 74-83` | |
| 5 | Encoder block = 1 cross + `num_self_attn_per_block` self layers | 4352-4367 | `model.py:176-192` (`Encoder.create_layer`) | |
| 6 | Encoder weight tying: `layer_1` then a single `layer_n` reused for depth>1 | 4430-4434, 4494-4499 | `model.py:194-197, 215-216` | `share_encoder_layers` defaults to `upstream_layout` (4400-4402) |
| 7 | Sensor token = [value, field-embedding, pos-enc] → Linear(kv_dim) | 4406-4415, 4487-4490 | upstream: value + pos enc, `enc_preproc_ch=64` default (`s_parser.py:45`) | **field embedding is our adaptation** ("our per-channel-mask adaptation", 4414); kv_dim defaults to latent_dim, not 64 (4408-4412) |
| 8 | Latent array init `randn*0.02` | 4416 | check upstream init | not cited |
| 9 | Decoder query = [pos-enc \| output token], no preproc unless `dec_preproc_ch` | 4442-4452, 4504-4511 | `model.py:235-255` | |
| 10 | Decoder readout = bare `nn.Linear` | 4461 | `model.py:242, 276` | pre-audit adds LayerNorm (4472-4475) |
| 11 | Loss `F.mse_loss` reduction='mean' | 5460 | `network_light.py:68` uses `reduction='sum'` | comment 5456-5459 argues Adam-invariance |
| 12 | Optimizer: plain Adam, constant LR, no wd, no schedule | 7928-7930 (only when `optimizer: upstream`) | `network_light.py:78` | **default path (all on-disk configs) is AdamW + CosineAnnealingLR** 7932-7940 |
| 13 | Gradient clipping: upstream has none | 5465-5476: `grad_clip` default **1.0** unless config says `null` | none upstream | comment 5465-5471 quotes the measurement (median 3.51, binds on 100 % of steps → "normalised-gradient Adam") |
| 14 | Query subsampling `n_query_points` per step | 5447-5453 | upstream trains on full field? (not cited) | our protocol choice |
| 15 | Sensor randomisation per step (`build_sparse_condition`, log-uniform 0.1-1 %) | 5438-5445 | upstream fixed sensors? (not cited) | protocol |
| 16 | Eval: K=1, fixed 19,531 sensors/channel, Voronoi floor | `eval_senseiver_iclr.py` docstring 1-33 | n/a | |

**Declared deviations (quoted).**
* `model_baseline.py:4414`: "+ field embedding (our per-channel-mask adaptation)".
* `model_baseline.py:5465-5471`: "Upstream has NO gradient clipping. Measured on this config (diag_senseiver_gradnorm.py, 80 steps): the pre-clip global norm has median 3.51, min 1.27, max 28.6 -- so max_norm=1.0 binds on 100% of steps ... `grad_clip: null` restores upstream exactly."
* `model_baseline.py:4169-4171`: "`upstream_layout=False` is the pre-audit layout and exists ONLY so that the frozen pre-audit run directories ... stay loadable. New runs must set it True."
* Audit §3: "Restored: per-axis max_freq from grid shape (was scalar 64, halving the spectrum), residual feed-forwards after both cross-attentions, encoder weight tying, width-preserving MLP, bare-Linear readout, plain Adam constant LR. Gradient clip at 1.0 ... removed to match upstream. New config 64x320 = 6,419,396 params (-1.3%)".

**IMPORTANT finding for tonight.** The audit-restored layout is only selected by `upstream_layout: true`, `max_freq: null`, `optimizer: upstream`, `grad_clip: null` in the YAML. `grep -rn upstream_layout Save_config` returns **nothing**. Every on-disk Senseiver config (`Save_config/kolmogorov2d/config_baseline_Det_kolm.yaml:70-93`, `cylinder2d/config_baseline_Det_cyl.yaml`, `cylinder2d_surface/...`, `config_baseline_Det_xcube.yaml:113-136`, and the backed-up as-run copies in `Save_config/kolmogorov2d/det_baseline/Baseline_senseiver_Stage1_DemoN61_*.yaml`) therefore ran the **pre-audit layout with AdamW + cosine + clip 1.0 + scalar max_freq 64 + ff_mult 4**. Only the JHU row (config `config_baseline_Senseiver_iclr.yaml`, **missing**) would have used the restored layout; `check_xattn.py:22-31` shows what that config contained (latent 320, 128 latents, heads 2/2/1, ff_mult 1, max_freq [125,125,125], upstream_layout True, enc_preproc_ch 320, share_encoder_layers True). **The 2D Kolmogorov/cylinder/surface Senseiver rows are not the upstream-faithful architecture.** Decide whether the paper's "upstream-faithful configurations are frozen as the headline rows" (`main.tex:382`) holds for those rows.

**Configs / launchers.** JHU: `train_senseiver_iclr.sh` → `train_Det_Baseline.py --config Save_config/config_baseline_Senseiver_iclr.yaml` (missing); eval `eval_senseiver_iclr.sh` → `eval_senseiver_iclr.py`; sweep `train_senseiver_sweep.sh` → `train_det_sweep.py`. 2D: `submit_2d_fleet_delta.sh` / `train_kolm_baseline.sh` / `train_cyl_baseline.sh` with `Save_config/{kolmogorov2d,cylinder2d,cylinder2d_surface,cylinder2d_uonly}/config_baseline_Det_*.yaml` (`baseline_model: senseiver`); eval `eval_kolm_fleet.sh` → `eval_kolm_ensemble.py`. Cylinder Senseiver reads `Cylinder2D_mesh.h5` (`config_baseline_Det_cyl.yaml:21`).

---

## A.3 Geo-FNO (2D rows) and FNO3D (JHU row)

**Upstream.** `neuraloperator` **2.0.0 pip release** (`from neuralop.models import FNO`, `model_baseline.py:13-24`; `fno3d_backbone.py:56-57`), i.e. a dependency, **not copied code** — with one exception: `SpectralConvOddSafe.forward` (`fno3d_backbone.py:78-217`) is a copy of `neuralop/layers/spectral_convolution.py::SpectralConv.forward` from the installed 2.0.0 with the upstream-HEAD `ifftshift` fix (`:196-199`; upstream commit `00b7d86`, `:70-71`). Paper cites Li et al. 2021 (FNO) and Li et al. 2023 (Geo-FNO) (`main.tex:458`).

**Our implementation.**

| File | Element | Lines | Kind |
|---|---|---|---|
| `src/model_baseline.py` | `FNOSupervisedGrid` (neuralop `FNO(n_modes=(ny,nx), in=2*n_fields, out=n_fields, positional_embedding="grid")`; input = [values, mask]) | 4520-4549 | dependency wrapper (fork parent) |
| | `FNOSupervisedIrregular` (scatter to latent grid, FNO, `grid_sample` back) | 4552-4625 | ours/fork parent; **not used** (all configs `geofno_variant: "fno"`) |
| | `run_epoch_geofno` (MSE on padded grid; AdamW; clip 1.0 at 5630) | 5566-5638 | glue |
| | `GeoFNOAdapter` (AdamW+cosine 8138-8146) | 8100-8213 | glue |
| `src/fno3d_backbone.py` (628) | `SpectralConvOddSafe` 78-217; `validate_regular_grid_compatibility_3d` 218-288; `SpectralConvOddSafeIsland` (fp32 island) 289-301; `FNO3D` 303-614 (neuralop `FNO(..., conv_module=SpectralConvOddSafe*)` at 381-392; condition rasterisation `_build_condition_maps` 519-552; `forward` 553-614); `FNO3DFFM(PointCloudFFM)` 615-628 | | dependency + our 3D wrapper inside the DMF-Gen rectified-flow framework |
| `src/train_pointcloud_ffm.py` | `--backbone fno3d` branch | 1129-1175 | glue |
| `src/evaluate_ffm.py` | FNO3D rebuild for eval | 242-260 | glue |
| `src/bench_fno3d.py` (144) | memory/param sweep | | diagnostic |

**Checklist.**

| # | Element | Ours | Upstream |
|---|---|---|---|
| 1 | FNO block (lifting, `SpectralConv`, skip, projection) | `neuralop.models.FNO` instantiation `model_baseline.py:4539-4546` / `fno3d_backbone.py:381-392` | neuraloperator 2.0.0 `neuralop/models/fno.py` |
| 2 | `SpectralConv.forward` copy with `ifftshift` on inverse path | `fno3d_backbone.py:81-217` (fix at 196-199) | installed `neuralop/layers/spectral_convolution.py` (2.0.0) vs upstream HEAD `00b7d86` |
| 3 | Mode truncation `n_modes` 2-tuple (2D) / 3-tuple (3D) | `4540`, `4572`; `fno3d_backbone.py:381+` | `FNO(n_modes=...)` |
| 4 | "Geo-FNO" label: **no learned deformation is used** in the paper rows (`geofno_variant: "fno"`, e.g. `Save_config/kolmogorov2d/config_baseline_GeoFNO_kolm.yaml:105`); cylinder reads the uniform-grid export `Cylinder2D_grid.h5` (`config_baseline_GeoFNO_cyl.yaml:21`); the deformer classes `SiTLearnedGridDeformer` 3028-3050 / `S3GMLearnedGridDeformer` 4058-4088 are not wired into `GeoFNOAdapter` | — | Li et al. 2023 Geo-FNO has a learned coordinate deformation; ours is a plain FNO on a mask+value raster. Paper text `main.tex:457-461` describes exactly the raster, so the *description* is honest, the *name* is arguable. |
| 5 | Conditioning channels: value + binary mask per field | `4549`; FNO3D `4*n_fields+1` (`fno3d_backbone.py:20-27`) | ours |
| 6 | Optimizer AdamW + cosine, clip 1.0 (2D); FNO3D uses the DMF-Gen trainer's optimiser | `8138-8146`, `5630` | not an upstream training recipe (neuraloperator has none for this task) |
| 7 | `domain_padding=None` on a non-periodic cutout; time channel only at k=0 | audit §13 "Class (iii), flagged and NOT changed" | — |

**Declared deviations.** `fno3d_backbone.py:59-75` (odd-axis `fftshift` bug in 2.0.0, "Measured ... max relative error 2.0e+00 at N=125 versus 2.5e-07 at N=124 ... Upstream fixed this at HEAD ... 00b7d86 ... We restore that behaviour here rather than upgrading the shared virtualenv"). `fno3d_backbone.py:15-21` "spectral core ... upstream neuralop.models.FNO verbatim ... The only change is n_modes becoming a 3-tuple". Paper: Kolmogorov Geo-FNO budget error 200k steps → matched rerun (`main.tex:375-379`; `train_kolm_config_baseline_GeoFNO_kolm_matched_*.log`).

**Configs / launchers.** 2D: `Save_config/{kolmogorov2d,cylinder2d,cylinder2d_surface}/config_baseline_GeoFNO_*.yaml` (modes 32/32 Kolm; 32/24 cyl; hidden 64; 4 layers), as-run copies `Save_config/kolmogorov2d/det_baseline/Baseline_geofno_*.yaml`; launch `submit_2d_fleet_delta.sh`, `geofno_matched_sweeps.sh`. JHU FNO3D: `train_fno3d_matched.sh` → `train_pointcloud_ffm.py --backbone fno3d` with `config_baseline_fno3d_xcube.yaml` (missing); eval `eval_fno3d.sh` → `ensemble_eval.py`.

---

## A.4 MLP-RBF

**Upstream.** None — **our own control**. `Save_config/config_baseline_Det_xcube.yaml:8-9`: "MLP-RBF uses the original FFM point MLP + RBF sparse-gather backbone in direct supervised mode." `model_baseline.py:317-318`: "This is your current model, kept under a clearer name so it can be compared directly against the Perceiver backbone." The class is a near-verbatim copy of the DMF-Gen backbone `src/Model.py:164` `ConditionalPointMLPRBF` (diff = two cosmetic lines). Paper `main.tex:461-465` calls it "a point-native deterministic control".

**Our implementation.** `model_baseline.py`: `make_mlp` 115-124, `FourierPositionalEncoding` 125-141, `ConditionalPointMLPRBF` 309-405, `DeterministicMLPRBFRegressor` (t=0, x_t=0 wrapper) 406-440, `run_epoch_mlp_rbf` 5512-5563 (MSE, AdamW, clip 1.0), `MLPRBFAdapter` 8006-8098.
**Checklist.** Only against `src/Model.py:164-260` (fork-parent DMF-Gen backbone) — this belongs to Section B, not to any external upstream.
**Configs.** `*_params.mlp_rbf_params` in the same `config_baseline_Det_*.yaml` files as Senseiver, plus `config_baseline_MLPRBF_{kolm,cyl,cyl_surface}.yaml`; cylinder reads the mesh export (`config_baseline_MLPRBF_cyl.yaml:21`).

---

## A.5 DeepONet (upstream-faithful + set-encoder adaptation) and DeepONet++ (our arm)

**Upstream.** `lululxvi/deepxde @ 99b6620386d18cefb1549dddb7b7fe468cfad607` (files `deepxde/nn/pytorch/deeponet.py`, `deepxde/nn/pytorch/fnn.py`, `deepxde/nn/deeponet_strategy.py`, `deepxde/nn/initializers.py`) and `lululxvi/deeponet @ 8d62345afd39e1df9c2c8c8d0e7c41882b06a9bf` (`src/deeponet_pde.py`) — `src/deeponet_baseline.py:9-16`. Line cites: `deeponet_pde.py:275, :259, :168`; `deeponet.py:121-123, :292-294`; `fnn.py:49-50, :56-67, :66`; `deeponet_strategy.py:35` (SplitBranchStrategy); `initializers.py:134`. `deepxde` is not installed — the network is **reimplemented** (~100 lines).

**Our implementation.**

| File | Element | Lines | Kind |
|---|---|---|---|
| `src/deeponet_baseline.py` (195) | `_fnn_layers` (Linear + Glorot-normal W + zero b) 73-82; `DeepONetSetBranch` 84-167: field embedding 109-114, branch phi 116-118, rho 120-121, trunk 123-124, bias `b` 126-127, ReLU 129, `encode` 132-143 (mean-pool at 138-139, linear output 142), `trunk_forward` 146-151 (activated output 151), `combine` 153-155 (einsum + bias); `build_deeponet` 170-181 | | reimplemented + adaptation |
| `src/train_deeponet.py` (423) | standalone trainer ("Deliberately standalone", :3-6); `run_epoch` 147-200 (MSE 184, pure-measurement clip `max_norm=inf` 188-192); `Adam(lr)` 243 | | glue |
| `src/eval_deeponet_iclr.py` (319) | canonical JHU eval (K=1) | | glue |
| `src/deeponetpp.py` (285) | `DeepONetPP` 90-256: binned pooling `_binned_pool` 166, Fourier moments `_fourier_moments` 184, RFF trunk 226 | | **our own arm** |
| `src/train_deeponetpp.py` (526), `src/eval_deeponetpp.py` (99) | trainer/eval for ++ (Adam :298, measurement clip :246) | | glue |

**Checklist (open `deeponet_baseline.py` beside deepxde):**

| # | Element | Ours | Upstream |
|---|---|---|---|
| 1 | Unstacked DeepONet: one branch, one trunk, inner product, trainable scalar bias init 0 | `:126-127`, `:153-155` | `deeponet_pde.py:275` (`stacked=False`), `deeponet.py:121-123, 292-294` |
| 2 | Branch output LINEAR | `:142` | `fnn.py:56-67` |
| 3 | Trunk output ACTIVATED | `:151` | `deeponet_strategy.py:35` |
| 4 | ReLU, Glorot-normal weights (`xavier_normal_`), zero bias | `:73-82`, `:129` | `initializers.py:134`, `fnn.py:49-50` |
| 5 | Multi-output via `split_branch`: shared trunk width p, branch emits `num_outputs*p` | `:120-121`, `:143` | `deeponet_strategy.py` SplitBranchStrategy (Lu et al. CMAME 2022 §3.1.6) |
| 6 | Adam lr 1e-3 constant, no wd, no schedule, no clipping | `train_deeponet.py:242-243`, `:188-192` | `deeponet_pde.py:259, :168` |
| 7 | **Branch input = permutation-invariant mean over (coord, value, field-embedding) tokens** | `:132-139` | vanilla: fixed sensor vector in R^m — **declared adaptation** |
| 8 | Coordinates mapped to [-1,1] before branch/trunk | `:135`, `:147` | check upstream scaling |
| 9 | Field embedding `nn.Embedding` init N(0,1) | `:113-114` | not upstream ("OUR ADAPTATION", :109-112) |
| 10 | Query subsampling + random sensor draws per step | `train_deeponet.py:147-200` | protocol |

**Declared deviations (quoted).** `deeponet_baseline.py:35-61` "The one structural adaptation: the branch input ... We therefore make the branch permutation-invariant over the observed set ... This is DeepSets/PointNet pooling, NOT attention". Audit §27b: "Chosen: (b), a permutation-invariant set-encoder branch. Cost stated plainly: this is not vanilla DeepONet and the table caption must say so." §27d "Class (iii), flagged and NOT changed: No positional encoding on the trunk ... Mean aggregation over sensors". Paper `main.tex:468-473`. DeepONet++ (`deeponetpp.py:1-70`) is explicitly "the labelled 'DeepONet++ (structured branch)' variant", keeps conventions listed at `:57-62`.

**Configs / launchers.** `train_deeponet.sh` → `train_deeponet.py --config Save_config/config_baseline_DeepONet_iclr.yaml` (**missing**); `eval_deeponet.sh`; `train_deeponetpp.sh` / `eval_deeponetpp.sh` with `config_baseline_DeepONetPP_iclr.yaml` (**missing**). Default arch in code: p=768, branch 736, phi 3 layers, rho 2, trunk 960×4, embed 32 (`deeponet_baseline.py:94-102`) = 6,505,732 params (audit §27).

---

## A.6 Latent flow matching (LFM)

**Upstream.** **None.** It is the fork parent's (cosmos2w) own implementation, first appearing in parent commit `556e5b6` (2026-04-21). No external repo is referenced anywhere in the LFM classes; the only citation is `model_baseline.py:2134-2136` "Adapted from Kashefi et al. (arXiv:2601.03030) 'Flow Matching and Diffusion Models via PointNet ...'" for the optional `PointNetSensorEncoder` (`cond_mode: pointnet`, **not used in any paper row** — all configs use `cond_mode: image`, e.g. `config_baseline_Gen_xcube.yaml:123`, `kolmogorov2d/config_baseline_S3GM_kolm.yaml:102`). Paper `main.tex:477-483` frames it as "the latent-diffusion-family stand-in" citing Rombach 2022 / Du 2024 — i.e. a family stand-in, not a port. **There is nothing to diff word-for-word; the audit is Section B (fork diff) plus a design review.**

**Our implementation (`src/model_baseline.py` unless noted).** `_gn_groups` 1833, `_ResBlock2d` 1839-1864, `ConvAE` 1866-1991 (fork parent), `_ResBlock3d` 1993-2008 + `ConvAE3D` 2009-2069 (ours), `_TimestepEmbedding` 2074-2101, `_AdaGNResBlock` 2102-2130, `PointNetSensorEncoder` 2131-2212, `LatentFMUNet` 2213-2358, `_AdaGNResBlock3D` 2359-2376 + `LatentFMUNet3D` 2377-2460 (ours), `LatentFlowMatching` 2461-2609 (`_downsample_mask` 2504, `_encode_condition` 2522-2559 incl. avg-pool of masked sparse grid at 2536-2545, `training_loss` 2561-2577 = rectified flow `x_t=(1-t)x0+t x1`, target `x1-x0`, MSE; `sample` 2581-2608 Euler/Heun). Training: `LatentEMA` 5010-5052, `run_epoch_ae` 5121-5161, `run_epoch_latentfm` 5162-5260, `LatentFMAdapter` 7204-7537 (stage 1 AdamW+cosine eta_min 1e-6; stage 2 AdamW+cosine + `LatentEMA`), `visualize_reconstruction_latentfm` 6160-6477. Eval: `src/eval_latentfm_ensemble.py` (544) JHU canonical; `eval_kolm_ensemble.py` for 2D (`bundle.model.sample(cond, n_steps=nfe, ...)` at :701/:906). Fixes: `src/lfm_fixes.py` (A.0.5).

**Checklist — internal consistency only (no upstream):**

| # | Element | Ours | Compare with |
|---|---|---|---|
| 1 | Stage-1 AE (conv, GroupNorm/SiLU, L1+MSE? — check `run_epoch_ae` 5121-5161) | 1866-2069 | LDM AE has KL/VQ + LPIPS + GAN; audit §3: "a pure L1+MSE autoencoder with no KL/VQ/LPIPS/adversarial — which biases our own spectral-floor result in our favour and must be disclosed" |
| 2 | Latent scale: raw AE latents into N(0,I) flow | 2561-2577 | LDM `scale_factor`; fix = `latent_scale_mode` in `lfm_fixes.py:41-58` |
| 3 | Conditioning: masked sparse grid avg-pooled by 2^n_levels | 2536-2545 | `lfm_fixes.py:15-40` "FIX 1 -- conditioning attenuation (cond_mode: image_norm)" |
| 4 | Rectified-flow loss / Euler sampler | 2561-2608 | Liu et al. rectified flow (idea) |
| 5 | Optimiser/EMA | 7204-7537 | ours |

**Declared deviations / audit incidents.** Paper `main.tex:366-367` "audit incidents (a conditioning defect in the latent flow-matching baseline ...) were repaired or labeled". `lfm_fixes.py:1-58` gives the measured attenuation (cond std 0.0014-0.0085 vs field std 0.841) and the two opt-in fixes. **Which JHU LFM row (shipped vs `image_norm`) the paper reports** is decided by the run config that `eval_latentfm_ensemble.py:187-207` reads; not determinable here because `Save_TrainedModel/JHU/baseline_latent_fm` is empty. The 2D configs on disk use `cond_mode: image` with no `latent_scale_mode` (shipped numerics).

**Configs / launchers.** JHU: `train_baseline_lfm_xcube.sh` → `train_Gen_Baseline.py --config Save_config/config_baseline_Gen_xcube.yaml` (stage 1 then 2); eval `eval_latentfm_canonical.sh`. 2D: `latent_fm_params` block inside `Save_config/{kolmogorov2d,cylinder2d,cylinder2d_surface}/config_baseline_Gen_*.yaml` (`baseline_model: latent_fm`), as-run copies `Save_config/kolmogorov2d/gen_baseline/Baseline_latent_fm_Stage{1,2}_*.yaml`.

---

## A.7 CoNFiLD (Du et al., Nat. Commun. 2024)

**Upstream.** `github.com/jx-wang-s-group/CoNFiLD @ 449835e` (`Save_config/config_baseline_CoNFiLD_xcube.yaml:2-4`: "The SIREN auto-decoder and the latent-diffusion UNet are imported verbatim from that checkout"). The clone must sit at **`confild_params.upstream_root`** = `/work/hdd/bilr/ntricard/datasets/baselines/CoNFiLD` (`config_baseline_CoNFiLD_xcube.yaml:52`; default `DEFAULT_CONFILD_ROOT` `confild_upstream_core.py:15-17`; hard-coded `CONFILD_ROOT` in `confild_eval_unified.py:40-41`, `confild_conditional.py:23-24`, `confild_baseline.py:34-35`). Upstream modules imported (so **not** vendored — they run from the clone):
* `ConditionalNeuralField.cnf.nf_networks.SIRENAutodecoder_film` (`confild_upstream_core.py:44`, `confild_eval_unified.py:50`)
* `UnconditionalDiffusionTraining_and_Generation.src.script_util.{create_model, create_gaussian_diffusion}` (`confild_upstream_training.py:699`, `confild_eval_unified.py:51`)
* `ConditionalDiffusionGeneration.src.guided_diffusion.{condition_methods.get_conditioning_method, measurements.get_noise, gaussian_diffusion.create_sampler}` (`confild_eval_unified.py:43-49`)
Upstream line cites in our code: `scripts/train.py:400-401` (`confild_upstream_training.py:531`), `case4.yml` (config comments), `measurements.py:219` `Case4Operator._unnorm` (`confild_eval_unified.py:7`), `condition_methods.py:79-87` (`:10`), `condition_methods.py:31` (audit §15b).

**Two generations of our code — check the right one.** The paper numbers come from the **unified** path (`confild_eval_unified.py:194-200` refuses legacy `cnf_last.pt`: "retrain via train_confild_unified.slurm"). The legacy path (`confild_baseline.py`, `confild_stage2.py` with its own `TimeEmbedMLP` denoiser, `confild_conditional.py`) is superseded — note `confild_stage2.py:5-10` declares an MLP-ResNet replacing upstream's UNet, which the unified path reverted.

| File | Element | Lines | Kind |
|---|---|---|---|
| `src/confild_upstream_core.py` (363) | `import_upstream_decoder` 40-46; `FieldStatistics` (GLOBAL per-channel min/max; long justification) 57-138; `PackedJHUCubes` 139-210; `octahedral_transform` 211-243, `octahedral_gather` 244-297; `build_latent_windows` ("Arrange lumped codes as upstream [time, latent] diffusion images") 313-351 | | our glue around upstream classes |
| `src/confild_upstream_training.py` (968) | `train_stage1` 422-694: decoder + latent table, two Adams (466-467), per-epoch decoder step placement 531-535 ("matches upstream scripts/train.py:400-401"), per-batch latent step 561-569, MSE 565, random point subsample 545-549; `_import_upstream_diffusion` 695-700; `train_stage2` 703-959: `create_model(...)` 737-746, `create_diffusion(steps, noise_schedule="cosine")` 750, AdamW lr wd=0 751, EMA deepcopy 748-749, parameter-budget gate 758-772; `run_confild_training` 960-968 | | glue |
| `src/model_baseline.py` | `CoNFiLDAdapter` (custom lifecycle) | 8215-8270 | glue |
| `src/confild_eval_unified.py` (458) | `WindowSensorOperator` (decoder as measurement operator, row-chunk checkpointing) 68-110; `load_stage1` 194-215; `load_stage2` 217-241; `create_sampler(ddpm, cosine, epsilon, fixed_large, clip_denoised=True)` 311-314; `get_noise(sigma=0)` 315; `--dps-scale 1.0`, `--steps 1000` (argparse 257-258) | | glue |
| `src/confild_eval_unified2.py` (364) | TUNE-split sweeps, post-sensor correction, window1 arms ("Stage-B fix") | | our arms |
| `src/confild_stageA_fit.py`, `confild_select_and_final.py`, `confild_pick_best.py`, `confild_stage1_plateau.py`, `confild_stage2_ckpts.py`, `confild_split_summary.py` | selection/diagnostics | | ours |
| Legacy: `confild_baseline.py`, `confild_stage2.py`, `confild_conditional.py`, `evaluate_confild_*.py` | superseded | | ours |

**Checklist (upstream files live in the clone; ours cannot deviate in the imported classes, so check the *call sites* and the glue):**

| # | Element | Ours | Upstream |
|---|---|---|---|
| 1 | SIREN autodecoder class, args (`in_coord_features=3, in_latent_features, out_features, num_hidden_layers, hidden_features`) | `confild_upstream_training.py:454-461`, `confild_eval_unified.py:205-209` | `ConditionalNeuralField/cnf/nf_networks.py::SIRENAutodecoder_film` |
| 2 | Latent table zeros-init, lumped latent dim (tied to hidden unless `tie_latent_to_hidden: false`) | `:427-431`, `:465` | `case4.yml` |
| 3 | Two-timescale Adam: decoder lr 1e-4 stepped once per epoch on accumulated grads; latent lr 1e-5 per batch | `:466-467`, `:531-535`, `:566-569`; config `:71-73` "Upstream case4.yml verbatim" | `scripts/train.py:400-401`, `case4.yml` |
| 4 | Field normalisation: upstream per-point over time → ours GLOBAL per-channel min/max, normalise-then-augment | `confild_upstream_core.py:57-83`; `confild_upstream_training.py:435-452` | `Normalizer_ts(dim=0)` — **declared class-(ii) adaptation** (audit §12) |
| 5 | Random point subsample per item (`points_per_item: 16384`) | `:545-549`; config `:66-70` | upstream feeds every point — declared, "unbiased estimator" |
| 6 | Octahedral 48-group table expansion (`groups: 48`) | `:432-433`, config `:96-100` | upstream: none — declared |
| 7 | Stage-2 latent "image" windows `[window_length × latent]`, [-1,1] global scalar range | `confild_upstream_core.py:313-351`; `training.py:717-728` | upstream `UnconditionalDiffusionTraining...` data prep; `measurements.py:219` |
| 8 | Stage-2 UNet `create_model(image_size, num_channels, num_res_blocks, num_heads, num_head_channels, attention_resolutions="32,16,8", channel_mult)` | `:737-746`; config `:111-122` | `script_util.create_model` (openai guided-diffusion style) |
| 9 | Diffusion: 1000 steps cosine, epsilon, MSE; AdamW lr 5e-5 wd 0; EMA 0.9999 | `:750-751`; config `:107-108, 116, 123` | `create_gaussian_diffusion`, upstream train script |
| 10 | Guidance: DDPM sampler + `ps` conditioning scale 1.0, `clip_denoised=True`, `fixed_large` | `confild_eval_unified.py:311-315`, `get_conditioning_method` call in `main` | `condition_methods.py:79-87` PosteriorSampling; Case-4 notebook |
| 11 | Measurement operator = frozen decoder at sensor coords, row-wise per window row | `:68-110` | `measurements.py` Case4Operator |
| 12 | Window-joint reconstruction of 32 consecutive snapshots, each row its own sensors | docstring `:13-19` | upstream Case-4 arrangement (claimed) |
| 13 | Three arms: P published prior (118.9M, waived), C 1024-d capacity-matched, F 384-d faithful | `main.tex:370, 484-489, 991-993`; `confild_stageA_fit.py:40-44` | `case4.yml` sets latent=hidden=384 |

**Declared deviations (quoted).** Config `:64-70` (points_per_item), `:82-95` (capacity-matched 1024-d code, decoder width 384→256), `:96-100` (augmentation ×48). `confild_upstream_core.py:60-83` (global vs per-point statistics; "Class (ii) required adaptation"). Audit §3: "Latent LR restored to upstream 1e-5 ... the +/-10% rule was waived for P because it forces a 75x reduction in exactly the component under evaluation". Audit §12 records the normalisation bug and its fix; FLEET_AUDIT: "C≡P stage-1 bit-identity is BY DESIGN".

**Configs / launchers.** `train_confild_unified.slurm <config-basename>` → `train_Gen_Baseline.py --training-stage {1,2}`; only `Save_config/config_baseline_CoNFiLD_xcube.yaml` (arm C) is on disk; arms P/F configs missing. Eval `confild_eval_canonical.slurm <arm-save-root>` / `confild_eval_full.sh` → `confild_eval_unified.py`; Stage-A/B arms `sA*.sh`, `sB*.sh`.

---

## A.8 Gen4Turb (Oommen et al.)

**Upstream.** `github.com/vivekoommen/Gen4Turbulence` (MIT; data Zenodo 17088765 — `DATA_TRANSFER_TIER2.md:48`). Commit **`8af0a7b`** is recorded only in the deleted audit (§3: "model code byte-identical, deviation surface is 4 places"). **No vendored code at all**: the model runs from the clone at `/work/hdd/bilr/ntricard/datasets/baselines/Gen4Turbulence/3_flow_reconstruction/dm` (`gen4turb_eval.py:17-18`; `sys.path.insert(0, DM)` :34; `from utils.architecture import Unet`, `from utils.diffusion import ElucidatedDiffusion` :35-36). Training also ran **inside the clone**: `train_gen4turb_jhu.sh:14` `cd .../dm`, `train_gen4turb_uxuz.sh:13-15` runs `python train_model_uxuz.py` — a script that lives in the (untransferred) clone, **not in this repo**. Paper: `main.tex:491-494` "retrained under the benchmark's observation model (the most generous variant is reported)"; table row "Gen4Turb (anneal, 32-step)" `main.tex:994`.

**Our implementation.**

| File | Element | Lines |
|---|---|---|
| `src/export_for_baselines.py` (101) | writes `data/u.npy [4,T,nx,ny,nz]`, `MIN_u.npy`, `MAX_u.npy`; centre-crop to 120^3 because "Gen4Turbulence's 3D UNet asserts every spatial dim is divisible by 8" (:54) | 44-83 |
| `src/gen4turb_eval.py` (116) | builds `Unet(dim=16, dim_mults=(1,2,4,8), channels=4, self_condition=True, flash_attn=True)` + `ElucidatedDiffusion(..., image_size 120^3, sigma_data from Par.pkl)` 41-45; loads ckpt 48-51; **mask variants** `shared` (one 1 % mask on all 4 channels) vs `strict` (own 1 % on Ux,Uz; Uy,p unobserved) 72-82; conditioning `[x*mask, mask]` 83; `model.sample(cond)` 86-90 (seeded 88); converts to our z-score units and scores with `ensemble_metrics` 92-96 | |
| `src/benchmark_scaling_baselines.py` | `bench_gen4turb` ("their repo, unmodified") | 100-160 |
| `src/assemble_baseline_table.py:54-57` | table ingests `canon_anneal_4190_strict.json` then `canon_uxuz_4930_strict.json` ("Anneal arm supersedes the constant-LR 4930 run") | |
| Launchers | `train_gen4turb_jhu.sh`, `train_gen4turb_uxuz.sh`, `run_gen4turb_eval.sh`, `run_gen4turb_uxuz_eval.sh`, `run_gen4turb_all50.sh`, `run_scaling_baselines.sh` | |

**Checklist.**

| # | Element | Ours | Upstream |
|---|---|---|---|
| 1 | Network/diffusion classes | imported unchanged (`gen4turb_eval.py:35-45`) | `3_flow_reconstruction/dm/utils/{architecture,diffusion}.py` @ 8af0a7b — diff the clone against `8af0a7b` to confirm "byte-identical" |
| 2 | `dim=16` (6,157,076 params, -5.4 %) | `:41` | upstream default dim (check `train_model.py`) |
| 3 | Data layout / normalisation min-max per field | `export_for_baselines.py:44-83`; `gen4turb_eval.py:53-55, 68-70` | upstream loader (`crops [:, :, :256, :256]`, :12) |
| 4 | Conditioning = `[x*mask, mask]` 8 channels | `:83` | upstream training mask U(0,1) coverage shared over channels (`:3-5`) |
| 5 | **Protocol retrain "uxuz"** (mask only Ux/Uz) — `train_model_uxuz.py` | **not in this repo; not determined** | upstream `train_model.py` |
| 6 | **"anneal" arm** = upstream's scheduler actually stepped ("Upstream never calls scheduler.step() ... An annealing arm runs as a labelled deviation", audit §3; FLEET_AUDIT "anneal arm = 1 labelled scheduler.step line") | in the clone; **not determined here** | — |
| 7 | Sampling steps 32 (`main.tex:994` "32-step"); `model.sample` default | `:89` | ElucidatedDiffusion sampler defaults |
| 8 | Checkpoint choice: budget-matched epoch (4190 anneal / 4930), not upstream's spectral-error val-selected best (contaminated split) | audit §3, §8 | upstream selects `best_model.pt` on spectral error |

**Declared deviations.** `gen4turb_eval.py:3-9` (shared-mask = information advantage; strict variant "out-of-their-training-distribution"); `assemble_baseline_table.py:146-149` (report STRICT only). Paper `main.tex:493-494`. The "4 places" deviation surface is not itemised anywhere on this host — **ask for the clone's diff vs 8af0a7b** (`git -C .../Gen4Turbulence diff 8af0a7b`), which should show `train_model_uxuz.py`, the anneal `scheduler.step()` line, `Par.pkl`, and possibly the overwritten `models/best_model.pt` (FLEET_AUDIT item 9: "Gen4Turb tracked-file hygiene (Par.pkl, models/best_model.pt overwritten in upstream clone)").

---

## A.9 S3GM (Li et al., Nat Mach Intell 2024)

**Upstream.** `https://github.com/lzy12301/S3GM @ 2343293bf627e7b4afc0d63b11ad37ece0919653` (`src/s3gm3d.py:3`). Upstream files named by our section headers: `models/nn.py`, `models/fp16_util.py`, `models/rpe.py`, `models/unet_video.py`, `models/ema.py`, `sampler/sde.py` (`model_baseline.py:3106, 3235, 3291, 3440, 3835, 3884`); `trainer/loss.py::loss_fn_video` and `predict_fn` (`s3gm3d.py:110, 137`); `trainer/datasets.py` (`:122`, audit "datasets.py:82-98"); `sampler/utils.py::generate_parallel_2d` with `:706, :709, :727, :728` (`s3gm3d.py:149, 242, 261, 276`; audit §15); `train.py:46` (sigma_max) and `train.py` optimiser (`s3gm3d.py:553`).

**Two implementations — the 2D rows and the 3D row differ materially.**

*(a) Vendored model code (fork parent, "Consolidated from s3gm_core.py", `model_baseline.py:3097-3101`)* — audit §15: "Model code is a verbatim port (diff shows only th->torch, docstrings, reflow)":

| Ours (`src/model_baseline.py`) | Upstream file |
|---|---|
| `SiLU` 3108, `GroupNorm32` 3114, `conv_nd` 3120, `linear` 3131, `avg_pool_nd` 3136, `update_ema` 3147, `zero_module` 3153, `scale_module` 3160, `mean_flat` 3167, `normalization` 3174, `timestep_embedding` 3179, `CheckpointFunction` 3192, `checkpoint` 3221 | `models/nn.py` |
| `convert_module_to_f16` 3237 … `zero_grad` 3282 | `models/fp16_util.py` |
| `RPENet` 3293, `RPE` 3323, `RPEAttention` 3363-3441 | `models/rpe.py` |
| `TimestepBlock` 3442, `TimestepEmbedAttnThingsSequential` 3449, `Upsample` 3470, `Downsample` 3493, `ResBlock` 3511, `FactorizedAttentionBlock` 3585-3636, `UNetVideoModel` 3637-3832 (forward 3788-3814: obs/latent mask concat `x*(1-obs_mask)+x0*obs_mask`, indicator channel, `in_channels+1`) | `models/unet_video.py` |
| `ExponentialMovingAverage` 3837-3881 | `models/ema.py` |
| `SDE` 3886, `VESDE` 3952-3998, `VPSDE` 3999-4055 | `sampler/sde.py` |
| `S3GMLearnedGridDeformer` 4058-4088 | **ours** (Geo-FNO trick; unused by the adapters) |

*(b) 2D training/sampling path (fork parent; used for **Kolmogorov, cylinder, surface rows** "S3GM (PC-DPS)", `tab_kolm_body.tex:5`, `tab_cyl_body.tex:5`):* `s3gm_loss` 5053-5077 (T=1, `frame_indices=zeros`), `run_epoch_s3gm` 5080-5118 (`clip_grad_norm_(0.5)` + skip if >50 at 5105-5111), `S3GMAdapter` 7046-7202 (Adam lr/wd from config, **100-epoch warmup + cosine** 7087-7095, EMA, `VESDE(N=num_scales=1000)`), `_score_fn_*` 5741-5773, `dps_sample` 5776-5843 (Langevin corrector with `clamp(step_size, max=1.0)` 5804-5812; DPS `alpha_obs * sum(residual**2)` 5829-5840; clamp ±1e8), `visualize_reconstruction_s3gm` 5846-6035. Eval via `eval_kolm_ensemble.py:678-686, 888-894` with the run config's `sampling_N / snr / n_corrector_steps / alpha_obs`.

*(c) 3D path (ours; JHU rows):* `src/s3gm3d.py` (673): `volume_to_video` 78-86, `video_to_pointcloud` 88-93, `draw_slabs` 95-106, `s3gm3d_loss` 112-133 ("Upstream loss_fn_video, verbatim"), `_net_fn` 136-145 ("predict_fn with continuous=True"), `_langevin_corrector` 152-163, `s3gm_reconstruct` 166-287 ("Upstream generate_parallel_2d with the KSE/Kolmogorov/ERA5 notebook settings"), `s3gm_reconstruct_ensemble` 293-311, `obs_to_video` 313-321, `run_epoch_s3gm3d` 326-385 (no clipping except loose guard 362-370), `visualize_s3gm3d` 391-513, `S3GM3DAdapter` 518-673 (Adam betas (0.9,0.999) eps 1e-8 wd 0, **no schedule** 553-557; `VESDE(N=num_scales)`; sampling defaults `alpha_case 0.5, beta 0.4, snr 0.128, n_corrector 0` 562-570). `src/train_s3gm3d.py` (18) swaps the adapter into the registry. `src/eval_s3gm3d.py` (237): arms `jhu_tuned` (alpha_case 0.05/beta 0.004, "JHU-TUNED DEVIATION (NOT upstream)") vs `upstream` (0.5/0.4, "UPSTREAM-FAITHFUL (diverges at JHU scale)") at 104-113; `--n-steps` default = config `sampling_N`. `src/s3gm_norm_guidance.py` (339): `s3gm_reconstruct_norm` 81-250, `install` 276-299 — the "S3GM (normalized guidance)" row. Diagnostics `s3gm_isolate.py`, `s3gm_watchdog.py`, `s3gm_improve_common.py`.

**Checklist (3D path, open `s3gm3d.py` beside `S3GM/`):**

| # | Element | Ours | Upstream |
|---|---|---|---|
| 1 | Loss: `t~U(eps,T)`, `perturbed=mean+std*z`, `score*std+z` squared, mean over dims then batch | `s3gm3d.py:112-133` | `trainer/loss.py::loss_fn_video` |
| 2 | Masks during training: `obs_mask` zeros, `latent_mask` ones, `frame_indices=arange(T)` | `:122-126` | `trainer/datasets.py:82-98` |
| 3 | `predict_fn` continuous: `labels = marginal_prob(0,t)[1]` | `:136-145` | `trainer/loss.py::predict_fn` |
| 4 | VESDE `sigma_min 0.1`, **`sigma_max` (upstream 20, `train.py:46`)**, `N` = sampling step count | `S3GM3DAdapter:559-560`; config (missing) | `train.py:46`; notebooks `VESDE(..., N=outer_loop)` (`:186-188`) |
| 5 | Optimiser: Adam(lr, (0.9,0.999), 1e-8, wd 0), no schedule, no clipping | `:553-557`, `:362-370` | `train.py` |
| 6 | Window sampler `generate_parallel_2d`: `b = ceil((Nz-ol)/(t_win-ol))`, overlapping slabs, `x_to_sample` stitching, reverse-diffusion `f,G = sde.discretize`, `temp_u = temp_mean + G*z` | `:194-239` | `sampler/utils.py::generate_parallel_2d` |
| 7 | DPS term `alpha * sum((y - x0_hat)**2 * m)` (unnormalised) | `:241-246` | `sampler/utils.py:706` |
| 8 | Overlap-consistency `beta * sum((a - c)**2)` with `.detach()` on the left neighbour | `:248-259` | `sampler/utils.py:709` |
| 9 | Guard `clamp(dx, ±1e8)` and update `x = temp_u - dx` (no step size) | `:261-267` | `sampler/utils.py:727-728` |
| 10 | `alpha = alpha_case / sqrt(observed fraction)` (notebook rule) | `visualize_s3gm3d:423-425`; `eval_s3gm3d.py` | notebooks |
| 11 | Corrector = `NoneCorrector` (off by default) | `:152-163`, default 0 | notebooks pass `corrector=NoneCorrector` |
| 12 | EMA rate, `use_checkpoint` | `:558`, `:547` | `train.py` |
| 13 | z-axis as frame axis, `t_win=10`, `overlap=1`, `slabs_per_snapshot=16` | docstring 15-38; `:581, 594-595` | **declared adaptation** |

**Checklist (2D path) — every row below is a deviation from upstream that the audit fixed only in the 3D module (audit §15 table, quoted):**

| # | 2D ours (`model_baseline.py` / on-disk 2D configs) | Upstream | Audit class |
|---|---|---|---|
| 1 | `sigma_max: 7.0` (`Save_config/kolmogorov2d/config_baseline_S3GM_kolm.yaml:69`; same value in the two cylinder files) | 20 (`train.py:46`) | (i) |
| 2 | T=1, whole field, `frame_indices=zeros` (`5063-5065`) | `arange(T)` + windows | (i) |
| 3 | `clip_grad_norm_(0.5)` + skip if >50 (`5105-5111`) | no clipping | (i) |
| 4 | 100-epoch warmup + cosine (`7087-7095`) | no scheduler | (i) |
| 5 | lr 1e-4 / wd 1e-6 (config `:53-54`) | 2e-4 / 0 | (i) |
| 6 | `alpha_obs: 1.0` fixed (config `:77`; `dps_sample:5832`) | `alpha_case/sqrt(1-sparsity)` | (i) |
| 7 | 5 Langevin corrector steps (config `:76`) | `NoneCorrector` | (i) |
| 8 | `clamp(step_size, max=1.0)` in corrector (`5811`) | absent | (i) |
| 9 | step size from continuous schedule while `sde.N=1000` (`5791, 5800-5802`) | `VESDE(N=outer_loop)` | (i) |
| 10 | overlap-consistency loss absent | `utils.py:709` | (i) |

The 2D S3GM configs on disk (`kolmogorov2d/config_baseline_S3GM_kolm.yaml:49-78`, `cylinder2d/config_baseline_S3GM_cyl.yaml`, `cylinder2d_surface/config_baseline_S3GM_cyl_surface.yaml`, identical `s3gm_params`) carry all ten. The paper's S3GM paragraph (`main.tex:495-499`) only discusses the 3D adaptation. **Decide tonight whether the 2D "S3GM (PC-DPS)" rows are to be described as a pre-audit port.**

**Declared deviations (quoted).** `s3gm3d.py:15-38` (z as time; "ZERO changes to upstream model code are required"; anisotropy caveat). `eval_s3gm3d.py:49-53, 100-113` ("'jhu_tuned' = alpha_case 0.05 / beta 0.004 (OUR DEVIATION, stable at JHU scale). 'upstream' = alpha_case 0.5 / beta 0.4 (upstream-faithful, DIVERGES at JHU scale -- run only to document the divergence)"). `s3gm3d.py:417-422` monitor override. `s3gm_norm_guidance.py:1-60` (normalised guidance, "WHY THIS EXISTS"). Paper `main.tex:1077-1088`, table `main.tex:995-997`. Audit §24: "both of upstream's published guidance settings diverge independently at our problem size".

**Configs / launchers.** JHU: `s3gm_train.sh` → `train_s3gm3d.py --config Save_config/config_baseline_S3GM_xcube.yaml` (**missing**; audit §15 says nf=32, ch_mult (1,2,3,4), 6,348,228 params, T_WIN 10); eval `s3gm_eval.sh` / `s3gm_finalize.sh` → `eval_s3gm3d.py --arm {jhu_tuned,upstream}`. 2D: `s3gm_params` in `Save_config/{kolmogorov2d,cylinder2d,cylinder2d_surface}/config_baseline_S3GM_*.yaml` (`baseline_model: s3gm`, nf 128, ch_mult [1,2,4,8]) via `submit_2d_fleet_delta.sh`; as-run copy `Save_config/kolmogorov2d/gen_baseline/Baseline_s3gm_Stage1_DemoN63_*.yaml`; eval `eval_kolm_fleet.sh`.

---

## A.10 SiT / SiT-point (Ma et al. 2024)

**Upstream.** SiT (github.com/willisma/SiT). Commit **`cbde832`** recorded only in the deleted audit §3 ("SiT-point — upstream cbde832 ... transport lib byte-identical"). Nothing in `src/` names the repo or commit; `SiTPhysics` docstring (`model_baseline.py:2805-2823`) says "unchanged from the original SiT" and lists the changes. **Assume you are checking against SiT HEAD unless you can confirm cbde832.**

**Our implementation.**

*Vendored (fork parent; byte-identical to cosmos2w merge-base and `upstream/main`):* `src/sit_transport/{__init__.py (64), transport.py (461), path.py (191), integrators.py (117), utils.py (28)}` ↔ upstream `SiT/transport/{__init__.py, transport.py, path.py, integrators.py, utils.py}`.

*Re-typed from `SiT/models.py` (fork parent):* `_sit_modulate` 2617-2619 (`modulate`), `SiTTimestepEmbedder` 2625-2651 (`TimestepEmbedder`), `SiTBlock` 2657-2684 (`SiTBlock`), `SiTFinalLayer` 2690-2706 (`FinalLayer`), `_sit_get_2d_sincos_pos_embed_nonsquare` 2779-2787 / `_sit_get_1d_sincos_pos_embed` 2790-2798 (`get_2d_sincos_pos_embed`, `get_1d_sincos_pos_embed_from_grid`), `SiTPhysics` 2804-3022 (`SiT`: `__init__`, `initialize_weights` 2899-2933, `unpatchify` 2935-2948, `forward` 2950-3022), `SiTEMA` 3056-3092 (cf. `train.py::update_ema`).

*Ours / fork-parent additions:* `SiTPointTokenEmbedder` 2712-2755 and `SiTPointFinalLayer` 2758-2773 (point tokenizer; not in SiT), `SiTLearnedGridDeformer` 3028-3050 (unused), `run_epoch_sit` 5261-5406 (spike-skip logic 5377-5400), `SiTAdapter` 7538-7867 (param groups, AdamW betas (0.9,0.95), eps from config, 200-epoch warmup + cosine 7645-7673, `create_transport` 7675-7679), `sit_conditional_sample` 6478-6506 (`Sampler(transport).sample_ode(sampling_method, num_steps)`), `sit_conditional_sample_points_chunked` 6510-6561 (chunked random-permutation sampling for point tokens), `visualize_reconstruction_sit` 6565-6801. Eval: `src/eval_sit_ensemble.py` (194) JHU; `eval_kolm_ensemble.py:715, 920` 2D.

**Checklist (open `model_baseline.py` + `src/sit_transport/` beside `SiT/models.py` + `SiT/transport/`):**

| # | Element | Ours | Upstream |
|---|---|---|---|
| 1 | `transport/transport.py` `Transport`, `Sampler`, `training_losses` | `sit_transport/transport.py:40-460` | `SiT/transport/transport.py` — **`training_losses` has an added `huber_beta` argument (`:122, :142-150`)**; upstream is plain `mean_flat((model_output - ut)**2)`. This edit predates our fork (byte-identical to cosmos2w). |
| 2 | `path.py` (ICPlan/VPCPlan/GVPCPlan), `integrators.py` (sde/ode), `utils.py`, `__init__.create_transport` | whole files | upstream same names |
| 3 | adaLN-Zero modulate, timestep embedder (freq 256, MLP SiLU) | 2617-2651 | `models.py::modulate`, `TimestepEmbedder` |
| 4 | `SiTBlock`: LN(no affine, eps 1e-6), timm `Attention(qkv_bias=True, **qk_norm=True, norm_layer=nn.LayerNorm**)`, Mlp GELU-tanh, adaLN 6×hidden | 2657-2684 | `models.py::SiTBlock` — upstream `Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)`; **`qk_norm=True` is a deviation** (comment 2664-2667; audit: "Class-(iii), unchanged: qk_norm=True") |
| 5 | `FinalLayer` | 2690-2706 | `models.py::FinalLayer` |
| 6 | `PatchEmbed(in_chans = in_channels + cond_channels)` widened input; fixed sin-cos pos-embed (non-square) | 2858-2873, 2908-2913 | `models.py` `PatchEmbed(in_chans=in_channels)`, `get_2d_sincos_pos_embed` (square) |
| 7 | Weight init (xavier on Linear, patch-embed xavier, t-embed N(0,0.02), zero adaLN + zero final) | 2899-2933 | `models.py::initialize_weights` — upstream also inits `y_embedder` (removed here) |
| 8 | No class label / `y_embedder`, `learn_sigma=False`, out_channels=in_channels | 2816-2822 | `models.py` |
| 9 | Conditioning: `[x, cond_feat, obs_mask, any_mask]` channel concat before patchify (`cond_channels = 2*n_fields+1`, 7624) | 2971-2991 | none upstream (class-conditional) — **declared** |
| 10 | Point tokenizer (Fourier features, MLP), per-token final layer | 2712-2773, 3000-3022 | none upstream — **declared** ("SiT-point") |
| 11 | Training loss: velocity, Linear path, `huber_beta` (2D configs 0.1; JHU 0.0 = "restore the original SiT MSE velocity loss", `config_baseline_SiT_xcube.yaml:67`) | `run_epoch_sit:5365-5371` | `transport.training_losses` MSE |
| 12 | Optimiser: AdamW (0.9,0.95) eps 1e-6, wd 1e-4 (decay/no-decay groups), 200-ep warmup + cosine; `clip_grad_norm_(0.5)` + spike-skip | 7645-7673, 5377-5400 | `SiT/train.py`: AdamW lr 1e-4, wd 0, no schedule, no clipping |
| 13 | EMA 0.9999 | `SiTEMA`, 7674 | `train.py::update_ema` decay 0.9999 |
| 14 | Sampler: ODE **Euler**, 50 steps (2D) / 32 (JHU) | 6491-6494; configs `sampling.ode_solver: euler`, `sampling_N` | `SiT/sample.py` defaults `dopri5`, 250 steps (`Sampler.sample_ode` defaults `transport.py:360-363`) — **declared Euler-32 deviation** |
| 15 | Chunked sampling over random permutation, 8192 tokens | 6510-6561 | none — declared (`main.tex:499-507`) |

**Declared deviations (quoted).** `model_baseline.py:2805-2823` ("The only difference is a widened patch-embed input ... Changes from the original SiT: No class label embedding; in/out channels = n_fields; learn_sigma disabled; non-square grids; optional cond_channels"). `:2664-2667` (qk_norm rationale). `:6525-6541` (chunk incoherence, random-permutation chunks). Audit §3: "Class-(iii), unchanged: qk_norm=True, the AdamW/warmup/cosine package, grad clip + spike-skip, 32-step Euler vs upstream 250-step dopri5"; "Both known bugs genuinely fixed in the trained weights (randperm sampling :6457; huber_beta=0.0 giving upstream MSE)". Paper `main.tex:499-507`, capability footnote `:421-422` (patch-tokenised in 2D, point-tokenised in 3D). Note: `Save_config/kolmogorov2d/config_baseline_S3GM_kolm.yaml:125` and the SiT 2D configs keep `huber_beta: 0.1` (2D rows are Huber, JHU is MSE).

**Configs / launchers.** JHU: `train_sit_xcube.sh` → `train_Gen_Baseline.py --config Save_config/config_baseline_SiT_xcube.yaml` (on disk: pointnet tokenizer, hidden 256, depth 6, heads 4, node_subsample 8192, Euler 32); matched arm `train_sit_xcube_matched.sh` with `config_baseline_SiT_xcube_matched.yaml` (**missing**); eval `eval_sit_matched_seeded.sh` / `eval_sit_seeded_array.sh` → `eval_sit_ensemble.py`. 2D: `Save_config/{kolmogorov2d,cylinder2d,cylinder2d_surface}/config_baseline_SiT_*.yaml` and `_p4` variants (patch 8 / 4, hidden 256, depth 8), as-run `Save_config/kolmogorov2d/gen_baseline/Baseline_sit_Stage1_*.yaml`; wing `config_baseline_SiT_wing.yaml` (not PoF).

---

## A.11 DMF-Gen / PointCloudFFM — note only

`src/Model.py` (`PointCloudFFM` at 2024 region), `src/model_baseline.py:1567-1828` (`PointCloudFFM`, `FNOFFM`), `src/phycoflow_pointcloud/models/portable_core.py`, `src/train_pointcloud_ffm.py`, `src/ensemble_eval.py` — this is the fork parent's own method (`main.tex:508-516` "We evaluate it as published"). Not analysed here; **covered by Section B (fork diff against cosmos2w/PhyCoFlow_demo, merge-base `828c3c6`, `upstream/main` f163ad7)**.

## A.12 RectifiedFlow (github.com/gnobitab/RectifiedFlow) — the three references

`src/model_baseline.py:1572`, `src/Model.py:2024`, `src/phycoflow_pointcloud/models/portable_core.py:2354` — each is the same four-line docstring "Core 1-RF idea: (https://github.com/gnobitab/RectifiedFlow) 1) Draw a source sample x0 ~ prior 2) Draw a target sample x1 from data 3) Interpolate linearly 4) Train the velocity model to predict x1 - x0" inside the DMF-Gen `PointCloudFFM` class (present in the fork parent at the same place, `828c3c6:...model_baseline.py:1569`). **No code was taken from that repo** — it is an idea citation for the 1-rectified-flow objective; LFM's `training_loss` (`model_baseline.py:2561-2577`) implements the same four lines independently. Nothing to diff.

---

## A.13 Consolidated "not determined / needs the other host" list

1. Configs missing on disk (A.0.3): JHU Senseiver, S3GM, DeepONet, DeepONet++, FNO3D, SiT-matched, CoNFiLD P/F. Recover from Engaging `Save_config/` or the run dirs' `run_config.yaml` before checking hyper-parameters.
2. `Save_TrainedModel/JHU/*` empty here — no `run_config.yaml`, `final_summary.json`, or eval JSONs for JHU rows.
3. Gen4Turb: `train_model_uxuz.py`, the anneal `scheduler.step()` edit, `Par.pkl`, and the "4 places" deviation surface live in the untransferred clone `.../baselines/Gen4Turbulence/` — need `git diff 8af0a7b` there.
4. SiT and Gen4Turb commits (`cbde832`, `8af0a7b`) are only in the deleted audit; nothing in `src/` pins them. Suggest adding them to `model_baseline.py:2611` and `gen4turb_eval.py:1` headers once verified.
5. Whether the reported JHU LFM row is the shipped `cond_mode: image` model or an `image_norm`/`latent_scale_mode` model (`lfm_fixes.py`) — determined by the run config on the other host.
6. Senseiver 2D rows ran the pre-audit layout (A.2 finding) — a wording decision for the paper, not a code question.
7. S3GM 2D rows carry the ten pre-audit deviations (A.9 table) — same.
8. Upstream `file:line` numbers quoted in our comments (`positional.py:18-22`, `model.py:33-43`, `sampler/utils.py:706-728`, `deeponet.py:121-123`, `condition_methods.py:79-87`, `measurements.py:219`, …) are from the ports' clones at the recorded commits; if you clone HEAD instead, expect drift and search by symbol name.


---

# Section B — Our fork against cosmos2w/PhyCoFlow_demo, change by change

Repo: `/work/hdd/bilr/ntricard/PhyCoFlow_demo`. All paths below are relative to
`0_demo_TurbulentCombustion/` unless they start with `/` or `.gitignore`. Line numbers are
HEAD (`41203e0`) line numbers, verified with `git blame`; still valid at `8aa01c9` (the 2026-09-22 pull changed none of the 16 source/config files below — only `.gitignore`, see B.3.14).

Classification tags used throughout:
`[3D/JHU generalization]`, `[bug fix]`, `[protocol/leakage/integrity guard]`, `[new feature/arm]`,
`[HPC portability]`, `[performance/memory]`, `[cosmetic]`, `[merge kept older upstream code]`
(explained in B.1). Hunks that change the numerics of DMF-Gen / PointCloudFFM itself carry a bold
**BEHAVIOUR CHANGE** marker. Where the commit trail gives no reason I say
**REASON NOT DETERMINED — ask Nick** rather than guess.

---

## B.0 How to reproduce this comparison yourself

```bash
cd /work/hdd/bilr/ntricard/PhyCoFlow_demo

# 1. Upstream remote (already present; harmless to re-run)
git remote add upstream https://github.com/cosmos2w/PhyCoFlow_demo.git 2>/dev/null
git fetch upstream

# 2. Confirm the fork point. Expected: 828c3c6b52a863ca8d15d6dcfdfecec833df9226
git merge-base HEAD upstream/main
git log -1 --format='%h %ad %an %s' --date=short 828c3c6      # 2026-06-05 cosmos2w "Updated ffm config with ENH"
git log -1 --format='%h %ad %an %s' --date=short upstream/main # f163ad7 2026-08-17 cosmos2w "Merged all three coherence terms"
git log -1 --format='%h %ad %an %s' --date=short HEAD          # 8aa01c9 2026-09-22 "Record Engaging run configs ..." (41203e0 when B was written)
git rev-list --count 828c3c6..HEAD           # 139 (ours; 123 at 41203e0)
git rev-list --count 828c3c6..upstream/main  # 11 (Jason's, we do not have them)

# 3. Which shared files did WE modify since the fork point (17; 3 yaml covered in another section)
git diff --diff-filter=M --stat 828c3c6 HEAD
git diff --diff-filter=D --name-only 828c3c6 HEAD   # 0_demo_TurbulentCombustion/README.md
git diff --diff-filter=A --name-only 828c3c6 HEAD | wc -l   # 2431 (668 at 41203e0; +1730 force-added Engaging run outputs, +21 configs/scripts, see Section D)

# 4. Per-file diff (what to read while following B.3 below)
git diff 828c3c6 HEAD -- 0_demo_TurbulentCombustion/src/Model.py
# ...same for helpers.py, train_pointcloud_ffm.py, etc.

# 5. Per-file commit trail (which of our commits touched the file, newest first)
git log --format='%h %ad %s' --date=short 828c3c6..HEAD -- 0_demo_TurbulentCombustion/src/Model.py

# 6. Who introduced a specific line range in HEAD (used for every hunk below)
git blame -s -L1521,1553 HEAD -- 0_demo_TurbulentCombustion/src/Model.py
git log -L1521,1553:0_demo_TurbulentCombustion/src/Model.py --format='%h %ad %s' --date=short

# 7. What Jason did since the fork point (B.6)
git log --format='%h %ad %an %s' --date=short 828c3c6..upstream/main
git diff --stat 828c3c6 upstream/main
git diff 828c3c6 upstream/main -- 0_demo_TurbulentCombustion/src/Model.py   # etc.

# 8. Full commit body of any commit cited below
git show --stat --format='%h %ad %an%n%s%n%n%b' --date=short cf42d1b
```

Reference docs consulted for the "why" (all in `0_demo_TurbulentCombustion/` unless noted):
`HANDOFF_2026-09-12.md` (sec. 2 protocol invariants, sec. 7 uncommitted-change ledger),
`DELTA_STATUS_2026-09-08.md` (sec. 1 portability fixes, sec. 20+ incident log),
`HPC_MIGRATION_AND_REMAINING_WORK.md` (sec. 3.4 gotchas, sec. 4 protocol invariants), and
`/eval_infra_audit.patch` at repo root (the 2026-08-29 audit patch, commit 060c9a5 -> merged as 693fbd3).
Note: `README.md`, `HANDOFF.md`, `BASELINE_AUDIT_2026-08-28.md`, `FLEET_AUDIT_2026-08-29.md` and 10
other markdown/pptx/pdf files were deleted in 464c72e ("Removed old MD files", 2026-09-08); they are
still readable via `git show 464c72e^:0_demo_TurbulentCombustion/<name>`.

---

## B.1 Read this first: why some of "our" hunks look like reverting Jason's work

Our history is not a straight line off 828c3c6. Nick forked in early May 2026 (first fork commit
817654b, 2026-05-03 "Adapted FFM and baseline models to JHU dataset"), worked for a month on a
3-D/JHU port, and then **merged** upstream 828c3c6 into that fork on 2026-06-05 (merge commit
7c5e94a). The merge message records the resolution policy:

> Conflict resolutions (6 files), favoring local 3D work where logic overlaps while keeping upstream's
> new features ... `train_pointcloud_ffm.py` / `evaluate_ffm.py`: kept the local 3D versions
> (MetricsLogger + run_reconstruction_benchmark + 3D metrics/visualization) and ported upstream's
> GL_rbf_ENH backbone wiring (CLI args + _build_model branch) onto them.

Consequence: in `git diff 828c3c6 HEAD` some hunks show upstream code being *removed* that Jason
added between 2026-04 and 2026-06 (e.g. `TrainingHistoryLogger` and the `Evaluation/` run-dir layout
from 7ce6a86 2026-05-28; the RAM-checkpoint / `--run-dir` / `--ode-solver` /
`--obs-consistency-compare-modes` evaluator features from 8125691 / 576804e). `git blame` on those
lines points at *older upstream* commits (8856b6e, 7ec3ce1, 556e5b6, e225883), because our merge kept
the older text. I tag those `[merge kept older upstream code]`; they are not deliberate reverts, and
Nick should decide tonight whether any of them should be restored (see B.6 for what depends on them).

Pre-merge fork commits that appear in blame: 817654b, f8ad92f, cab5e18, b50c4d0, bc6c61c, d396d4b,
0ca1294, 6580e18, d005de1, 756f2b7, c93c08a, 985d961 (all May 2026, one-line messages, no bodies).

Post-merge campaign commits that appear in blame: 7a3f03e (Jun 22), cf42d1b (Aug 25 "ICLR campaign
state" — a very large squash; its body lists the features but not per-hunk reasons), de74834, c8e888a,
38f42bb, 0d303fa, 988bc86, 693fbd3 (merge of 060c9a5 audit), e5ee749 (Aug 30 checkpoint squash),
e7745ad, b3b4163, 3790338 (Sep 16 Delta port).

---

## B.2 The BEHAVIOUR-CHANGE list (what Nick most needs to see)

These are the hunks that change what the DMF-Gen / PointCloudFFM method computes, versus upstream
828c3c6, in decreasing order of how likely they are to matter for a number in the paper. Each is
detailed in B.3 with its commit.

| # | File:lines | What | Active in PoF campaign configs? |
|---|---|---|---|
| 1 | `helpers.py:584` | `torch.randint(...)` for the per-field sensor **count** lost `device=device` → drawn from the **CPU** RNG stream instead of CUDA (7a3f03e). Changes the sensor index draw relative to upstream even for identical seeds, because the CUDA Philox offset no longer advances before `randperm`. `HPC_MIGRATION...md` sec. 4 confirms "helpers_baseline's CUDA-randint variant is NOT equivalent — ~2% index overlap". Our canonical fingerprint (`idx_sum 37987162596`, `ensemble_eval.py:41`) was computed with *our* draw. | YES — every sensor draw |
| 2 | `Model.py:2081-2088` | Flow-matching time `t` can be **logit-normal** instead of uniform (`model.t_sampling`, cf42d1b). | YES — `t_sampling: "logit_normal"` in `kolmogorov2d/`, `cylinder2d/`, `config_iclr_jhu_xcube*.yaml` |
| 3 | `train_pointcloud_ffm.py:332-381, 1190-1207, 1323-1336` | **EMA** shadow weights (`ema_decay`), **bf16 autocast** (`use_amp`), **torch.compile** of the backbone forward. Checkpoint stores both raw and EMA; `ensemble_eval.py:125-133` evaluates the EMA weights when present. | YES — `ema_decay: 0.9995`, `use_amp: true`, `compile_model: true` |
| 4 | `Model.py:1521-1553, 1607-1614, 2196-2271` | Inference-only **ODE step cache**: sensor branch + top-K gather + coord-type readout computed at ODE step 1 and reused for steps 2..n when `gather_mode == "topk_rbf_glres"` (cf42d1b, de74834). Claimed exact (the cached terms do not depend on `t` or `x_t`); gated by `tests/test_ode_cache.py`. Floating-point-identical up to kernel ordering. | YES — `gather_mode: topk_rbf_glres` everywhere |
| 5 | `Model.py:994-1026, 1045-1046, 1058-1059` | Masked latent<-sensor cross-attention rewritten as per-item unmasked attention over valid keys (flash-eligible). Mathematically identical to `key_padding_mask`; only differs when an item has **zero** valid sensors (old: NaN; new: residual/FF only). | YES |
| 6 | `train_pointcloud_ffm.py:445-531` | `sample_query_subset` `obs_mix` mode rewritten: no duplicate-suppression, weights clamped to `finfo.tiny` instead of 0, chunked `cdist` (6580e18, 7a3f03e). Loss-point distribution differs from upstream in obs_mix mode. | NO — campaign uses `query_sampling: "uniform"`, which is unchanged |
| 7 | `Model.py:2102-2124` + `train_pointcloud_ffm.py:655-683` | Optional binned **spectral loss** on an appended grid block (`spectral_weight>0`), with Hann window option (cf42d1b, 0d303fa). | NO for PoF (`spectral_weight: 0.0`); YES in ICLR ablation arms `*_spec*.yaml`, `*_kprior.yaml` |
| 8 | `train_pointcloud_ffm.py:960-976`, `evaluate_ffm.py:181-190` | New source priors `rff_powerlaw` / `rff_kolmogorov` (`spectral_prior.PowerLawRFFPrior`, 0d303fa). | NO for PoF (`prior: "rff"`); YES in `config_iclr_jhu_xcube_kprior.yaml` |
| 9 | `Model.py:1421-1440, 746-763` | Optional second, coarser gather scale (`gather_multiscale`, zero-init). | NO (only `config_iclr_jhu_xcube_ms*.yaml`) |
| 10 | `Model.py:809-815` | `field_embed` table sized by `n_obs_field_types` (may exceed `n_fields`). No effect unless set. | NO (wing dataset only) |
| 11 | `helpers.py:261-282` | `JHU_SPLIT_MODE=block` temporal split with gap; `round()` boundary fix (cf42d1b, b3b4163). Changes which frames are train vs. test. | YES — evals must export `JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0` (HANDOFF sec. 2) |
| 12 | `helpers.py:571-590` | `valid_mask=` restricts sensor placement to a pool (surface task). Default `None` reproduces the canonical draw bit for bit (3790338; unit-tested per DELTA_STATUS sec. 1). | Only cylinder surface arms |
| 13 | `helpers.py:288-329, 374-390` | `JHU_AUGMENT` exact symmetry augmentation on the train split (octahedral / so3 / translate / pdatum / reflect_y) via `augment_symmetry.py` (cf42d1b). | NO unless env var set (ICLR `*_aug`/`*_sym` arms) |
| 14 | `train_pointcloud_ffm.py:628-654` | Training-time measurement operators (noise / occlusion / channel dropout) via `--measurement-ops` (cf42d1b). | NO for PoF; `config_iclr_jhu_robust.yaml`, firebench v4/v5 |
| 15 | `evaluate_ffm.py` (whole `main`) | Dropped upstream's `--ode-solver` and obs-consistency plumbing from the evaluator; `visualize_reconstruction` now gets `ode_solver=None` (defaults) `[merge kept older upstream code]`. Canonical numbers come from `ensemble_eval.py`, which imports only `_build_model`/`_normalize_eval_config` from this file. | Indirectly YES (`_build_model` is how every eval rebuilds the model) |

---

## B.3 Per-file walkthroughs (14 non-yaml shared files, in order of importance)

### B.3.1 `src/Model.py` (+235 / -61)

Commit trail: `3790338` (Sep 16), `0d303fa` (Aug 26), `de74834` (Aug 25), `cf42d1b` (Aug 25),
`7a3f03e` (Jun 22). Nothing from the May fork touched Model.py — Nick's May work left the model
untouched and only the campaign changed it. Only `ConditionalPointHybridLocalGlobalRBF` (the GL_rbf /
GL_rbf_ENH backbone), `PointCloudFFM` and `FNOFFM` are modified; `ConditionalPointMLPRBF`,
`ConditionalPointPerceiver`, `FNO`, priors are byte-identical to 828c3c6.

**Hunk M1 — ctor signature, `ConditionalPointHybridLocalGlobalRBF.__init__` (lines 677-679, 695).**
Adds `gather_multiscale=False`, `gather_topk_coarse=64`, `gather_coarse_sigma_scale=4.0`,
`n_obs_field_types=None`. `[new feature/arm]`, commit cf42d1b. All default-off; see M2, M3, M7.

**Hunk M2 — `_ode_cache = None` and multiscale init (lines 746-763).** Declares the inference cache
slot (see M8/M9) and, when `gather_multiscale`, a zero-initialised `nn.Linear(cond_dim, cond_dim)`
(`ms_coarse_out`) so enabling it on an existing checkpoint is a no-op at load. `[new feature/arm]`,
cf42d1b. The commit body itself calls the multiscale gather "(unused)"; it is exercised only by
`config_iclr_jhu_xcube_ms.yaml` / `_ms_aug.yaml` (ICLR ablation arms, not PoF).

**Hunk M3 — `field_embed` sized by `n_obs_field_types` (lines 809-815).** `nn.Embedding(n_fields, ...)`
becomes `nn.Embedding(self.n_obs_field_types, ...)`. Reason (inline comment): observations may carry
field ids beyond the reconstructed channels, e.g. wall shear stress observed on a wing surface while
only volumetric fields are generated. `[new feature/arm]` for the SHIFT-WING dataset, cf42d1b. With the
default `None` it is exactly `n_fields`. **BEHAVIOUR CHANGE** only if set (it changes the embedding
table shape, hence checkpoint compatibility).

**Hunk M4 — new `_cross_attn_valid` (lines 994-1026).** `[performance/memory]`, cf42d1b. Replaces
`key_padding_mask` attention with per-batch-item unmasked attention over that item's valid sensors.
Docstring: any key-padding mask forces SDPA onto the CUTLASS backward, measured ~70x slower than flash
at the 128-query x 39k-key shape (`test_attn_mechanism.py`). When all sensors are valid (`m.all()`)
it calls the block exactly as before. **BEHAVIOUR CHANGE (edge case only):** if an item has zero valid
sensors the old code produced NaN (softmax over all-masked keys); the new code returns
`x + attn.ff(attn.norm_ff(x))`. Otherwise mathematically identical (LayerNorm on kv is per-token, so
row-filtering commutes with it — verified at `CrossAttentionBlock.forward`, lines 308-331).

**Hunk M5 — `_encode_latents` uses `_cross_attn_valid` (lines 1045-1046, 1058-1059).** Both the
initial latent<-sensor cross-attention and the Senseiver-style re-injection between latent blocks
route through M4. `[performance/memory]`, cf42d1b. The `sensor_padding_mask` variable is still computed
at line 1042 but is now unused `[cosmetic]`.

**Hunk M6 — `@torch.compiler.disable` on `_knn_search_keops` (lines 1197-1201).** `[HPC portability]`,
3790338. Reason in the inline comment and DELTA_STATUS sec. 1: on DeltaAI (aarch64, torch 2.14)
dynamo mis-parses the KeOps LazyTensor formula (`IndexError` in pykeops `complete_aliases`), so the kNN
stays eager while the rest of the backbone compiles. Commit body: "no numerical effect".

**Hunk M7 — coarse gather branch in `_aggregate_chunk` (lines 1421-1440).** When `gather_multiscale`,
draws `k_coarse` neighbours, weights them with `sigma * gather_coarse_sigma_scale`, and adds
`ms_coarse_out(coarse_cond)` to `local_cond`. `[new feature/arm]`, cf42d1b. Zero-init projection, so
identical output at init. **BEHAVIOUR CHANGE** only when enabled (ICLR `_ms` arms).

**Hunk M8 — ODE-cache read path at top of `forward` (lines 1521-1553).** `[performance/memory]`,
cf42d1b (lines 1521-1540, 1546-1553) + de74834 (lines 1541-1545, the coord-type readout cache).
`_cache` is non-None only when (a) `self._ode_cache` is a dict (set by `PointCloudFFM.sample`, see
M12), (b) `not self.training`, (c) `gather_mode == "topk_rbf_glres"`. On a cache hit the sensor tokens,
latents, global summary, refined sensor features, importance bias and `local_cond` are reused; if
`query_readout_type == "coord"` the readout output is reused too (it depends only on coords and
latents: `_build_query_readout_tokens`, lines 1085-1090, never touches `point_feat`). Only
`point_feat` (which carries `x_t`, `t`), `coarse_pred` and the head are recomputed. Justification in the
inline comment: everything on the sensor side depends on `(coords, obs_*)` only, and in glres mode the
gather logits use `d2` and the importance bias, never `point_feat`. **BEHAVIOUR CHANGE (claimed
exact):** gated by `src/test_ode_cache.py` (added in cf42d1b). Nick may want to re-run that test on
GH200: `python src/test_ode_cache.py`.

**Hunk M9 — ODE-cache write path (lines 1607-1614).** Fills the cache after the first full pass (and
the coord readout output, de74834). Same classification/commit as M8.

**Hunk M10 — `PointCloudFFM.training_loss` signature (lines 2074-2078).** Adds
`compute_metrics=True` (7a3f03e), `spectral_block_shape`, `spectral_weight`, `spectral_bins` (cf42d1b),
`spectral_window` (0d303fa). `compute_metrics=False` skips the `.detach().cpu()` metric floats, which
force a host sync every step — `[performance/memory]` ("uniform is 10x faster" commit).

**Hunk M11 — logit-normal time sampling (lines 2081-2088).** `t = sigmoid(randn)` when
`getattr(self, "t_sampling", "uniform") == "logit_normal"` (attribute set by the trainer at
`train_pointcloud_ffm.py:1190`). Inline comment cites Esser et al. (SD3). `[new feature/arm]`,
cf42d1b. **BEHAVIOUR CHANGE — active in every PoF and ICLR DMF-Gen config
(`t_sampling: "logit_normal"`).** Upstream 828c3c6 always uses `torch.rand`. This is a training-time
distribution change; the sampler is unaffected.

**Hunk M12 — binned spectral loss and metric return (lines 2102-2124).** If `spectral_weight > 0` and
a block shape is given, computes `x1_hat = x_t + (1-t)·v` on the last `n_blk` query points (the
appended grid block, see T-hunks), calls `spectral_loss.binned_spectral_loss(..., window=spectral_window)`
weighted by `t`, and adds it to the MSE. `[new feature/arm]`, cf42d1b; `window` added by 0d303fa whose
body explains the bug it fixes: the JHU cutouts are non-periodic, an unwindowed FFT puts a broadband
leakage floor in the top shells (~100x their true power) which made the loss ~7x weaker than intended;
`spectral_window` defaults to **off** "so frozen results stay reproducible". **BEHAVIOUR CHANGE** only
when `spectral_weight > 0` (ICLR `_spec`, `_spec02`, `_specwin`, `_kprior` arms; PoF uses 0.0).

**Hunk M13 — `PointCloudFFM.sample` wrapped in try/finally with `_ode_cache` (lines 2193-2273).**
Sets `self.model._ode_cache = {}` before the Euler/Heun loop and resets it to `None` in `finally`. The
loop body (Euler / Heun / endpoint consistency / `default_hard` clamp / final clamp) is **re-indented
but textually unchanged** — compare with `git diff -w`. `[performance/memory]`, cf42d1b. See M8 for
the exactness claim.

**Hunk M14 — `FNOFFM.training_loss` `compute_metrics` (lines 2296, 2327-2328).** Same early-return as
M10 for the grid FNO baseline. `[performance/memory]`, 7a3f03e.

Not changed but worth knowing: upstream (B.6) later added a Fourier positional encoding option to
`ConditionalPointPerceiver` and a print-once guard in the same class we modified (a972f92, 85b8e59);
those will merge cleanly except for the print statement at our lines 766-767.

### B.3.2 `src/helpers.py` (+352 / -58)

Commit trail: 3790338, b3b4163, 693fbd3, 988bc86, cf42d1b, 7a3f03e, 7c5e94a(merge), 985d961, 756f2b7,
b50c4d0, f8ad92f, 817654b.

**Hunk H1 — `TurbulentCombustionH5Dataset.__init__` gets `sensor_pool` (lines 213, 222-227) and
loads the surface pool (lines 238-258).** With `sensor_pool="surface"` the H5's `surface_indices`
dataset (or the `.surface_indices.npy` sidecar written by `add_grid_surface_indices.py` when the H5
was locked by running jobs — DELTA_STATUS sec. 1) becomes a boolean `valid_sensor_mask [N]` exposed per
sample. `[new feature/arm]` — the cylinder surface-to-field task (HPC_MIGRATION sec. P0.5,
SURFACE_TASK_PREREGISTRATION.md), commit 3790338.

**Hunk H2 — temporally-blocked split (lines 261-282).** `JHU_SPLIT_MODE=block` puts a contiguous
validation block at the end separated by `JHU_SPLIT_GAP` frames (default 50) instead of the upstream
shuffled split; the gap frames are discarded. Reason (inline): consecutive DNS frames correlate at
r~1.0 so a shuffled split leaks. `[protocol/leakage/integrity guard]`, cf42d1b. Lines 267-270 are
b3b4163's `[bug fix]`: `n_val` computed with `int()` under-counted by one at `train_ratio 0.8`
(`1.0-0.8 = 0.19999...`), silently putting the first held-out frame into train; now `round()`. Commit
body: "JHU 0.75 unaffected; exact in binary". Default when the env var is unset is still `"shuffle"`
(= upstream behaviour); HANDOFF sec. 2 notes the eval drivers refuse to run without the export.
**BEHAVIOUR CHANGE** (data split), active in all campaign evals.

**Hunk H3 — symmetry augmentation setup (lines 287-329) and application in `__getitem__`
(lines 374-390).** `JHU_AUGMENT` (comma list) enables exact symmetries on the train split via the new
`augment_symmetry.py`: `octahedral` (grid-safe, also given to baselines), `octahedral_proper`, `so3`,
`translate`, `pdatum` (pressure offset), `reflect_y` (FireBench). `AUG_GRID_SHAPE` for non-cubic grids.
`[new feature/arm]`, cf42d1b. Off unless the env var is set (ICLR `_aug`/`_sym` arms).

**Hunk H4 — stats loading `weights_only=True` -> `False` (line 342).** Commit 817654b ("Adapted FFM
and baseline models to JHU dataset", no body). Loosens the pickle restriction on
`dataset_stats.pt`. The stats file written by this same class contains only two tensors, so the strict
load should work; **REASON NOT DETERMINED — ask Nick** (plausibly a stats file written by another
tool or torch version on the JHU path). Not numerical. `[cosmetic]` / security-hygiene note.

**Hunk H5 — `__getitem__` returns shared tensors instead of clones (lines 391-398).** `coords`,
`coords_raw`, `physical_time` no longer `.clone()`d (817654b; inline comment "immutable shared tensor,
stacked in collate"). `[performance/memory]` — at 125^3 points a clone is ~23 MB per sample per worker.
Also appends `valid_sensor_mask` when present (3790338, H1). Note `helpers_baseline.py:283-292` still
clones (kept separate deliberately so baseline behaviour is untouched).

**Hunk H6 — `MetricsLogger` split into `log_csv` / `plot_history`, CSV gains `epoch_time_s`,
`peak_gpu_mem_mb`, `cumul_train_time_s` (lines 416-476).** b50c4d0 ("Updated baselines"). Cost
instrumentation feeding the paper's cost columns (HPC_MIGRATION sec. 4: "Cost fields ... required on
every row"). `[new feature/arm]` (instrumentation). Note this whole class is code that upstream deleted
in 7ce6a86 (replaced by `TrainingHistoryLogger` inside the trainer); we kept and extended the old one
`[merge kept older upstream code]`.

**Hunk H7 — `create_recon_dir` path `ffm_tc_pointcloud/` -> `Recon/` (line 481).** cf42d1b.
`[cosmetic]` run-dir layout (`<run>/Recon/demo_N<k>_<ts>/Epoch_<e>/`). Upstream's layout is
`<run>/Evaluation/epoch_XXXX/`.

**Hunk H8 — `build_sparse_condition`: `return_counts`, `valid_mask`, and the randint device change
(lines 513-514, 571-603).**
- `return_counts=True` returns the per-item valid sensor count so the query sampler can slice
  `obs_coords[b, :n_valid]` without a boolean gather (7a3f03e, `[performance/memory]`).
- `valid_mask` restricts the draw to a pool: `randperm` over the pool under the same seeding; `m`
  capped at pool size (3790338, `[new feature/arm]`, surface task). Docstring: "Default None reproduces
  the canonical draw bit for bit (no extra RNG consumption)" — DELTA_STATUS sec. 1 says unit-tested.
- **Line 584: `m = int(torch.randint(low=nmin, high=nmax + 1, size=(1,)).item())` — the `device=device`
  argument was dropped (7a3f03e).** Upstream draws the count on the CUDA generator, we draw it on the
  CPU generator. Consequence: under the protocol seeding `torch.manual_seed(seed*777+snap)` the CUDA
  Philox offset is no longer advanced by the randint before `randperm(n_pts, device=cuda)`, so **our
  sensor index draw is not the same permutation as upstream's for the same seed**, even when
  `nmin == nmax`. **BEHAVIOUR CHANGE — protocol-defining.** `HPC_MIGRATION...md` sec. 4 documents
  exactly this: "helpers_baseline's CUDA-randint variant is NOT equivalent — ~2% index overlap", and the
  canonical fingerprint `idx_sum` in `ensemble_eval.py:41` is of *our* draw. All campaign numbers
  (ours and every baseline routed through `helpers.build_sparse_condition`) are internally consistent;
  they are just not reproducible with upstream's helpers. The reason in 7a3f03e is speed (a CUDA
  `randint(size=(1,)).item()` is a device sync per field per item). Nick should decide whether to
  record this explicitly in the paper's protocol appendix.

**Hunk H9 — new `build_sparse_condition_from_pool` (lines 607-658) and `append_extra_tokens`
(lines 661-683).** Sensors drawn from a dedicated observation point set with its own channels (wing
surface pool), and always-valid parameter tokens (flow conditions) appended to the observation tuple.
`[new feature/arm]` SHIFT-WING, cf42d1b. Unused by PoF.

**Hunk H10 — `_save_single_field_plot` (lines 700-702, 717-748, 785-812).** Adds `triang`,
`metric_true_f`, `metric_pred_f`; reports full-volume L2 in the title alongside the slice L2; **1-99
percentile colour clip** instead of min/max; narrower figure with dedicated colourbar axes; smaller
sensor markers. Commits: 985d961 (May, figure layout), 988bc86 (volume-vs-slice metric, sensor overlay;
body explains: sliced visualisers reported slice-L2 while unsliced ones reported volume-L2, so the same
field in `metrics.json` meant different things across models), 693fbd3 (percentile clip; inline comment:
raw min/max makes truth and reconstruction look smoother and more alike than they are — "the single
change that most affects whether these figures can be trusted"). `[bug fix]` (metric semantics) +
`[cosmetic]`. Does not affect any number in a results JSON other than the plot title.

**Hunk H11 — `save_smooth_mask_plot` z-midplane slice (lines 936-952).** `[bug fix]`, 693fbd3 (audit
060c9a5): `payload["coords_xy"]` is the full volume projected on (x,y); for 3-D it superposed all
z-planes into a smear. Now slices the middle z-level. Plot only.

**Hunk H12 — `visualize_reconstruction`: `field_names` kwarg, `import inspect`, 3-D midplane
slicing, per-field sensor overlay restricted to the slice, volume metric passed separately
(lines 1116, 1129-1130, 1197-1236).** f8ad92f / 756f2b7 (May: "no longer collapse 3D points") +
988bc86. `[3D/JHU generalization]`. Metrics returned are full-volume (`metric_true_f=truth_phys[:, c]`),
plots are the middle z-plane. The local `import inspect` at 1129 is redundant with the module-level
import at line 14 but is used at 1175 (`inspect.signature(model.sample)` — upstream's own kwarg-probing
refactor, kept by the merge per 7c5e94a's body) `[cosmetic]`.

**Hunk H13 — duplicate `"ode_solver"` key in the metrics JSON dict (line 1268).** f8ad92f. The dict
literal at 1262-1270 has `"ode_solver": ode_solver` twice; Python keeps the last, harmless.
`[cosmetic]`, a merge artefact.

### B.3.3 `src/train_pointcloud_ffm.py` (+616 / -286)

Commit trail: 3790338, e5ee749, 0d303fa, c8e888a, cf42d1b, 7a3f03e, 7c5e94a, d005de1, 6580e18,
d396d4b, f8ad92f, 817654b.

**Hunk T1 — imports (lines 15-63).** Drops `csv`, `h5py`, `matplotlib`, `F`; adds `time`,
`Sequence`, `datetime`, `ShiftWingDataset`/`collate_wing`, `MetricsLogger`, `append_extra_tokens`,
`build_sparse_condition_from_pool`, `create_recon_dir`, `apply_measurement_operators`. Mostly cf42d1b;
`MetricsLogger` import is d396d4b (May). `[cosmetic]` + `[merge kept older upstream code]` (upstream
removed `MetricsLogger` in 7ce6a86).

**Hunk T2 — `--data` default `Dataset/...` -> `../Dataset/...`, `--save-dir` f-string dropped,
`--field-names` added (lines 64-71).** 817654b. Launchers `cd $SLURM_SUBMIT_DIR` and run
`python train_pointcloud_ffm.py` from inside `src/` (e.g. `train_bench_kolm_ffm.sh:17-24`), hence the
`..`. `--field-names` lets the JHU/Kolmogorov/cylinder H5s (which are not the 5-field combustion
layout) name their channels. `[HPC portability]` / `[3D/JHU generalization]`.

**Hunk T3 — `--backbone` gains `fno3d` (line 76).** e5ee749. `[3D/JHU generalization]`, see T15.

**Hunk T4 — spectral / multiscale CLI (lines 138-162).** `--spectral-weight/-block/-bins`,
`--gather-multiscale`, `--gather-topk-coarse`, `--gather-coarse-sigma-scale`. cf42d1b.
`[new feature/arm]`.

**Hunk T5 — FNO3D CLI (lines 215-220).** `--Num-z`, `--fno-modes-z`, `--fno-domain-padding`.
e5ee749. `[3D/JHU generalization]`.

**Hunk T6 — prior choices and spectral window CLI (lines 255-263).** `--prior` adds
`rff_powerlaw`, `rff_kolmogorov`; `--prior-slope/-k-min/-k-max`; `--spectral-window`. 0d303fa.
`[new feature/arm]`.

**Hunk T7 — training-efficiency and dataset CLI (lines 298-328).** `--use-amp`, `--t-sampling`,
`--compile-model`, `--ema-decay`, `--measurement-ops` (JSON), `--sensor-pool` (3790338),
`--dataset {jhu,shiftwing}`, `--processed-root` (default is a Delta path — `[HPC portability]`).
cf42d1b except where noted. `[new feature/arm]`.

**Hunk T8 — `class EMAWeights` (lines 332-381).** fp32 shadow of every floating parameter,
`update` via `lerp_`, `copy_to`/`restore` for evaluating with EMA weights, `load_state_dict` keeps
shadow entries for params added after the checkpoint (multiscale branch) and moves loaded shadows to the
live device. cf42d1b. `[new feature/arm]`. **BEHAVIOUR CHANGE — active (`ema_decay: 0.9995`):** the
canonical evaluator (`ensemble_eval.py:125-133`) loads `ckpt["ema"]["shadow"]` when present, so the
reported model is the EMA model, not the raw weights that `best.pt`'s `val_loss` was computed on.

**Hunk T9 — `collate_snapshots` stacks `valid_sensor_mask` (lines 434-442).** 3790338, surface task.

**Hunk T10 — `sample_query_subset` rewrite (lines 445-531).** Three commits:
- 6580e18 (May 14, lines 498-509): chunk the `cdist` from every grid point to the sensors
  (`obs_mix_chunk_size=65536`) — the full `[1.95M x n_obs]` matrix is infeasible in 3-D.
  `[performance/memory]`.
- 7a3f03e (Jun 22, lines 464-493, 521-526): takes `obs_counts` instead of `obs_mask` so it can slice
  `obs_coords[b, :n_valid]` without a boolean gather; **removes the `selected` bookkeeping** that
  prevented duplicate picks across the near/far/uniform pieces and the refill step; `take_weighted`
  now clamps weights to `finfo.tiny` (so zero-weight points can be drawn with vanishing probability
  instead of being excluded); `finalize_indices` deliberately does *not* `torch.unique` ("CUDA unique
  has variable-size output and triggers host synchronization. Rare duplicates are cheaper than a
  per-batch sync"). `[performance/memory]`. **BEHAVIOUR CHANGE in `obs_mix` mode:** the loss-point
  set can contain duplicates and the near/far weighting differs slightly from upstream. **Not active**
  in any PoF/ICLR DMF-Gen config (all use `query_sampling: "uniform"`, whose path — `randperm[:n].sort()`
  — is unchanged).
- d005de1 (May 18, line 456): `obs_mix_chunk_size` default.

**Hunk T11 — `run_epoch` (lines 534-704).** Returns `(loss, epoch_time_s, peak_mem_mb)` (d396d4b /
cf42d1b; `[new feature/arm]` cost instrumentation); resets CUDA peak stats; `non_blocking` transfers;
three observation paths: (a) wing pool + measurement ops + parameter tokens (cf42d1b), (b) H5 with
`valid_mask` from the batch (3790338) and `return_counts=True` (7a3f03e), then training-time
measurement operators if configured (cf42d1b, `[new feature/arm]`); passes `obs_counts` to the query
sampler; **appends a contiguous grid block for the spectral loss** (lines 655-665, cf42d1b: "taking it in
grid-index space keeps it a genuine cube under the symmetry augmentations"); wraps `training_loss` in
`torch.autocast(bfloat16, enabled=use_amp)` (cf42d1b, **BEHAVIOUR CHANGE — active**, forward/loss in
bf16); passes `compute_metrics=False` (7a3f03e) and the spectral kwargs (`spectral_window` 0d303fa);
`ema.update(model)` after each optimizer step (cf42d1b). Validation calls use `spectral_weight=0.0`.

**Hunk T12 — new `run_reconstruction_benchmark` (lines 707-739) replaces the inline benchmark loop
that upstream had in `main` (old lines 967-1010).** f8ad92f (May) + cf42d1b. Same logic factored out,
plus `field_names=args.field_names`. `[cosmetic]` / `[3D/JHU generalization]`. `[merge kept older
upstream code]` for the `Recon/Epoch_<e>` layout.

**Hunk T13 — upstream `TrainingHistoryLogger` class deleted (old lines 527-596).** This is the
inverse of upstream 7ce6a86; our merge kept `helpers.MetricsLogger`. `[merge kept older upstream code]`,
7c5e94a. Not a functional loss (same CSV/PNG, different columns/paths), but note the run-dir layout now
differs from upstream: ours writes `Loss_DemoN<k>_<ts>/` + `Recon/`, upstream writes
`loss_history.{csv,json,png}` + `Evaluation/`.

**Hunk T14 — RELOAD prefers `last.pt` over `best.pt` (lines 833-852).** f8ad92f. Upstream resumed
from `best.pt`; ours tries `last.pt` first so a resumed run continues from where it stopped rather than
from the best-val epoch. `[bug fix]` (resume semantics) — no body; the reason is self-evident from the
code. Also drops upstream's `shutil.copy(config_path, save_dir/"run_config.yaml")` (old line 676-677)
`[merge kept older upstream code]` — **note `evaluate_ffm.py` in upstream relies on `run_config.yaml`
for `--run-dir` mode; ours does not have that mode either (see E-hunks).**

**Hunk T15 — dataset construction (lines 861-912).** `shiftwing` branch (cf42d1b) else H5 with
`field_names` (817654b) and `sensor_pool` (3790338); `persistent_workers`/`prefetch_factor` when
`num_workers > 0` (cf42d1b, `[performance/memory]`, comment: reduce epoch-boundary stalls). Lines
866-874 keep the old upstream layout text (blame 8856b6e) `[merge kept older upstream code]`.

**Hunk T16 — spectral grid-shape inference and prior selection (lines 935-976).** Infers a cube from
`num_points` or reads `AUG_GRID_SHAPE`; disables the spectral term if no grid; builds
`PowerLawRFFPrior` for `rff_powerlaw`/`rff_kolmogorov` (0d303fa body: the single-lengthscale RFF prior
held only 4.7% of its energy above k=8 while the data reach k=62 — "leading explanation for smooth
samples on unobserved channels"). `[new feature/arm]`. Default `rff` path unchanged.

**Hunk T17 — `GL_rbf_CQ` backbone branch (lines 1007-1013).** c8e888a: routes to `model_cq.build_cq_model`
(vendored upstream portable package `src/phycoflow_pointcloud/`, v0.9.0-pre1, commit 27f7308). `[new
feature/arm]` — only `config_iclr_jhu_xcube_cq.yaml` uses it.

**Hunk T18 — GL_rbf ctor gets multiscale args and `n_obs_field_types` (lines 1078-1080, 1099).**
cf42d1b. See M1-M3.

**Hunk T19 — FNO grid diagnostics prints removed (old lines 846-876) and `grid_info` unused.**
Upstream f33967e added these prints; our merge dropped them. `[merge kept older upstream code]`,
`[cosmetic]`.

**Hunk T20 — `fno3d` backbone branch (lines 1129-1175).** e5ee749: `fno3d_backbone.FNO3D/FNO3DFFM`
("faithful upstream neuraloperator core, rfftn spectral conv, our sensor-conditioning API, full 125^3
grid"), with parameter/mode accounting prints. `[3D/JHU generalization]` (an architecture-ablation
baseline, not DMF-Gen).

**Hunk T21 — param count print, `t_sampling`, `torch.compile`, EMA, AMP, measurement-ops prints
(lines 1183-1207).** cf42d1b (d396d4b for the param count). `model.model.forward = torch.compile(...,
mode="max-autotune-no-cudagraphs")` compiles the bound forward so state-dict keys are unchanged.
**BEHAVIOUR CHANGE — active (`compile_model: true`)**; numerically this is kernel fusion, not a model
change, but it is why M6 exists.

**Hunk T22 — RELOAD state loading (lines 1209-1239).** `strict=False` with an allow-list: only missing
keys containing `ms_coarse_out` are tolerated (zero-init multiscale branch), anything unexpected raises;
optimizer state incompatibility is caught and the optimizer restarts; EMA state reloaded; a
reconstruction benchmark is re-run for the reloaded epoch if it is a `save_every` multiple (f8ad92f).
cf42d1b / f8ad92f. `[bug fix]`/`[new feature/arm]`.

**Hunk T23 — epoch loop (lines 1242-1343).** Unpacks the 3-tuple from `run_epoch`; passes spectral,
measurement, AMP, EMA kwargs; prints time and peak memory; checkpoint dict gains `ema`, `Num_z`,
`fno_*`, `condition_blur*` (e5ee749 lines 1306-1315); benchmark only when `args.dataset == "jhu"` and
**with EMA weights swapped in** (cf42d1b); `logger.log_csv(...)` + periodic `plot_history()`
(d396d4b / cf42d1b). `[new feature/arm]`.

### B.3.4 `src/evaluate_ffm.py` (+658 / -427; largest rewrite by line count)

Commit trail: e5ee749, 0d303fa, c8e888a, cf42d1b, 7c5e94a, c93c08a, bc6c61c, 817654b. The bulk is
bc6c61c (May 12 "Updated FFM evaluation script") and c93c08a (May 20 "Updated visualization and adapted
evaluation code to 3D") — both pre-merge, one-line messages. Two things to hold in mind:
(1) the canonical numbers do **not** come from this script's `main`; `ensemble_eval.py` (new file)
imports only `_build_model` and `_normalize_eval_config` from it (`ensemble_eval.py:30`), so those two
functions are load-bearing and the rest is the legacy single-snapshot evaluator; (2) upstream's later
additions to this file (RAM checkpoints, `--run-dir`, `--config-path`, `--ode-solver`, obs-consistency
compare modes, FNO grid prints) are **absent** in ours because the merge kept the local version.

**Hunk E1 — module docstring and Matplotlib setup (lines 1-28).** Drops the "Standalone evaluator for
base PointCloudFFM and RAM-finetuned checkpoints / Loading structure" paragraph
`[merge kept older upstream code]`; adds `MPLCONFIGDIR=/tmp/phycoflow_mplconfig` and `matplotlib.use("Agg")`
(c93c08a) `[HPC portability]` — headless compute nodes with a read-only home.

**Hunk E2 — imports (lines 32-38).** Drops `validate_regular_grid_compatibility`, adds
`reconstruct_snapshot` (bc6c61c). `[3D/JHU generalization]`.

**Hunk E3 — CLI (lines 55-79).** `--Demo-Num` required again (was optional with `--run-dir`);
`--run-dir`, `--config-path`, `--ode-solver` removed `[merge kept older upstream code]`;
`--n-steps-generation` default 2 -> 4 (blame: upstream e225883 — upstream itself changed 4 -> 2 in
51650cf on 2026-06-04, the day before the fork point, and our merge kept 4 `[merge kept older upstream
code]`; PoF happens to report DMF-Gen at NFE 4 per HANDOFF sec. 3, but canonical evals pass
`--n-steps` explicitly to `ensemble_eval.py`, so this default is inert); `--extra-metrics` help says "2D or 3D"; upstream's SSIM /
grad interpretive comments removed `[cosmetic]`. The `--obs-consistency-*` flags (lines 81-100) are
still parsed but no longer used in `main` (dead arguments) `[cosmetic]`.

**Hunk E4 — helpers `_extract_demo_num`, `_resolve_demo_path`, `_load_config_file` deleted
(old lines 162-183).** `[merge kept older upstream code]`. These supported `--run-dir`.

**Hunk E5 — `_build_prior` power-law branch (lines 181-190).** 0d303fa. Must mirror T16 so a
checkpoint trained with `rff_kolmogorov` rebuilds the same source. `[new feature/arm]`.

**Hunk E6 — `_build_model`: `fno3d` (lines 242-262, e5ee749) and `GL_rbf_CQ` (263-272, c8e888a)
branches; `n_obs_field_types` passed to GL_rbf (322-323, cf42d1b).** `[3D/JHU generalization]` /
`[new feature/arm]`. **Load-bearing for every eval** via `ensemble_eval.py`. Note the GL_rbf branch
still reads config keys with plain `cfg.get(key, default)`; upstream 85b8e59 changed these to
`_cfg_get_not_none` so a YAML `null` falls back to the enhanced default — a small divergence Nick may
want (B.6).

**Hunk E7 — `_infer_structured_grid` (lines 348-473) + compatibility wrapper
`_infer_structured_grid_from_coords` (476-496).** 2-D-only grid recovery generalised to 3-D
(`lexsort((x, y, z))`, `nz`, `dz`), returning `ndim`. bc6c61c / c93c08a. `[3D/JHU generalization]`.

**Hunk E8 — `_reshape_flat_field_to_grid` 3-D (lines 499-503).** bc6c61c.

**Hunk E9 — `_gaussian_kernel`/`_ssim` 2-D/3-D with `_ssim2d` alias (lines 511-570).** bc6c61c /
c93c08a. `[3D/JHU generalization]`. Same constants (C1, C2, 11-tap window).

**Hunk E10 — `_gradient_metrics` 3-D branch (lines 572-610).** Adds z-gradient to
`grad_mse`, `grad_rel_l2`, `h1_rel`. bc6c61c.

**Hunk E11 — `_radial_spectrum` 3-D branch and Nyquist truncation (lines 630-728).** 3-D shell
average (bc6c61c); lines 686-692 and 729-732 (cf42d1b) truncate shells at the per-axis Nyquist in both
2-D and 3-D: "corner shells (|k| up to sqrt(3) x Nyquist) are populated by a handful of modes with
essentially zero physical energy, and band ratios computed over them are spurious". `psd2` renamed
`psd`. `[bug fix]` (analysis metric) — **changes `spectral_*` extra-metric values relative to
upstream** (band-energy ratios no longer include corner shells). Note 0d303fa later moved all paper
spectra to `spectral_utils.shell_spectrum` (Hann-windowed), so this function is legacy.

**Hunk E12 — `_spectral_metrics`, `_band_energy_breakdown` signature/`dz` (lines 776-830).**
bc6c61c. Plumbing for E11.

**Hunk E13 — new `_save_3d_slice_plots` (lines 904-1026).** Three orthogonal mid-plane slices
(XY/XZ/YZ), each Truth / Reconstruction / |Error|. bc6c61c + c93c08a. `[3D/JHU generalization]`,
plot only.

**Hunk E14 — `main` run resolution (lines 1033-1053).** Only the `--Demo-Num` path remains
(`_find_latest_yaml` on `Save_config/pointcloud_ffm/`); RAM `source_config` handling, `run_config.yaml`
lookup and `demo_root`-relative data resolution are gone. `[merge kept older upstream code]`.
`model_root` is `demo_root / save_dir.parent / f"{save_dir.name}_DemoN{k}_{ts}"`.

**Hunk E15 — dataset construction (lines 1063-1071).** Data path default `../Dataset/...`
(817654b, `[HPC portability]`, run from `src/`); `field_names` from cfg (bc6c61c). Drops upstream's
FNO grid prints. Checkpoint is now loaded **after** the model is built (upstream built after loading to
avoid holding both on device — the comment was removed) `[cosmetic]`; the `_metadata` key strip is kept.

**Hunk E16 — output dir `Save_reconstruction_files/ForOfflineEvaluation/eval_N...` ->
`<model_root>/Evaluation/eval_N<k>_<ts>_from_<train_ts>` (line 1117).** bc6c61c. `[cosmetic]`; all
artefacts now live under the run directory (matches the DELTA layout where `Save_TrainedModel/` is a
symlink to `/work/hdd/.../Save_TrainedModel`).

**Hunk E17 — `main` evaluation body (lines 1124-1318).** Detects 3-D from the first sample's
`coords_raw`; the 3-D path calls `helpers.reconstruct_snapshot` (no plotting through
`visualize_reconstruction`), computes per-field full-volume relative L2 in physical units, writes
`snapshot_XXXX_metrics.json`, infers the grid, saves E13 slices; the 2-D path calls
`visualize_reconstruction` **without** `ode_solver` (defaults to `None`) and without obs-consistency
kwargs; extra metrics (SSIM/grad/spectrum) are computed uniformly for 2-D/3-D. bc6c61c / c93c08a.
`[3D/JHU generalization]` + `[merge kept older upstream code]` (compare-modes loop and `SenConsis`
comparison table removed). **BEHAVIOUR CHANGE for this legacy script only:** the 3-D path has no
obs-consistency clamp choice — `reconstruct_snapshot` uses whatever `visualize_reconstruction`'s defaults
are (`default_hard`). Irrelevant to canonical numbers (`ensemble_eval.py` handles clamping itself via
`scatter_observed_values`).

**Hunk E18 — summary JSON (lines 1319-1334).** Drops `ode_solver` and the six obs-consistency keys,
adds `is_3d`. bc6c61c. `[cosmetic]`.

### B.3.5 `src/helpers_baseline.py` (+259 / -27)

Commit trail: 3790338, b3b4163, 693fbd3, 988bc86, 38f42bb, cf42d1b, 7c5e94a, c93c08a, d005de1, f8ad92f.
This is the baselines' copy of the dataset/plot helpers; upstream's `build_sparse_condition` here
already had `valid_mask` (556e5b6, April) and the CUDA `randint`, so **the baselines' own sensor draw
is unchanged from upstream** — but the campaign routes every baseline's canonical eval through
`helpers.build_sparse_condition` (HPC_MIGRATION sec. 4), not this one.

**Hunk HB1 — imports/`__all__` (lines 13, 47, 55, 65, 67).** `time`; exports `pointcloud_to_grid3d`,
`midplane_slice`, `grid3d_to_pointcloud`, `build_obs_grid_mask3d`. cf42d1b / f8ad92f / 988bc86.

**Hunk HB2 — `TurbulentCombustionH5Dataset`: `sensor_pool` (lines 133, 142-146, 157-177).**
Identical to H1, 3790338. `[new feature/arm]`.

**Hunk HB3 — block split + `round()` fix (lines 180-201).** Identical to H2 (cf42d1b, b3b4163).
`[protocol/leakage/integrity guard]` + `[bug fix]`. HPC_MIGRATION sec. 4: "fixed in both helpers.py and
helpers_baseline.py".

**Hunk HB4 — octahedral / reflect_y augmentation for the grid baselines (lines 206-233, 277-282).**
cf42d1b. Comment: "Same exact octahedral augmentation the point-cloud model gets, so the baselines are
trained under identical data conditions ... continuous SO(3) ... deliberately withheld rather than
approximated." Uses `augment_octahedral.py` (a separate, older module) and `augment_symmetry.reflect_axis_augment`.
`[new feature/arm]` — fairness of the `_aug` arms.

**Hunk HB5 — `__getitem__` returns `valid_sensor_mask.clone()` when present (lines 283, 290-292).**
3790338.

**Hunk HB6 — `pointcloud_to_grid3d` (lines 1168-1173), `grid3d_to_pointcloud` (2252-2258),
`build_obs_grid_mask3d` (2304-2346).** f8ad92f (May). 3-D grid <-> point-cloud utilities for the
latent-FM baseline. `[3D/JHU generalization]`.

**Hunk HB7 — `_build_structured_triangulation` "zcollapse-ok" annotation (line 1290).** 693fbd3.
`[cosmetic]` — the audit's `check_no_zcollapse.py` greps for this marker.

**Hunk HB8 — new `midplane_slice` (lines 1323-1349).** 988bc86. Picks the exact middle z-level (the
older `|z - median| == min` convention selected *two* planes for an even level count — verified 4608 vs
2304 points at n=48; harmless at 125^3 only because the median lands on a level). `[bug fix]`, plot only.

**Hunk HB9 — `_save_single_field_plot` (lines 1616, 1641-1670, 1709-1716, 1724, 1747-1753).**
Same as H10 (988bc86 metric semantics + sensor overlay that was accepted and never drawn; 693fbd3
percentile clip; c93c08a layout). `[bug fix]` + `[cosmetic]`.

**Hunk HB10 — `_save_car_surface_field_plot` sensor overlay removed (old lines 1630-1631,
1735-1741).** Blame: these lines exist at 828c3c6 and are absent in HEAD. Committed in c93c08a. No
message body. **REASON NOT DETERMINED — ask Nick** (possibly that the car dataset's sensor coords are
in a different frame; irrelevant to PoF).

**Hunk HB11 — `visualize_reconstruction` `BENCH_SAMPLE=1` timing (lines 1983-1986, 2000-2004).**
cf42d1b. Prints sample wall time and peak memory for the cost table. `[new feature/arm]`
(instrumentation).

**Hunk HB12 — new `nearest_sensor_fill_nodes` (lines 2364-2400).** 38f42bb: Voronoi-style
conditioning for the point-token SiT (with a random node subsample, sensors are generally not members of
the token set, so exact scatter would silently discard the conditioning). `[new feature/arm]` (SiT-point
baseline).

### B.3.6 `src/model_baseline.py` (+1342 / -219)

Commit trail: 3790338, e7745ad, e5ee749, 988bc86, 38f42bb, cf42d1b, 7c5e94a, 756f2b7, 0ca1294,
b50c4d0, cab5e18. This file is the baselines only; nothing here touches DMF-Gen numerics. Grouped by
theme rather than strictly by position.

**Group MB-A: 3-D latent flow matching (cab5e18, May 7 "Updated 3D model-baselines"; 0ca1294 May 12
"Fixed baseline stage training issue").** `[3D/JHU generalization]`.
- `_ResBlock3d`, `ConvAE3D` (lines 1993-2069): 3-D conv autoencoder with padding to a multiple of
  `2^n_levels`.
- `_AdaGNResBlock3D`, `LatentFMUNet3D` (2359-2460): 3-D velocity UNet on the AE latent volume.
- `LatentFlowMatching` (2476-2591): `spatial_dim` from the AE class; 3-D only supports
  `cond_mode="image"`; `_pad_to_ae`, `_downsample_mask`, `_latent_hw`, `_encode_condition`,
  `training_loss` (`t_` view generalised), `sample` shape generalised.
- `run_epoch_ae` / `run_epoch_latentfm` (5121-5259): `num_z` branch, 3-D grid masks, 3-D `obs_coords_2d`
  (`ix, iy, iz`).
- `LatentFMAdapter.build_for_training` (7239-7340): builds `ConvAE3D`/`LatentFMUNet3D` when
  `spatial_dim == 3`; **infers `num_res_blocks` from the stage-1 state-dict keys** (0ca1294, lines
  7292-7307) — comment: the stored `ae_num_res_blocks` metadata "may be wrong when the checkpoint was
  saved by an older code version that had a mismatched default between build_for_training and
  build_checkpoint". `[bug fix]` (this is the "baseline stage training issue").
- `build_checkpoint` (7417-7479): stores `ae_num_res_blocks`, `Num_z`, `spatial_dim`.
- `visualize_ae_reconstruction` / `visualize_reconstruction_latentfm` `Num_z` (5650-5718, 6168-6229,
  7497-7521): 3-D grids and midplane plots.
- `validate_and_normalize_config` defaults `num_z`, `field_names` (4758-4759); `build_dataset` passes
  `field_names` (4863).

**Group MB-B: cost instrumentation (b50c4d0, May 12 "Updated baselines").** `[new feature/arm]`.
`import time` (3); `TrainingHistoryLogger` CSV/JSON gain `epoch_time_s`, `peak_gpu_mem_mb`,
`cumul_train_time_s` (4899-4940); `run_epoch_ae`/`run_epoch_latentfm`/`S3GMAdapter.run_epoch`/
`SiTAdapter.run_epoch` return `(loss, time, peak_mem)` (5121-5122, 5157-5163, 7136-7143, 7782-7789);
`BaseBaselineAdapter.run_epoch` signature (7008).

**Group MB-C: Senseiver fidelity audit (e5ee749, Aug 30 checkpoint squash; the reasoning is entirely
in the inline comments, lines 4151-4172 and 5456-5476).** `[bug fix]` in the sense of "our Senseiver
was not the published architecture". Adds `upstream_layout=True` reproducing
github.com/OrchardLANL/Senseiver @ e443eb0 exactly: per-axis top frequency from the grid shape
(`SenseiverFourierPositionalEncoding`, 4175-4215; `freqs` buffer made non-persistent so old checkpoints
still load, 4212), width-preserving MLP (`_senseiver_mlp`, 4216), `Residual(CrossAttention)+Residual(mlp)`
in every cross-attention (`SenseiverCrossAttentionLayer`, 4269-4307), residual-branch dropout, tied
encoder layers (`share_encoder_layers`, 4389-4415, 4494-4502), bare `Linear` decoder read-out, `enc_preproc_ch`,
`dec_latent_dim`, `dec_preproc_ch` (4418-4475). `upstream_layout=False` (default) is the pre-audit
layout "ONLY so that the frozen pre-audit run directories ... stay loadable". `SenseiverAdapter`
(7876-7960): `max_freq: null` -> per-axis from `num_x/y/z`; parameter/bottleneck prints; `optimizer:
upstream` -> plain Adam, no schedule, no weight decay (upstream `network_light.py:78`); `load_checkpoint`
drops `pos_enc.freqs`. `run_epoch_senseiver` (5415-5423, 5456-5508): `grad_clip` configurable (default
1.0 preserves pre-audit behaviour; comment: upstream has **no** clipping, and at max_norm=1.0 the clip
bound on 100% of steps — "normalised-gradient Adam, a different optimiser from the published one";
`grad_clip: null` restores upstream), pre-clip gradient-norm logging to `grad_norm_history.json`.
No PoF config sets `upstream_layout` (grep is empty) — **Nick should confirm which layout the PoF
Senseiver rows used.**

**Group MB-D: SiT point-token baseline (38f42bb, Aug 25).** `[new feature/arm]`.
`SUPPORTED_BASELINES` unchanged here; `run_epoch_sit` (5305-5332): `node_subsample` token-budgeted
training with `nearest_sensor_fill_nodes` (sensors still drawn from the FULL field, "protocol
unchanged"); `sit_conditional_sample_points_chunked` (6510-6563): chunked full-field sampling **over a
random permutation** (e5ee749 lines 6532-6550 explain: the H5 raster order makes a contiguous 8192-slice a
single x-plane, far out of distribution for a model trained on space-filling subsets, which produced
sheared vertical striping); `visualize_reconstruction_sit` (6586-6660) uses it; `SiTAdapter` (7576-7602,
7700-7702, 7863-7864) skips grid validation for `pointnet` tokenizer.

**Group MB-E: ensemble / probabilistic scoring hooks (cf42d1b).** `[new feature/arm]`.
`ENSEMBLE_K` env var draws K samples through each baseline's own sampler and scores them with
`ensemble_eval.ensemble_metrics` (latent-FM 6307-6350; SiT 6688-6717, e5ee749/38f42bb); `ENSEMBLE_NPZ`
raw dump (comment: "a reimplementation is what silently dropped latent-FM from the first panel run");
deterministic baselines report `crps = MAE` with `spread 0.0, deterministic: True` (6876-6895) — comment:
"a deterministic forecast is a degenerate predictive distribution, for which CRPS reduces exactly to the
mean absolute error". `_apply_eval_measurement_ops` (6125-6159): env-gated test-time noise / occlusion /
field dropout with a fixed `ENSEMBLE_OP_SEED` so every model sees the identical realisation; called in
latent-FM 3-D path (6210) and deterministic (6837). **Latent bug introduced here (cf42d1b, lines
6122-6125):** the `@torch.no_grad()` that used to decorate `visualize_reconstruction_latentfm` now sits
above two blank lines and decorates `_apply_eval_measurement_ops` instead; `visualize_reconstruction_latentfm`
is no longer `no_grad`. Benign in practice because `LatentFlowMatching.sample` (line 2580) is itself
`@torch.no_grad()`, but worth fixing. `[bug fix needed]`.

**Group MB-F: z-collapse plot fixes (988bc86, 756f2b7).** `[bug fix]`, plot only. `midplane_slice`
routed into `visualize_reconstruction_s3gm` (5959-5989) and `visualize_reconstruction_sit` (6729-6759);
`visualize_reconstruction_latentfm` (6361-6401) and `visualize_reconstruction_deterministic` (6909-6955)
slice inline; all pass `metric_true_f/metric_pred_f` so the title reports volume L2. 988bc86 body: "No
published number changes."

**Group MB-G: CoNFiLD baseline (e5ee749; the adapter design is from 38f42bb's body).**
`[new feature/arm]`. `SUPPORTED_BASELINES` adds `"confild"` (4148); `resolve_stage_config` handles
`confild_params.stage{1,2}` (4800-4801); `BaseBaselineAdapter.uses_custom_training_loop` +
`run_custom_training` (6997, 7023-7031); `CoNFiLDAdapter` (8215-8271) delegates to
`confild_upstream_training.run_confild_training` because "CoNFiLD's latent table and once-per-epoch
decoder update cannot be represented by the generic stateless minibatch adapter interface"; registry
entry (8281).

**Group MB-H: dataset gate and surface pool (e7745ad, 3790338).** `validate_and_normalize_config`
(4764-4766, 4778-4789): `sensor_pool` default; dataset-name gate widened from `turbulent_combustion`
only to `{turbulent_combustion, kolmogorov2d, cylinder2d}` (e7745ad body: canonical-H5 datasets flow
through the dimension-agnostic path). `build_dataset` (4847-4854 cf42d1b shiftwing branch; 4865
3790338 `sensor_pool`). `run_epoch_latentfm/sit`, `visualize_reconstruction_sit` forward
`valid_sensor_mask` to `build_sparse_condition(valid_mask=...)` (5193-5202, 5292-5345, 6600-6610,
3790338). `[3D/JHU generalization]` / `[new feature/arm]`.

**Group MB-I: `build_dataloader` `persistent_workers` opt-in (4870-4888, e5ee749).**
`[performance/memory]`. Long comment with measurements (`profile_io_sit.py`): ~4 s of every epoch was
worker process spawn; off by default (`JHU_PERSISTENT_WORKERS=1`) "so no other baseline's behaviour
changes".

### B.3.7 `src/train_Gen_Baseline.py` (+62 / -4)

Commit trail: e5ee749, d005de1, b50c4d0.
- Lines 123-130 (e5ee749): CoNFiLD requires CUDA; stage-2 checks its stage-1 checkpoint before
  creating artefacts. `[new feature/arm]`.
- Line 146 (e5ee749): `cuda_device_name` written to `run_meta.json`. `[protocol/leakage/integrity
  guard]` — the audit found sensor draws are GPU-SKU dependent, so the SKU is recorded.
- Lines 153-175 (e5ee749): custom-training-loop dispatch for CoNFiLD, writes `final_summary.json`.
- Lines 198-208 (b50c4d0): parameter count print. `[cosmetic]`.
- Lines 216-219, 271-272 (e5ee749): `loss_plot.LossTracker` live loss plot. `[cosmetic]`.
- Lines 221, 230, 301-303 (b50c4d0; 230 d005de1): 3-tuple `run_epoch`, time/mem in the CSV and log line.
  `[new feature/arm]` (cost).
- Lines 249-263 (e5ee749): opt-in checkpoint archiving (`archive_from`, `archive_every`) "so
  checkpoint-to-checkpoint noise can be measured after the fact"; no-op unless configured.
  `[new feature/arm]`.

### B.3.8 `src/train_Det_Baseline.py` (+141 / -5)

Commit trail: e5ee749, cf42d1b.
- Lines 5-6: `os`, `time` imports.
- Lines 163-212 (e5ee749): `LossTracker`; opt-in wall-clock budget `BASELINE_MAX_HOURS` (writes
  `budget.pt` and exits 0 on expiry — used for budget-matched comparisons such as the Geo-FNO
  `kolmogorov2d_fullbudget` row in HANDOFF sec. 3); `BASELINE_ARCHIVE_N`/`_TAIL_FRAC` archiving;
  `cost_train.json` with optimizer steps, wall time and duty cycle (comment: wall-clock alone cannot be
  compared across runs when shared-filesystem contention varies). `[new feature/arm]`.
- Lines 214-248 (cf42d1b timing, e5ee749 cost prints): per-epoch `synchronize` + peak memory,
  greppable `[cost]` lines. `[new feature/arm]`.
- Lines 272-273: tracker log/plot. Lines 287-311: time/mem in CSV and log; archive write. Lines
  313-332: budget stop. Lines 336-343: final `[cost]` summary.
Nothing here changes what the deterministic baselines compute.

### B.3.9 `src/train_finetune.py` (+14)

Commit trail: 693fbd3 (audit merge). Two hunks, both `[bug fix]` plot-only, from 060c9a5:
- Line 783: `# zcollapse-ok` annotation on `_save_rollout_field_figure`.
- Lines 928-938: `run_rollout_evaluation` slices the z-midplane before handing `coords_xy` to the
  triangulation (was superposing 125 z-levels). Metrics (`rel_l2_phys`, `row`) are unchanged.
Everything else in this 900+-line RAM/LoRA fine-tuning script is byte-identical to 828c3c6 — but note
upstream has since rewritten it substantially (a972f92 "Updated LoRA score calculation", 088b3c8,
85b8e59: +611 lines net; see B.6).

### B.3.10 `src/evaluate_full_dataset.py` (+37)

Commit trail: 693fbd3, c93c08a.
- Lines 438-451 (c93c08a): `_effective_field_names` — falls back to `field_i` names when the dataset's
  `field_names` length does not match `num_fields` (JHU H5s have 4 channels vs the 5 combustion names).
  `[3D/JHU generalization]` / `[bug fix]` (would otherwise `zip`-truncate or index-error).
- Lines 702, 709, 721-724, 738-746, 757, 1288, 1329-1330, 1368, 1391 (c93c08a): `num_z` threaded from
  cfg (`Num_z`/`num_z`) into `_infer_structured_grid_from_coords` (full coords, not `[:, :2]`) and
  `dz` into `_gradient_metrics`; per-snapshot field-count guard. `[3D/JHU generalization]`.
- Line 1149 (693fbd3): `# zcollapse-ok` on the summary heat-map (not a spatial field). `[cosmetic]`.
- Lines 1445-1446 (c93c08a): metric notes say "2D or 3D". `[cosmetic]`.

### B.3.11 `src/evaluate_coherence.py` (+15)

Commit trail: 693fbd3. Both from the 060c9a5 audit:
- Lines 2256-2269: `save_worst_direction_spatial_map` slices the z-midplane for the picture only
  (`rmse`/`linf` stay full-volume). `[bug fix]`, plot only.
- Line 2406: `# zcollapse-ok` on `save_pairwise_heatmap` (an `[n_fields, n_fields]` matrix).
Upstream has since changed this file by +345 lines (d9d32b6 "Corrected coherence evaluation", 088b3c8,
85b8e59, a142905, 0cd5263) — a guaranteed merge conflict region if Nick pulls those; see B.6.

### B.3.12 `src/View_Dataset.py` (+3)

Commit trail: 693fbd3. Lines 91-92, 103: `# zcollapse-ok` comments only ("2-D demo dataset viewer;
never point this script at a 3-D volume without adding a z-slice"). `[cosmetic]`.

### B.3.13 `Dataset/README` (1 line)

3790338. `Dataset can be downloaded at: https://drive.google.com/...` -> `data is located at
../../../datasets` (i.e. `/work/hdd/bilr/ntricard/datasets/`). `[HPC portability]`. Nick may want to
keep Jason's Google-Drive line for the combustion demo data alongside ours.

### B.3.14 `.gitignore` (repo root, 1 line)

e5ee749. Adds `!config_baseline_CoNFiLD_xcube.yaml` to the un-ignore list (the root `.gitignore`
ignores `Save_config/*.yaml` except an allow-list). `[new feature/arm]` (CoNFiLD). The 2026-09-22 pull (8aa01c9) appended `0_demo_TurbulentCombustion/Save_TrainedModel/` and `*.out` to the ignore list — the Engaging run outputs in Section D were force-added before that rule, so they are tracked but new ones will not be. Note upstream has
since rewritten the root `.gitignore` (+66/-?) in 0c2ca93/0cd5263/cb279fe — conflict candidate. Also note
a *second* `.gitignore` was added at `0_demo_TurbulentCombustion/.gitignore` (3790338/7d85f1f) which
ignores `*.log`, `slurm-*.out`, `slurm_logs/`, `Save_TrainedModel/`, `*.blg`.

---

## B.4 The deleted `0_demo_TurbulentCombustion/README.md`

Deleted in 464c72e (2026-09-08, "Removed old MD files", no body) together with 14 other files
(`HANDOFF.md`, `BASELINE_AUDIT_2026-08-28.md`, `FLEET_AUDIT_2026-08-29.md`, `PLAN_*.md`,
`HEADLINE_COMPARISON_2026-09-04.md`, `GL_rbf_CQ_UPDATE_GUIDE.md`, `GL_rbf_CQ_RELEASE_MANIFEST.yaml`,
two reference PDFs, `weekly_update_2026-08-30.pptx`).

What it contained (`git show 828c3c6:0_demo_TurbulentCombustion/README.md`, 645 lines — identical to
what we deleted; we never edited it): Jason's user guide for the demo. Section headings: 1 Main
Workflow: Point-Cloud Flow Matching (1.1 `Model.py` contents, 1.2 backbone choices, 1.3
`train_pointcloud_ffm.py`, 1.4 `evaluate_ffm.py`, 1.5 outputs, 1.6 example commands); RAM Fine-Tuning;
2 Unified Generative Baselines (2.1-2.5); 3 Unified Deterministic Baselines (3.1-3.4); 4 Relationship
Between the Pipelines; 5 Recommended Starting Point.

Where the content went: **nowhere in our tree.** None of our docs (`HANDOFF_2026-09-12.md`,
`DELTA_STATUS_2026-09-08.md`, `HPC_MIGRATION_AND_REMAINING_WORK.md`, `Paper/*`) reproduce it; they
document the campaign, not the code's usage. Upstream still has it and has grown it by +65/-2 lines
since the fork (a972f92 LoRA score section, 088b3c8, 0c2ca93 "direct coherence post-training"). If the
fork is ever published, restoring `upstream/main:0_demo_TurbulentCombustion/README.md` and appending a
"campaign" section is the cheapest fix. Sections 1.3-1.5 of that README describe upstream's
`TrainingHistoryLogger` / `Evaluation/` layout, which ours does not use (B.1).

---

## B.5 The 668 added files, grouped

Counts are from `git diff --diff-filter=A --name-only 828c3c6 HEAD`. Paths relative to
`0_demo_TurbulentCombustion/` unless noted.

| # | Group | Count | Purpose |
|---|---|---|---|
| 1 | `src/*.sh` + `src/*.slurm` (launchers) | 232 | One SLURM launcher per run/eval/probe. Naming: `train_iclr_*` (ICLR JHU/FireBench/wing arms), `train_bench_{kolm,cyl}_ffm.sh` (PoF DMF-Gen), `train_baseline_*`, `eval_*` (canonical + probes), `dump_*` (field dumps for galleries), `calib_*` (conformal/recalibration), `smoke_*` (gates), `confild_*`, `s3gm_*`, `sA*/sB*` (S3GM improvement stages), `run_classical_*`, `submit_*_delta.sh`. All now carry `ghx4 / bilr-dtai-gh` headers (3790338). |
| 2 | `src/*.py` — DMF-Gen model & training extensions | ~15 | `spectral_loss.py`, `spectral_prior.py`, `spectral_utils.py` (Hann `shell_spectrum`), `augment_symmetry.py`, `augment_octahedral.py`, `measurement_ops.py`, `model_cq.py` (CQ adapter), `model_ema.py`, `persistent_topk_geometry_cache.py`, `cq_flash_patch.py`, `fno3d_backbone.py`, `dataset_shiftwing.py`, `dataset_wing_baseline.py`, `lfm_fixes.py`, `loss_plot.py`. |
| 3 | `src/phycoflow_pointcloud/` (vendored) | 19 | Jason's portable GL_rbf_CQ package v0.9.0-pre1 vendored verbatim (27f7308), used only by the `GL_rbf_CQ` backbone branch. |
| 4 | `src/*.py` — evaluation drivers & integrity guards | ~25 | `ensemble_eval.py` (canonical probabilistic eval: K samples, CRPS, coverage; compute-node guard + sensor fingerprint), `eval_latentfm_ensemble.py`, `eval_sit_ensemble.py`, `eval_kolm_ensemble.py`, `eval_kolm_litprotocol.py`, `eval_senseiver_iclr.py`, `eval_deeponet*.py`, `eval_s3gm3d.py`, `evaluate_wing.py`, `run_final_eval.py`, `fleet_select.py` (payload-keyed row selection), `artifact_guard.py` + `test_artifact_guard.py` (write-time identity guard, faa8060), `sensor_fingerprints.py`, `rng_parity_probe.py`, `verify_seedcheck.py`, `check_no_zcollapse.py`, `gen4turb_eval.py`, `export_for_baselines.py`. |
| 5 | `src/*.py` — baselines | ~30 | `confild_*.py` (13: upstream-faithful CoNFiLD stages, eval, selection), `s3gm3d.py`, `train_s3gm3d.py`, `s3gm_*` (isolate/norm-guidance/watchdog/improve), `deeponet_baseline.py`, `deeponetpp.py`, `train_deeponet*.py`, `baseline_classical_2d.py`, `baseline_classical_jhu.py`, `baseline_classical_figs.py`, `baseline_gappy_pod_wing.py`, `floor_variants.py`, `sen_local_xattn.py`, `sen_sweep_fixes.py`, `train_det_sweep.py`, `instrument_sit.py`. |
| 6 | `src/*.py` — calibration / UQ | ~10 | `calib_*.py` (7: archive window, ckpt variance, probes, table, truth spectrum), `conformal_recalib.py`, `recalibrate_spread.py` (`--split-by position`, 3790338), `dump_calib_points.py`, `k_ablation.py`, `seed_variance_2d.py`, `paired_ci_2d.py`, `paired_ci_jhu.py`. |
| 7 | `src/*.py` — spectra / diagnostics / profiling / benches | ~25 | `compare_spectra*.py`, `spectra_arms.py`, `spectra_paper.py`, `diagnose_spectra.py`, `diagnose_sample_spectra.py`, `turbulence_stats.py`, `profile_*.py`, `bench_*.py`, `benchmark_*.py`, `audit_*.py`, `check_*.py`, `diag_senseiver_gradnorm.py`, `cross_re_cylinder.py`, `summarize_2d_fleet.py`, `archive_checkpoints.py`, `compare_ckpts.py`, `merge_winner.py`. |
| 8 | `src/*.py` — figures & galleries | ~15 | `qualitative_{jhu,firebench,wing}.py`, `replot_*.py`, `regen_figs.py`, `paper_jhu_panels.py`, `dump_fields_baseline.py`, `dump_jhu_gallery.py`, `dump_classical_gallery.py`, `gen_firebench_field.py`, `assemble_baseline_table.py`. |
| 9 | `src/figs_pof/` | 23 | PoF paper figure/table generators (`fig_*.py`, `make_*_tables.py`, `make_recon_galleries.py`, `pof_style.py`, `verify_sources.py`). |
| 10 | `src/test_*.py` + `tests/` | 9 | Gates: `test_ode_cache.py` (M8 exactness), `test_attn_mechanism.py`/`test_attn_fix.py`/`test_attn_compile.py` (M4), `test_cq_adapter.py`, `test_cq_flash.py`, `test_persistent_workers.py`, `test_artifact_guard.py`; `tests/test_portable_core.py` (vendored package isolation). |
| 11 | `src/*.py` — data conversion | 4 | `convert_kolmogorov.py` (Shu npy -> canonical H5, trajectory holdout), `convert_cylinder.py` (OpenFOAM -> H5, surface ring indices), `add_grid_surface_indices.py`, `extract_firebench.py`, `check_strouhal.py`. |
| 12 | `src/engaging/` | 8 | Launchers/merge scripts for the MIT Engaging cluster (seed replicates, FireBench merge) — a third HPC, pre-Delta. |
| 13 | `Save_config/` | 62 | `config_iclr_*.yaml` (28 ICLR arms), `config_baseline_*_{xcube,firebench,wing,...}.yaml`, `kolmogorov2d*/` (26 incl. seed7/seed1337/fullbudget), `cylinder2d*/` (37 incl. surface/uonly/seeds), `backup/`, `config_smoke_spec.yaml`. Covered by the config section of the guide; listed here for completeness. |
| 14 | `Paper/` | 93 | `pof2026/` (50: AIP REVTeX template, `main.tex`, `tab_*_body.tex`, `figures/*.pdf`, bib), `iclr2027/` (34: earlier draft, `PAPER_PLAN.md`, `figures/`, `main.log` — the only tracked `.log`), `glu_arxiv_2026/` (9). |
| 15 | `openfoam/cylinder2d/` | 11 | Cylinder-wake dataset generation: `case_template/` (blockMesh O-grid, pimpleFoam laminar), `run_case.sh`, `submit_all.sh` (Re array), `env_test.sh` (8-stage preflight). |
| 16 | Docs & transfer manifests (`0_demo_TurbulentCombustion/*.md`, `transfer_tier*.txt`, `.gitignore`, `src/NOTES`, `Save_TrainedModel_pof/field_dumps/README.md`) | 12 | `HANDOFF_2026-09-12.md`, `DELTA_STATUS_2026-09-08.md`, `HPC_MIGRATION_AND_REMAINING_WORK.md`, `DATA_TRANSFER_MANIFEST.md`, `DATA_TRANSFER_TIER2.md`, `SURFACE_TASK_PREREGISTRATION.md`, `SURFACE_TASK_RESULTS_LOG.md`; `src/NOTES` is Nick's May run notebook. |
| 17 | Repo-root `temp/` and `eval_infra_audit.patch`, `tools/` | 18 | `temp/Model_{new,original}.py`, `temp/train_pointcloud_ffm_old.py`, `temp/pointcloud_ffm_updated.py` (scratch copies from the CQ integration), `temp/confild_upstream_jhu/` (12, standalone CoNFiLD port); `eval_infra_audit.patch` (the 060c9a5 audit as a patch, 37 KB); `tools/send_datasets_to_delta.sh`, `tools/make_transfer_lists.sh`. (`tools/package_2d_assets/` and `tools/package_2d_datasets.py` are untracked, not in the 668.) |

Correction to the brief: **no `src/*.log` job logs are committed.** 157 `.log` files exist on disk in
`src/` but are ignored by `0_demo_TurbulentCombustion/.gitignore:3` (`*.log`); the only tracked `.log`
is `Paper/iclr2027/main.log` (a LaTeX log). `git ls-files | grep '\.log$'` confirms.

---

## B.6 Upstream's 11 commits since the fork point (what Nick is missing)

`git log --format='%h %ad %an %s' --date=short 828c3c6..upstream/main`; all by cosmos2w.
`git diff --stat 828c3c6 upstream/main` = 335 files, +331,592 / -408 (most of it four `datagen/*/viz.ipynb`
notebooks and a new top-level `Proj_MultiFieldReconstruction/` package).

| Commit | Date | Subject | Files in `0_demo_TurbulentCombustion/` it touches | Overlaps a file we modified? |
|---|---|---|---|---|
| d9d32b6 | 06-09 | Corrected coherence evaluation | `src/evaluate_coherence.py` (+25), `Save_config/config_baseline_Det.yaml`, `config_pointcloud_ffm.yaml`; rest in `1_SubTask_SuperResolution/` | **yes**: evaluate_coherence.py (our +15 is in a different function, likely auto-mergeable), both yamls |
| a972f92 | 06-09 | Updated LoRA score calculation | `README.md` (+14), `Save_config/config_pointcloud_ffm_ram.yaml`, `src/Model.py` (+17: the `_print_gather_notice_once` classmethod replacing the two `print`s at our lines 766-767), `src/train_finetune.py` (+363) | **yes**: Model.py (trivial, same 2 lines), train_finetune.py (ours only +14 plot fix; theirs rewrites scoring — take theirs), README.md (we deleted it) |
| 088b3c8 | 06-23 | Updated coherence visualization | `README.md`, `config_pointcloud_ffm_ram.yaml`, new `src/diagnose_ram_reward_signal.py`, `src/evaluate_coherence.py` (+83), `src/evaluate_ffm.py` (+63), `src/evaluate_full_dataset.py` (+20), `src/train_finetune.py` (+241) | **yes**: evaluate_ffm.py (their `_as_int_list`/`_align_per_field_values` normalisation + `use_fourier_pe` — will conflict with our rewritten `main`), evaluate_full_dataset.py, evaluate_coherence.py, train_finetune.py |
| 0c2ca93 | 06-23 | Added direct coherence post-training | `.gitignore`, `README.md` (+45), new `Save_config/config_pointcloud_ffm_direct_posttrain.yaml`, new `src/direct_coherence_loss.py`, `src/train_pointcloud_ffm.py` (+656) | **yes, hard**: train_pointcloud_ffm.py — they added a post-training mode to the trainer we rewrote (EMA/AMP/compile/spectral). Expect a manual merge. |
| c1ee024 | 06-24 | Updated structured config_pointcloud_ffm_direct_posttrain | `config_pointcloud_ffm_direct_posttrain.yaml`, `src/train_pointcloud_ffm.py` (+121) | **yes** (same as above) |
| 85b8e59 | 07-08 | Updated coherence diagnosis and post-training | `config_baseline_Det.yaml`, `config_pointcloud_ffm.yaml`, `direct_posttrain.yaml`, `src/Model.py` (+20: Fourier PE option on `ConditionalPointPerceiver`), `direct_coherence_loss.py`, `evaluate_coherence.py`, `evaluate_ffm.py` (+24: `_cfg_get_not_none`), `evaluate_full_dataset.py`, **`helpers.py` (+40: `normalize_field_names` / `resolve_field_names` reading `selected_fields` from the H5 attrs; `field_names` ctor default becomes `None`)**, **`helpers_baseline.py` (+40 same)**, **`model_baseline.py` (+21: Perceiver Fourier PE + `build_dataset` passes `field_names`)**, `train_finetune.py`, `train_pointcloud_ffm.py` (+49) | **yes**: helpers.py / helpers_baseline.py ctor signature region (we added `sensor_pool` right there; both add a kwarg, so a textual conflict is likely but semantically compatible), Model.py (Perceiver class — we did not touch it, clean), model_baseline.py `build_dataset` (we pass `field_names` + `sensor_pool` at the same call — conflict) |
| a142905 | 07-10 | Added phy coherence evaluator | `config_pointcloud_ffm.yaml`, `direct_posttrain.yaml`, `evaluate_coherence.py` (+111), new `src/evaluate_phy_coherence.py` (1555 lines), `train_pointcloud_ffm.py` (+8) | **yes**: evaluate_coherence.py, train_pointcloud_ffm.py (small) |
| 0cd5263 | 07-13 | Updated eval procedure for mixed resolution demo | `.gitignore`, `direct_posttrain.yaml`, `evaluate_coherence.py` (+108), `train_pointcloud_ffm.py` (+228); `1_SubTask_SuperResolution/*` | **yes**: train_pointcloud_ffm.py, evaluate_coherence.py |
| cb279fe | 08-14 | Updated canonical data generation | `.gitignore` (+63), `config_pointcloud_ffm.yaml`, `direct_posttrain.yaml`, new `src/_OlderVersion/` (13 archived scripts), `direct_coherence_loss.py`, `evaluate_full_dataset.py` (+47), new `tests/test_config_gradient_update.py`; new top-level `datagen/` (solvers, H5 pipeline, notebooks) | **yes**: `.gitignore` root (ours +1 line — trivial), evaluate_full_dataset.py |
| 754ad8c | 08-15 | Updated PhyCoFlow Unified Project Dir | New top-level `Proj_MultiFieldReconstruction/` (~250 files: `phycoflow_reconstruction` package with coherence families, cases for brusselator / kolmogorov / ks / mass transport / turbulent combustion, tests, CI) | no overlap (new tree) |
| f163ad7 | 08-17 | Merged all three coherence terms | `Proj_MultiFieldReconstruction/` only (global-distribution + cross-spectrum + topology coherence configs, `ModelExplain.md`, `README.md`) | no overlap |

What Jason's later work gives that we lack, in one line each:
- **Coherence terms** (d9d32b6, 088b3c8, 85b8e59, a142905, f163ad7): corrected `evaluate_coherence.py`,
  a new `evaluate_phy_coherence.py` physical-coherence evaluator, and in `Proj_MultiFieldReconstruction`
  three coherence families (global distribution, cross-spectrum, topology/Betti curves).
- **Direct coherence post-training** (0c2ca93, c1ee024, 0cd5263): `direct_coherence_loss.py` and a
  post-training mode inside `train_pointcloud_ffm.py`.
- **LoRA/RAM score** (a972f92, 088b3c8): rewritten reward-signal scoring in `train_finetune.py` and a
  `diagnose_ram_reward_signal.py` tool.
- **Canonical data generation** (cb279fe): `datagen/` package (Burgers, KS, Brusselator, Navier-Stokes,
  electro-thermal, mass transport) with an H5 pipeline; also archives `_OlderVersion/` scripts.
- **Unified project dir** (754ad8c, f163ad7): `Proj_MultiFieldReconstruction/` — a clean packaged
  re-implementation with contracts/tests; independent of `0_demo_TurbulentCombustion/`.
- Small but relevant to us: `helpers.resolve_field_names` reading `selected_fields` from the H5 attrs
  (85b8e59) does what our `--field-names` CLI / `field_names` cfg does by hand; `evaluate_ffm._cfg_get_not_none`
  (85b8e59) and `_align_per_field_values` (088b3c8) harden config normalisation that `ensemble_eval.py`
  relies on through `_normalize_eval_config`.

Merge-conflict candidates, ranked by pain: `train_pointcloud_ffm.py` (theirs +992 vs ours +902 in the
same functions) > `evaluate_ffm.py` (theirs +87 in `main`/normalisation vs our rewritten `main`) >
`evaluate_coherence.py` (theirs +345, ours +15) > `helpers.py` / `helpers_baseline.py` /
`model_baseline.py` (small, same ctor/call sites) > `train_finetune.py` (theirs +611, ours +14 — take
theirs and re-apply our two-hunk z-slice) > `Model.py` (two print lines) > `.gitignore` (one line) >
`README.md` (we deleted; take theirs). The three yamls are covered elsewhere.

A practical way to preview: `git merge --no-commit --no-ff upstream/main` in a throwaway worktree
(`git worktree add /tmp/merge-preview HEAD`), inspect `git diff --name-only --diff-filter=U`, then
`git merge --abort`. Do not do this in the main checkout.


---

# Section C — Experiment YAMLs: inventory, upstream diffs, sensor and k tables

Scope: every YAML under `0_demo_TurbulentCombustion/Save_config/` (recursive, incl. `backup/`), compared against the upstream fork point `828c3c6` (cosmos2w/PhyCoFlow_demo) and against upstream HEAD `upstream/main` = `f163ad7`. All paths below are relative to `/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/` unless absolute. Nothing in the repo was modified; every number in the tables was read from the YAML with a yaml loader (not by eye) and cross-checked against the run-time snapshot copies and `args.json` files where those exist.

---

## C.0 Conventions and reference numbers

### Point counts N used for the "fraction" columns (all verified by opening the H5 files with h5py)

| Dataset | N points / snapshot | Fields | File | Verification |
|---|---|---|---|---|
| Kolmogorov 2D | **65,536** (256x256) | 1 (vorticity) | `/work/hdd/bilr/ntricard/datasets/kolmogorov2d/Kolmogorov2D_shu_stride4.h5` | `fields` shape `(1,3200,65536,1,1,1)`; attrs: 32 train trajs / 8 test trajs, block split gap 0, train_ratio 0.8 |
| Cylinder mesh | **23,800** cells | 3 (Ux,Uy,p) | `.../cylinder2d/Cylinder2D_mesh.h5` | `fields` `(1,1800,23800,1,1,3)`; `surface_indices` = **360** cells |
| Cylinder grid | **80,000** (400x200), body interior = 316 cells zeroed (`body_mask` sum 79,684 fluid) | 3 | `.../cylinder2d/Cylinder2D_grid.h5` | `fields` `(1,1800,80000,1,1,3)`; surface pool sidecar `Cylinder2D_grid.surface_indices.npy` = **62** unique cells |
| JHU isotropic cutouts | **1,953,125** (125^3) | 4 (Ux,Uy,Uz,p) | `.../datasets/JHU_4cubes_stride100.h5` | `fields` `(1,200,1953125,1,1,4)`; frames [0,150) cubes 0-2 train, [150,200) cube 3 held out |
| FireBench merged | **3,677,184** (152x126x192) | 5 (u,v,w,theta,rho_f) | `.../datasets/FireBench_u10u12_merged.h5` | `fields` `(1,120,3677184,1,1,5)` |
| FireBench v1 (u10 only) | 3,677,184 (same crop, from yaml comment) | 5 | `.../firebench3d/FireBench_u10_ramp0_3D.h5` | **file not present on this host** (`ls` fails); N taken from the v5clean yaml comment, line 121 |
| Upstream combustion (Merged_CH4COTU1P) | **40,300** (num_x 403 x num_y 100 from the yaml) | 5 (CH4,CO,T,U_1,p) | `Dataset/Merged_CH4COTU1P.h5` | **file not present** (`Dataset/` holds only `download_firebench.py`, `README`); N is inferred from `shared.data.num_x/num_y` in the fork-point yamls |
| Shift-wing | **not determinable from the yamls** (per-case unstructured near-body point clouds; split lives in `stats.json` under `shift_wing_processed_v3`) | Ux,Uy,Uz,Cp + surface pool [Cp, tau_x, tau_y, tau_z] | `/work/hdd/bilr/ntricard/datasets/shift_wing_processed_v3` | sensor budgets are absolute tap counts on a surface pool, so no fraction of N is meaningful |

### How a "sensor count" in a YAML becomes an actual draw

* `cond_fields` = list of field indices that are observed. `n_obs_min_list` / `n_obs_max_list` = **per observed field** bounds; a length-1 list broadcasts to every field in `cond_fields` (`src/helpers.py:544-545`, `_broadcast_per_field`).
* For every frame in a batch, for every observed field **independently**, the count `m ~ U{nmin..nmax}` is drawn and `m` distinct point indices are drawn by `randperm` (`src/helpers.py:583-589`). So `[24]` with `cond_fields [0,1]` means 24-238 sensors of Ux **and** 24-238 different sensors of Uy, total 48-476 readings. Fractions in the tables below are **per field**.
* With `sensor_pool: "surface"` the randperm is over the H5 `surface_indices` pool and `m` is capped at the pool size (`src/helpers.py:238-257`, `587-589`).
* Training draws are per-frame, per-step (amortised); the paper numbers come from **evaluation-time** draws with fixed seeds at a fixed count (Section C.3.6), not from these training ranges.
* `vis_cond_fields` / `vis_n_obs_list` are only used for the in-training visualisation snapshot (`src/train_pointcloud_ffm.py:398-401`, `729`; `src/model_baseline.py:4767-4770`, `6815`). They never touch a reported number.

### Where the split comes from

`train_ratio` is consumed in `src/helpers.py:261-277`. If the launcher exports `JHU_SPLIT_MODE=block`, the last `round(n*(1-train_ratio))` frames form a contiguous held-out block separated by `JHU_SPLIT_GAP` frames; otherwise (**default** `"shuffle"`, line 261) the frames are randomly permuted with the run seed. Every 2D / JHU-xcube / FireBench launcher exports `block`; the pre-ICLR root-level configs (`config_iclr_jhu_main.yaml`, `config_iclr_jhu_robust.yaml`, `config_baseline_Det.yaml`, `config_baseline_Gen.yaml`, `config_iclr_firebench.yaml` v1) are launched **without** that export and therefore used a shuffled split (see C.4-1). The wing configs also lack the export but are unaffected: the shiftwing loader takes its case split from `stats.json` (yaml comment `config_iclr_wing_v4.yaml:17`).

Symmetry augmentation is likewise an env var, not a yaml key: `JHU_AUGMENT` (`src/helpers.py:288-297`), train split only.

---

## C.1 Inventory of every YAML in `Save_config/`

Totals: **187** `.yaml` files on disk. **111** are git-tracked (the `*.yaml` gitignore at repo root is overridden by force-adds); **76** are untracked run-time snapshots that the trainers write at launch (`Save_config/**/det_baseline/`, `**/gen_baseline/`, `Save_config/pointcloud_ffm/`; written by `src/train_pointcloud_ffm.py:812-816` and the `config_backup_root` key of the baseline trainers). Every one of those 76 snapshots was diffed against the tracked yaml with the same demo number in the same directory: **all identical except one** (`kolmogorov2d_fullbudget/det_baseline/Baseline_geofno_Stage1_DemoN265_20260909_212527.yaml`, see C.4-8). They are therefore not listed individually below.

"Derives from" is by structure: the three upstream schemas are **D** = `config_baseline_Det.yaml` (deterministic baselines: mlp_rbf / senseiver / geofno), **G** = `config_baseline_Gen.yaml` (generative baselines: s3gm / latent_fm / sit), **F** = `config_pointcloud_ffm.yaml` (DMF-Gen / PointCloudFFM trainer, flat key space), **R** = `config_pointcloud_ffm_ram.yaml` (RAM fine-tune). "Used by" = launch/eval scripts under `src/` that name the file (grep of `src/*.sh`, `src/*.py`, `src/engaging/*.sh`). Role: **PAPER** = produces a row/figure in `Paper/pof2026/main.tex`; **ABL** = ablation/sweep arm; **SMOKE** = 2-3 epoch gate; **LEGACY** = superseded pre-PoF (ICLR) campaign or upstream leftover.

### C.1.1 Root `Save_config/` (upstream-era + ICLR/JHU-era, all tracked)

| File | Derives from | Method / regime / arm | Used by (src/) | Role |
|---|---|---|---|---|
| `backup/config_baseline_Det.yaml` | D — **byte-identical to upstream 828c3c6** | senseiver, combustion T+U_1, demo 10 | `train_Det_Baseline.py`, `evaluate_Det_Baseline.py`, `train_iclr_det_senseiver.sh` (all via the *name* `config_baseline_Det.yaml`, i.e. they hit the root file, not `backup/`) | LEGACY (pristine upstream copy) |
| `backup/config_baseline_Gen.yaml` | G — **not** the upstream file: an early JHU latent_fm adaptation (cond_fields [0,1,2,3], AE 64/128, FM base 64, ema 0.0, stage 2) | latent_fm, JHU all-4-fields | none by path (root name only) | LEGACY |
| `backup/config_pointcloud_ffm_ram.yaml` | R — **byte-identical to upstream 828c3c6** | RAM/LoRA fine-tune, combustion | `train_finetune.py` looks for `Save_config/config_pointcloud_ffm_ram.yaml`, which **does not exist** at root (C.4-10) | LEGACY |
| `config_baseline_Det.yaml` | D (modified, see C.2.1) | senseiver, JHU, shuffled split, demo 20 | `train_Det_Baseline.py`, `evaluate_Det_Baseline.py`, `train_iclr_det_senseiver.sh` | LEGACY (pre-xcube JHU) |
| `config_baseline_Gen.yaml` | G (modified, see C.2.2) | latent_fm, JHU, shuffled split, demo 0 | `train_Gen_Baseline.py`, `evaluate_Gen_Baseline.py`, `evaluate_full_dataset.py`, `train_baseline_lfm.sh` | LEGACY |
| `config_baseline_Gen_Linzheng_original.yaml` | G — identical to upstream 828c3c6 except a whitespace change on `sit_params.sampling.benchmark_n_steps` | sit, combustion T-only | none | LEGACY (reference copy of upstream) |
| `config_baseline_Det_xcube.yaml` | D via `config_baseline_Det.yaml`; changes: `data_path` -> JHU_4cubes, `train_ratio` 0.9->0.75, demo 30 | senseiver, JHU cross-cube, no aug | `train_baseline_det_xcube.sh` | ABL (no-aug arm) |
| `config_baseline_Det_xcube_aug.yaml` | = `Det_xcube` + demo 31 (aug is `JHU_AUGMENT=octahedral` in the launcher) | senseiver, JHU cross-cube **(paper Senseiver 3D row, DemoN31)** | `train_baseline_det_xcube_aug.sh`, `eval_xcube.sh`, `eval_xcube_matched.sh`, `eval_crps_all.sh`, `eval_stats_expand_base.sh`, `bench_baseline_sample.sh` | PAPER |
| `config_baseline_Det_bench.yaml` | = `Det_xcube_aug` with epochs 3, demo 95 | senseiver smoke | `smoke_det_bench.sh` | SMOKE |
| `config_baseline_Det_firebench.yaml` | D via `Det_xcube_aug`; FireBench grid 152x126x192, cond [0,1,2], n_obs 1000-36772, bs 6 | senseiver, FireBench | `train_baseline_det_firebench.sh`, `eval_firebench_ops*.sh`, `engaging/train_fb_baseline_engaging.sh` | LEGACY (ICLR FireBench; not in PoF paper) |
| `config_baseline_Gen_xcube.yaml` | G via `config_baseline_Gen.yaml`; JHU_4cubes, train_ratio 0.75, demo 22 | latent_fm, JHU cross-cube, no aug | `train_baseline_lfm_xcube.sh` | ABL |
| `config_baseline_Gen_xcube_aug.yaml` | = `Gen_xcube` + demo 23 | latent_fm, JHU cross-cube **(paper latent-FM 3D row, DemoN23)** | `train_baseline_lfm_xcube_aug.sh`, `eval_xcube*.sh`, `eval_crps_all.sh`, `eval_stats_expand_base.sh`, `bench_baseline_sample.sh` | PAPER |
| `config_baseline_Gen_firebench.yaml` | G via `Gen_xcube_aug`; FireBench dims, cond [0,1,2], n_obs 1000-36772 | latent_fm, FireBench | `train_baseline_lfm_firebench.sh`, `eval_firebench_ops*.sh`, `engaging/...` | LEGACY |
| `config_baseline_SiT_xcube.yaml` | G (sit block only, pointnet tokenizer) — new schema keys `node_subsample`, `cond_fill_sigma` | SiT-point, JHU cross-cube **(paper SiT 3D row, DemoN41)** | `train_sit_xcube.sh`, `train_sit_xcube_matched.sh`, `eval_sit_xcube.sh`, `eval_sit_xcube_array.sh` | PAPER |
| `config_baseline_SiT_xcube_smoke.yaml` | = `SiT_xcube`, epochs 4, huber_beta 0.1, sampling_N 4 | SiT smoke | `train_sit_xcube_smoke.sh` | SMOKE |
| `config_baseline_SiT_wing.yaml` | G (full G schema retained), dataset shiftwing | SiT, wing | `train_sit_wing.sh` | LEGACY |
| `config_baseline_CoNFiLD_xcube.yaml` | **new — no upstream analogue** (`confild_params` schema) | CoNFiLD arm C (cap-1024), JHU cross-cube, demo 23 | none by name (the `confild_*.sh` scripts use other names; `confild_final.sh` etc. reference missing `config_baseline_CoNFiLD_figsmoke.yaml`) | PAPER (CoNFiLD 3D row) — but the launch path is not reproducible from a script that names this file (C.4-10) |
| `config_pointcloud_ffm.yaml` | F (modified, see C.2.3) | DMF-Gen, JHU, shuffled split, gather_topk 64, demo 3 | `train_pointcloud_ffm.py` (its `--config` default) | LEGACY |
| `config_iclr_jhu_main.yaml` | F; = `config_pointcloud_ffm.yaml` with epochs 4000, save_dir iclr_jhu_main, demo 4 | DMF-Gen, JHU, **shuffled** split 0.9, k 64 | `train_iclr_jhu_main.sh` | LEGACY (leaky-split run, cf. paper §leakage) |
| `config_iclr_jhu_robust.yaml` | F; + `measurement_ops` (noise 0-0.2, slab occlusion p 0.3 / 10-35 %, field dropout 0.5), cond [0,1,2,3], n_obs 488-9765 | DMF-Gen robustness arm, JHU, shuffled | `train_iclr_jhu_robust.sh` | LEGACY |
| `config_iclr_jhu_xcube.yaml` | F; JHU_4cubes, train_ratio 0.75, epochs 6000, **gather_topk 16**, n_query 39062, `t_sampling logit_normal`, `use_amp`, `ema_decay 0.9995`, `compile_model` | DMF-Gen cross-cube base, no aug, DemoN4 | `train_iclr_jhu_xcube.sh` (launcher DEMO_NUM=12, run dir on disk is DemoN4) | ABL (no-aug) |
| `config_iclr_jhu_xcube_aug.yaml` | = `xcube` + demo 15 (aug in launcher) | DMF-Gen + octahedral aug | `train_iclr_jhu_xcube_aug.sh` | ABL |
| `config_iclr_jhu_xcube_sym.yaml` | = `xcube` + demo 17 (launcher: octahedral,so3) | DMF-Gen + SO(3) aug | `train_iclr_jhu_xcube_sym.sh` | ABL |
| `config_iclr_jhu_xcube_ti.yaml` | = `xcube` + demo 21 (launcher: octahedral_proper,so3,translate) | DMF-Gen + translation aug | `train_iclr_jhu_xcube_ti.sh` | ABL |
| `config_iclr_jhu_xcube_local.yaml` | = `xcube` + `num_latents` 16, `query_latent_readout` false, demo 20 | DMF-Gen "local-only" ablation | `train_iclr_jhu_xcube_local.sh` | ABL |
| `config_iclr_jhu_xcube_ms.yaml` | = `xcube` + `gather_multiscale` true, `gather_topk_coarse` 64, `gather_coarse_sigma_scale` 4.0 | DMF-Gen multiscale gather, no aug | `train_iclr_jhu_xcube_ms.sh` | ABL |
| `config_iclr_jhu_xcube_ms_aug.yaml` | = `xcube_ms` + demo 16 | multiscale + aug | `train_iclr_jhu_xcube_ms_aug.sh` | ABL |
| `config_iclr_jhu_xcube_spec.yaml` | = `xcube` + `spectral_weight` 0.05, block 32, bins 12, demo 22 | spectral loss 0.05 | `train_iclr_jhu_xcube_spec.sh` | ABL |
| `config_iclr_jhu_xcube_spec02.yaml` | = `xcube` + `spectral_weight` **0.02**, demo **29** | **DMF-Gen N29 = the frozen paper 3D row** | `train_iclr_jhu_xcube_spec02.sh`, `calib_ckptvar_train.sh`, `engaging/train_jhu_temporal_engaging.sh`, `engaging/train_jhu_xcube_seedrep_engaging.sh` | PAPER |
| `config_iclr_jhu_xcube_specwin.yaml` | = `spec02` + `spectral_window` true, demo 35 | windowed spectral loss | `train_iclr_jhu_xcube_specwin.sh` | ABL |
| `config_iclr_jhu_xcube_cq.yaml` | = `spec02` with backbone `GL_rbf_CQ`, demo 33 | upstream CQ backbone trial | `train_iclr_jhu_xcube_cq.sh` | ABL |
| `config_iclr_jhu_xcube_kprior.yaml` | = `spec02` with `prior rff_kolmogorov`, `rff_features` 1024, `prior_k_min` 1, `prior_k_max` 48, `prior_slope` 1.6667, demo 34 | spectrally matched prior | `train_iclr_jhu_xcube_kprior.sh` | ABL |
| `config_iclr_jhu_xcube_reg_s42/_s7/_s1337.yaml` | = `xcube` + dropout 0.1 (attn, mlp, sensor_local), weight_decay 1e-2, `measurement_ops` (noise 0-0.1, slab p 0.2 / 10-30 %, field dropout 0.3), epochs 3000, seeds 42/7/1337, demos 23/24/25 | regularised arms | `train_iclr_jhu_xcube_reg_s*.sh` | ABL (seed triplet) |
| `config_iclr_jhu_xcube_sup_s42/_s7/_s1337.yaml` | = `xcube` + `n_query_points` 78124 (4 %), epochs 3000, seeds 42/7/1337, demos 26/27/28 | doubled supervision arms | `train_iclr_jhu_xcube_sup_s*.sh` | ABL (seed triplet) |
| `config_smoke_spec.yaml` | = `spec02` with epochs 2, bs 4, n_query 8192, spectral 0.05, compile off, demo 96 | smoke | `smoke_spec.sh` | SMOKE |
| `config_iclr_firebench.yaml` | F via pre-xcube `config_pointcloud_ffm.yaml`: FireBench u10, cond [0,1,2], n_obs 1000-20000, bs 8, k 64, `measurement_ops` (noise 0-0.2, slab p 0.4, dropout 0.5), shuffled split | DMF-Gen FireBench v1, demo 7 | `train_iclr_firebench.sh` | LEGACY |
| `config_iclr_firebench_v2.yaml` | = v1; dense H5, epochs 12000, ops softened (noise 0-0.1, p 0.2, dropout 0.3), rff_lengthscale 0.05, demo 8 | v2 | `train_iclr_firebench_v2.sh` (block split, gap 10) | LEGACY |
| `config_iclr_firebench_v3.yaml` | merged u10u12 H5, n_obs max 36772 (1 %), n_query 36772, k 16, logit-normal t, compile, demo 11 | v3 | `train_iclr_firebench_v3.sh` | LEGACY |
| `config_iclr_firebench_v4.yaml` | = v3 with epochs 3500, demo 18 (launcher adds `reflect_y` aug) | v4 | `train_iclr_firebench_v4.sh` | LEGACY |
| `config_iclr_firebench_v5clean.yaml` | = v4 with `measurement_ops` **removed** (null), demo 31 | v5 clean | `train_iclr_firebench_v5clean.sh`, `engaging/train_fb_v5clean_engaging.sh` | LEGACY (last FireBench arm) |
| `config_iclr_wing.yaml` | F + `dataset: shiftwing`, `processed_root`, pool budgets `n_obs_min [64,16,16,16]` / `max [4096,1024,1024,1024]`, `measurement_ops` (ball occlusion), k 64, bs 8, demo 6 | DMF-Gen wing surface->volume | `train_iclr_wing.sh` | LEGACY |
| `config_iclr_wing_v2.yaml` | = wing, epochs 8000, demo 9 | | `train_iclr_wing_v2.sh` | LEGACY |
| `config_iclr_wing_v3.yaml` | = wing, processed_v3, k 16, wd 3e-5, logit-normal, epochs 4000, demo 13 | | `train_iclr_wing_v3.sh` | LEGACY |
| `config_iclr_wing_v4.yaml` | = v3, demo 19, save_dir `iclr_wing_v4_sym` | | `train_iclr_wing_v4.sh` | LEGACY |

### C.1.2 `Save_config/kolmogorov2d/` (+ `_seed7`, `_seed1337`, `_fullbudget`) — PoF Kolmogorov fleet

| File | Derives from | Method / arm | Used by | Role |
|---|---|---|---|---|
| `kolmogorov2d/config_bench_kolm_ffm.yaml` | F via `config_iclr_jhu_xcube_spec02.yaml` ("frozen N29 re-instantiated at 2D"): data, N, `Num_x/Num_y` 256, `n_query` 1310, `rff_lengthscale` 0.05, `spectral_weight` 0, epochs 400, train_ratio 0.8, demo 101 | **DMF-Gen, Kolmogorov (paper row)** | `train_bench_kolm_ffm.sh` (default CONFIG), `submit_2d_fleet_delta.sh` | PAPER |
| `kolmogorov2d/config_bench_kolm_ffm_smoke.yaml` | = above, epochs 3, compile off, demo 999 | | `smoke_kolm2d.sh` | SMOKE |
| `kolmogorov2d/config_baseline_Det_kolm.yaml` | D via `config_baseline_Det.yaml` schema: dataset kolmogorov2d, `field_names`, cond [0], n_obs 65-655, epochs 2500 (geofno bs 32), demo 61 | **Senseiver, Kolmogorov (paper)** | `submit_2d_fleet_delta.sh` -> `train_kolm_baseline.sh` | PAPER |
| `kolmogorov2d/config_baseline_MLPRBF_kolm.yaml` | = `Det_kolm`, baseline_model mlp_rbf, `coord_dim` 3, demo 64 | **MLP-RBF (paper)** | same | PAPER |
| `kolmogorov2d/config_baseline_GeoFNO_kolm.yaml` | = `Det_kolm`, baseline_model geofno, demo 65 | Geo-FNO, **200k-step run** (budget error, paper §Budget) | same | PAPER (original run; the reported row comes from `_fullbudget`, see `src/fleet_select.py:33`) |
| `kolmogorov2d_fullbudget/config_baseline_GeoFNO_kolm_matched.yaml` | = `GeoFNO_kolm`, geofno epochs **625**, demo 265, reload false, separate save_root | **Geo-FNO budget-matched (the reported Kolmogorov Geo-FNO row)** | `eval_kolm_fleet.sh` (`kolmogorov2d_fullbudget` case), `geofno_matched_sweeps.sh`, `dump_kolm_geofno_matched.sh`, `figs_pof/make_operator_table.py`, `figs_pof/fig_perf_vs_sensors.py` — **no training launcher names it** (was launched with `CONFIG=` env override of `train_kolm_baseline.sh`) | PAPER |
| `kolmogorov2d/config_baseline_Det_kolm_smoke.yaml` | = `Det_kolm`, epochs 3, demo 998 | | `smoke_kolm2d.sh` | SMOKE |
| `kolmogorov2d/config_baseline_Gen_kolm.yaml` | G via `backup/config_baseline_Gen.yaml` lineage (AE 64/128, FM 64, 3 res blocks) with 2D dims, cond [0], n_obs 65-655, s1 200 ep / s2 1250 ep, demo 60 | **latent FM (paper)** | `submit_2d_fleet_delta.sh` (LFM_STAGE 1 then 2) | PAPER |
| `kolmogorov2d/config_baseline_S3GM_kolm.yaml` | = `Gen_kolm`, s3gm, epochs 160 @ bs 8, use_checkpoint true, demo 63 | **S3GM (paper)** | same | PAPER |
| `kolmogorov2d/config_baseline_SiT_kolm.yaml` | = `Gen_kolm`, sit, patch 8, epochs 650, demo 62 | **SiT (paper)** | same | PAPER |
| `kolmogorov2d/config_baseline_SiT_kolm_p4.yaml` | = `SiT_kolm`, patch_size 4, demo 66 | SiT patch-granularity arm (P2-9) | none by name | ABL |
| `kolmogorov2d/config_baseline_Gen_kolm_smoke.yaml` | = `Gen_kolm`, all epochs 2, demo 997 | | `smoke_kolm2d.sh` | SMOKE |
| `kolmogorov2d_seed7/*_s7.yaml` (7 files) | each = the canonical file with `seed` 7, demo +100, save_root `kolmogorov2d_seed7/` — **verified identical otherwise** | seed replicate | `submit_2d_seed_replicates_delta.sh` (pattern `_s$SEED`) | PAPER (seed-variance) |
| `kolmogorov2d_seed1337/*_s1337.yaml` (7 files) | each = canonical with `seed` 1337, demo +200 — verified | seed replicate | same | PAPER (seed-variance) |

### C.1.3 `Save_config/cylinder2d/` (+ `_seed7`, `_seed1337`, `_surface`, `_uonly`) — PoF cylinder fleet

| File | Derives from | Method / arm | Used by | Role |
|---|---|---|---|---|
| `cylinder2d/config_bench_cyl_ffm.yaml` | = `config_bench_kolm_ffm.yaml` with mesh H5, `field_names` [Ux,Uy,p], cond [0,1], n_obs 24-238, `n_query` 476, epochs 850, train_ratio 0.6667, `Num_x/Num_y` 400/200, demo 102 | **DMF-Gen, cylinder (paper)** | `train_bench_cyl_ffm.sh`, `submit_2d_fleet_delta.sh` | PAPER |
| `cylinder2d/config_bench_cyl_ffm_smoke.yaml` | = above, epochs 3, demo 996 | | `smoke_cyl2d.sh` | SMOKE |
| `cylinder2d/config_baseline_Det_cyl.yaml` | = `Det_kolm` with mesh H5, cond [0,1], n_obs 24-238, epochs 5000 (geofno 1300, modes 32x24), demo 71 | **Senseiver (paper)** | `submit_2d_fleet_delta.sh` -> `train_cyl_baseline.sh` | PAPER |
| `cylinder2d/config_baseline_MLPRBF_cyl.yaml` | = `Det_cyl`, mlp_rbf, coord_dim 3, demo 72 | **MLP-RBF (paper)** | same | PAPER |
| `cylinder2d/config_baseline_GeoFNO_cyl.yaml` | = `Det_cyl` but **grid H5 (80,000 pts)**, n_obs **80-800**, demo 75 | **Geo-FNO (paper)** | same | PAPER |
| `cylinder2d/config_baseline_Gen_cyl.yaml` | = `Gen_kolm` but grid H5, cond [0,1], n_obs 80-800, s1 350 ep / s2 2600 ep, s3gm 340 ep @ bs 8, sit 1300 ep, use_checkpoint true, demo 70 | **latent FM (paper)** | same | PAPER |
| `cylinder2d/config_baseline_S3GM_cyl.yaml` | = `Gen_cyl`, s3gm, demo 74 | **S3GM (paper)** | same | PAPER |
| `cylinder2d/config_baseline_SiT_cyl.yaml` | = `Gen_cyl`, sit, demo 73 | **SiT (paper)** | same | PAPER |
| `cylinder2d/config_baseline_SiT_cyl_p4.yaml` | = `SiT_cyl`, patch 4, demo 76 | SiT patch arm | none by name | ABL |
| `cylinder2d/config_baseline_Det_cyl_smoke.yaml`, `config_baseline_Gen_cyl_smoke.yaml` | epochs 3 / 2, demos 995 / 994 | | `smoke_cyl2d.sh` | SMOKE |
| `cylinder2d_seed7/*_s7.yaml`, `cylinder2d_seed1337/*_s1337.yaml` (7 + 7) | canonical + seed, demo +100/+200 — verified identical otherwise | seed replicates | `submit_2d_seed_replicates_delta.sh` | PAPER (seed-variance) |
| `cylinder2d_surface/config_bench_cyl_surface_ffm.yaml` | = `bench_cyl_ffm` + `sensor_pool: "surface"`, cond **[0,1,2]**, n_obs **32-360**, demo 103 | **DMF-Gen surface-to-field (paper §surface)** | `submit_cyl_surface_fleet_delta.sh` (CONFIG=, DEMO_NUM=103) | PAPER |
| `cylinder2d_surface/config_baseline_{Det,MLPRBF,GeoFNO,Gen,S3GM,SiT}_cyl_surface.yaml` | each = the canonical cylinder file + `shared.conditioning.sensor_pool: "surface"`, cond [0,1,2], n_obs 32-360, vis 128, demo 8x | surface-to-field fleet (paper) | `submit_cyl_surface_fleet_delta.sh` | PAPER |
| `cylinder2d_surface/*_smoke.yaml` (3) | epochs 3, demos 991-993 | | `smoke_cyl_surface.sh` | SMOKE |
| `cylinder2d_uonly/config_bench_cyl_uonly_ffm.yaml` | = `bench_cyl_ffm` with cond **[0]** (Ux only), demo 104 | DMF-Gen observe-u-only (P2-10) | none by name (launched via CONFIG=/DEMO_NUM= env; `eval_kolm_fleet.sh` has a `cylinder2d_uonly` case) | ABL |
| `cylinder2d_uonly/config_baseline_Det_cyl_uonly.yaml` | = `Det_cyl` with cond [0], demo 91 | Senseiver observe-u-only | same | ABL |

---

## C.2 The YAMLs that exist in both trees: key-by-key walk-through of `git diff 828c3c6 HEAD`

Four upstream yamls existed at the fork point. Three still exist at the same path; the RAM one survives only under `backup/`.

### C.2.1 `config_baseline_Det.yaml` (`git log 828c3c6..HEAD`: single commit `cf42d1b` "ICLR campaign state: 3D protocols, probabilistic eval, spectral loss, perf work", 2026-08-25)

| Key | Upstream 828c3c6 | HEAD | Why (source) |
|---|---|---|---|
| `shared.demo_num` | 10 | 20 | new run id for the JHU Senseiver |
| `shared.device_ids` | [2] | [0] | single-GPU SLURM nodes |
| `shared.reload` | false | true | resume-by-default on the cluster |
| `shared.paths.data_path` | `Dataset/Merged_CH4COTU1P.h5` | `Dataset/JHU_TurbulenceDataset.h5` | dataset swap to JHU (`817654b` "Adapted FFM and baseline models to JHU dataset") |
| `shared.paths.save_root` | `Save_TrainedModel` | `Save_TrainedModel/JHU/baseline_senseiver` | per-dataset run tree |
| `shared.paths.config_backup_root` | `Save_config/det_baseline` | `Save_config/isotropic_turbulence/det_baseline` | same |
| `shared.data.num_x/num_y` (+`num_z`) | 403 / 100 (no z) | 125 / 125 / **125** | 125^3 cube; `num_z` presence switches the 3D code path (`src/model_baseline.py:5170-5171`) |
| `shared.conditioning.cond_fields` | [2, 3] (T, U_1) | [0, 2] (Ux, Uz) | JHU identifiability protocol: Uy, p never observed (paper §datasets, `main.tex:242-243`) |
| `n_obs_min_list` / `n_obs_max_list` | [192,192] / [384,384] | [1953] / [19531] | fractional protocol 0.1 %-1 % of 125^3 (comment in `config_pointcloud_ffm.yaml:136-137`) |
| `vis_cond_fields` / `vis_n_obs_list` | [2,3] / [256,256] | [0,2] / [19531,19531] | visualisation only |
| `senseiver_params.training.epochs` | 10000 | 4000 | JHU budget |
| `senseiver_params.training.batch_size` | 128 | 20 | 125^3 memory |
| `senseiver_params.training.n_query_points` | 4096 | 19531 | 1 % query subset, matches DMF-Gen's `n_query_points` at the time |
| `senseiver_params.architecture.coord_dim` | 2 | 3 | 3D coordinates |
| `senseiver_params.architecture.latent_dim` | 128 | 256 | capacity for 3D (upstream/main independently also went to 256, see below) |
| `mlp_rbf_params.*`, `geofno_params.*` | unchanged | unchanged | inert here (only the selected `baseline_model` block is read, `src/model_baseline.py:4794-4812`) |

Upstream/main (f163ad7) also moved this file since the fork: `demo_num` 1, `device_ids` [0], senseiver `epochs` 6000, `batch_size` 144, `latent_dim` 256 — i.e. upstream stayed on the combustion dataset; the fork's version is unrelated to those edits.

### C.2.2 `config_baseline_Gen.yaml` (commits: `56df602`, `8048971`, `7c5e94a` merge, `2acd8a8` "Reverted yamls to original", `cf42d1b`)

| Key | Upstream 828c3c6 | HEAD | Why |
|---|---|---|---|
| header comment block | present | removed | cosmetic |
| `baseline_model` | "sit" | latent_fm | fork's Gen file is the latent-FM entry point (SiT got its own `config_baseline_SiT_*.yaml`) |
| `shared.device_ids` / `reload` | [1] / false | [0] / true | cluster defaults |
| `shared.paths.*` | combustion paths | `Dataset/JHU_TurbulenceDataset.h5`, `Save_TrainedModel/JHU/baseline_latent_fm`, `Save_config/isotropic_turbulence/latent_fm` | JHU |
| `shared.data.num_x/num_y/num_z` | 403/100/- | 125/125/125 | JHU |
| `cond_fields` | [2] | [0, 2] | JHU protocol |
| `n_obs_min_list` / `max` | [192] / [384] | [1953] / [19531] | 0.1 %-1 % |
| `vis_cond_fields` / `vis_n_obs_list` | [2] / [256] | [0,2] / null (defaults to max) | |
| `latent_fm_params.stage1.training.epochs` / `batch_size` | 10000 / 128 | 5000 / 8 | 3D memory |
| `latent_fm_params.stage1.architecture.base_ch` / `latent_ch` | 128 / 128 | **48 / 48** | parameter parity with DMF-Gen (~6.5 M), comment "combined Stage 1+2 params ~= 6.53M" (`config_baseline_Gen.yaml:99`) |
| `latent_fm_params.stage2.training.batch_size` | 64 | 8 | memory |
| `stage2.architecture.base_ch` | 128 | **38** | parity |
| `stage2.sampling.benchmark_n_steps` | [2,4,8] | [2,4,16] | in-training eval NFE list |
| `s3gm_params.*`, `sit_params.*` | unchanged | unchanged | inert |

Note `backup/config_baseline_Gen.yaml` is an intermediate JHU version (AE 64/128, FM 64, `num_res_blocks` 3, `ema_decay` 0.0, cond [0,1,2,3], stage 2) — it is **not** the upstream original; the upstream original is preserved as `config_baseline_Gen_Linzheng_original.yaml`. The 2D PoF Gen configs (`kolmogorov2d/config_baseline_Gen_kolm.yaml`) descend from the `backup/` lineage (64/128/64/3 res blocks), not from the root JHU file (48/48/38/2).

### C.2.3 `config_pointcloud_ffm.yaml` (commits `817654b`, `56df602`, `d005de1`, `8048971`, `7c5e94a`, `2acd8a8`, `cf42d1b`)

| Key | Upstream 828c3c6 | HEAD | Why |
|---|---|---|---|
| header | "try GL_rbf_ENH ... 128-384 sensors per field ... RAM fine-tuning" | "GL_rbf backbone scaled to ~20.4M ... n_obs_max_list 5000->19531 (1 % budget)" + "3: increased n_query_points. topk_rbf_glres gather mode from topk_rbf. gather_topk from 32 to 128." | history comments; note the header still describes a **20.4 M-param / latent 384 / 256 latents / 8 blocks** variant that the body does **not** contain (body is 256/128/4); the second line says "gather_topk from 32 to 128" while the body says 64. Stale comments (C.4-11) |
| `Demo_Num` | 19 | 3 | |
| `device_ids` | [1] | [0] | |
| `data` | `Dataset/Merged_CH4COTU1P.h5` | `../Dataset/JHU_TurbulenceDataset.h5` | JHU (`817654b`) |
| `save_dir` | `Save_TrainedModel/ffm_tc_pointcloud` | `Save_TrainedModel/JHU/pointcloud_ffm/ffm_tc_pointcloud_match_lfm` | "match_lfm" = parameter-matched to latent FM (`8048971` "pointcloud now matches baseline parametrization") |
| `batch_size` | 128 | 20 | 3D memory |
| `field_names` | absent | ["Ux","Uy","Uz","p"] | new key (JHU H5 has no field_names attr) |
| `gather_topk` | **32** | **64** | see header comment; later reduced to 16 in the xcube family (C.3.4) |
| `n_query_points` | 4096 | 19531 | 1 % of 125^3 |
| `query_sampling` | "obs_mix" | "uniform" | drops the near/far-of-sensor mixture (the `query_sample_*_ratio` keys become inert) |
| `save_every` | 500 | 200 | |
| `cond_fields` | [2, 3] | [0, 2] | JHU protocol |
| `n_obs_min_list` / `max` | [192,192] / [384,384] | [1953] / [19531] | 0.1 %-1 % of N |
| `vis_n_obs_list` | [256] | null | defaults to max |
| `benchmark_n_steps` | [2,4,8] | [2,4,16] | |
| everything else (`backbone GL_rbf_ENH`, latent 256/128/4 blocks/8 heads, `sensor_local_topk` 16, `gather_mode topk_rbf_glres`, `rff_lengthscale` 0.15, `prior rff`, lr 1e-4, wd 1e-6, epochs 10000, seed 42, train_ratio 0.9) | | unchanged | |

Upstream/main since the fork changed the same file differently (combustion `Merged_COTU0U1P.h5`, `latent_dim` 128, bs 144, cond [1] CO-only, `query_sampling uniform`, `benchmark_n_steps` [1,2,4]) — no overlap with the fork's edits except `query_sampling`.

### C.2.4 `config_pointcloud_ffm_ram.yaml`

Deleted from the root in the fork; `backup/config_pointcloud_ffm_ram.yaml` is **byte-identical** to `828c3c6` (`git diff` empty). Upstream/main has since rewritten it substantially (LoRA on all linear layers, rank 16, `reward_transform top_bottom`, pairwise SWD on, lr 2e-4, ...). Never used in the PoF campaign. Upstream/main additionally added `config_pointcloud_ffm_direct_posttrain.yaml` (direct-coherence post-training template; `gather_topk` 32, `sensor_local_topk` 16, cond [2] at exactly 256 sensors) — **no fork analogue**, not used here.

---

## C.3 Comparison tables

### C.3.0 Glossary of the sensor / k / neighbour keys and where the code reads them

DMF-Gen / PointCloudFFM (flat yaml -> `argparse` namespace; unknown keys are printed as "not a recognized argument. Ignoring." — `src/train_pointcloud_ffm.py:801-809`):

| Key | Meaning | Consumer |
|---|---|---|
| `gather_mode` | how a query gathers sensor tokens: `rbf` (softmax over **all** sensors), `topk_rbf*` (softmax over the k nearest), `topk_rbf_glres` (top-k + global residual scaffold; used everywhere here) | `src/Model.py:740`, `1370-1400` |
| `gather_topk` | **k for the query-side kNN gather**; `k = min(gather_topk, n_sensors)` | `src/Model.py:741`, `1390` (via `train_pointcloud_ffm.py:1077`) |
| `sensor_local_topk` | k for the **sensor-to-sensor** local graph (each sensor token attends to its k nearest sensors; search k+1 and drop self) | `src/Model.py:789`, `1310-1325` |
| `gather_multiscale`, `gather_topk_coarse`, `gather_coarse_sigma_scale` | optional second, coarser kNN gather (k_coarse, sigma x scale) added to the local term | `src/Model.py:755-763`, `1422-1431` |
| `neighbor_backend` | `keops` = KeOps kNN (GPU), `torch` = dense | `src/Model.py:744`, `1147-1150` |
| `rbf_sigma`, `learnable_rbf_sigma` | RBF bandwidth of the gather (learned log-sigma when true) | `src/Model.py:714`, `1170`, `1365` |
| `cond_fields`, `n_obs_min_list`, `n_obs_max_list` | observed fields and per-field sensor-count range | `src/train_pointcloud_ffm.py:392-395`, `1248-1249` -> `src/helpers.py:507-590` |
| `sensor_pool` | `null` = anywhere; `"surface"` = restrict to H5 `surface_indices` | `src/train_pointcloud_ffm.py:900,910` -> `src/helpers.py:213-257` |
| `n_query_points`, `query_sampling` | Monte-Carlo query subset per step (uniform here; `obs_mix` would use the `query_sample_*_ratio` keys) | `src/train_pointcloud_ffm.py:1250-1251`; `calib_step_bench.py:53` |
| `prior`, `rff_features`, `rff_lengthscale`, `prior_k_min/max`, `prior_slope` | GP source distribution (RFF; `rff_kolmogorov` = spectrally sloped variant) | `src/train_pointcloud_ffm.py:965-975`; `evaluate_ffm.py:188-194` |
| `measurement_ops.*` | training-time corruption of the drawn sensors: Gaussian noise sigma in [min,max] (z-score units), slab/ball occlusion with prob and fraction range, whole-field dropout prob | `src/train_pointcloud_ffm.py:1205-1206`, `1261` -> `src/measurement_ops.py:150-177` |
| `spectral_weight/_block/_bins/_window` | binned radial-spectrum loss on a structured block; **auto-disabled unless N is a perfect cube** (or `AUG_GRID_SHAPE` set) | `src/train_pointcloud_ffm.py:940-957` |
| `t_sampling` | flow-matching time sampling (`logit_normal` vs uniform) | `src/train_pointcloud_ffm.py:1190-1192`; `Model.py:2087` |
| `ema_decay`, `use_amp`, `compile_model` | EMA of weights, bf16 autocast, torch.compile | `train_pointcloud_ffm.py` (grep) |
| `benchmark_n_steps`, `ode_solver`, `n_steps_generation` | NFE list for **in-training** validation curves only; paper NFE is set by the eval launchers (C.3.6) | `train_pointcloud_ffm.py:729` region |
| `Num_x/Num_y`, `fno_*`, `condition_blur*` | only for `backbone: fno`/`fno3d` (`train_pointcloud_ffm.py:1102-1136`); inert for GL_rbf_ENH |
| `hidden_dim`, `cond_dim`, `field_embed_dim`, `USE_FOURIER_PE`, `fourier_pe_*` | point/sensor encoders shared by mlp_rbf and GL_rbf (`train_pointcloud_ffm.py:1063-1064`) |
| `sigma_min` | legacy FFM key passed to `PointCloudFFM(..., sigma_min=)` but unused by rectified flow (comment in yaml; `train_pointcloud_ffm.py:990,1101`) |

Baseline trainers (nested yaml; `src/model_baseline.py:4755-4812` `normalize_baseline_config` / `resolve_stage_config`):

| Key | Meaning | Consumer |
|---|---|---|
| `shared.conditioning.{cond_fields,n_obs_min_list,n_obs_max_list,sensor_pool}` | same draw as DMF-Gen, same `build_sparse_condition` | `src/model_baseline.py:4762-4766`, `5175-5200` (latent FM), `5438-5443` (senseiver), `5532-5537` (mlp_rbf), `5588-5593` (geofno) |
| `<model>_params.training.n_query_points` | query subset per step for senseiver / mlp_rbf (default 4096) | `src/model_baseline.py:5414`, `5517` |
| `mlp_rbf_params.architecture.rbf_sigma` | fixed softmax-RBF bandwidth over **all** sensors — the MLP-RBF baseline has **no k** (full softmax, `src/model_baseline.py:382`) | adapter `src/model_baseline.py:8013-8022` |
| `senseiver_params.architecture.{num_latents,latent_dim,...}` | Perceiver-IO latent bottleneck (no kNN) | Senseiver adapter |
| `geofno_params.architecture.geofno_variant` | `fno` = rasterise sensors to the grid (value + mask channels); `irregular` would use `latent_Nx/Ny`, `gno_radius`, `gno_transform_type` — those four keys are **inert** in every config here | `src/model_baseline.py:8107-8136` |
| `latent_fm_params.stage2.conditioning.cond_mode` | `image` = masked field + mask rasterised to the AE latent grid | `src/model_baseline.py:2466-2537` |
| `sit_params.architecture.node_subsample`, `cond_fill_sigma` | (pointnet tokenizer, JHU only) token budget per step; nearest-sensor fill bandwidth | `src/model_baseline.py:7577`, `7701-7702` |
| `sit_params.sampling.sampling_N`, `s3gm_params.sampling.sampling_N` / `n_corrector_steps` / `snr` | sampler steps used at eval for SiT (50 / 32) and S3GM (200 predictor + 5 corrector) | `eval_kolm_fleet.sh:138` comment ("sit/s3gm: config sampling_N") |
| `shared.overrides.*` | global epochs/bs/lr/wd override, all null everywhere | `src/model_baseline.py:4809-4812` |
| `shared.logging.eval_every/save_every` | fallback only; the per-model `training.eval_every/save_every` win (`setdefault`, `model_baseline.py:4806-4807`) |

Classical anchors (not yaml-driven): IDW **k = 8** (`src/baseline_classical_2d.py:590`, `baseline_classical_jhu.py:340`; paper `main.tex:443`), gappy POD rank 80 and 20 (`baseline_classical_2d.py:596`; paper `main.tex:445`).

### C.3.1 Upstream combustion default (N = 40,300; fork point 828c3c6)

| Config | N | Sensor spec (raw) | Fraction of N (per field) | k / neighbour keys | Observed fields | Noise | Split | Epochs / steps | Batch | LR / WD | Model dims | NFE / sampler | Seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `config_pointcloud_ffm.yaml` @828c3c6 (DMF-Gen) | 40,300 | `n_obs_min [192,192]`, `max [384,384]`; vis 256 | **0.476 %-0.953 %** per field (vis 0.635 %) | `gather_topk 32`, `sensor_local_topk 16`, `gather_mode topk_rbf_glres`, `rbf_sigma 0.05` learnable, keops | [2,3] = T, U_1 (of CH4,CO,T,U_1,p) | none | `train_ratio 0.9`, **shuffled** (no env) | 10000 ep | 128 | 1e-4 / 1e-6 | GL_rbf_ENH: latent 256, 128 latents, 4 blocks, 8 heads, ff x4; hidden 256, cond 128, field_embed 128; Fourier PE 32 bands; `n_query_points 4096` (10.2 % of N), `obs_mix` | rff prior (256 feats, ls 0.15); Euler, `benchmark_n_steps [2,4,8]`, fallback 32 | 42 |
| `config_baseline_Det.yaml` @828c3c6 (senseiver) | 40,300 | [192,192] / [384,384]; vis [256,256] | 0.476 %-0.953 % | none (Perceiver-IO) | [2,3] | none | 0.9 shuffled | 10000 | 128 | 1e-4 / 1e-6 | 128 latents x 128 dim, 3 enc layers x 3 self-attn, 8 heads, field_embed 32, 32 space bands; `n_query_points 4096` | deterministic | 42 |
| same file, mlp_rbf block | | same | | none (full softmax, `rbf_sigma 0.05`) | | | | 10000 | 128 | 1e-4 / 1e-6 | hidden 256, cond 128, field_embed 128, coord_dim 3, Fourier PE | det | 42 |
| same file, geofno block | | same | | none (grid raster) | | | | 10000 | 128 | 1e-4 / 1e-6 | fno, modes 24x12, 64 ch, 4 layers | det | 42 |
| `config_baseline_Gen.yaml` @828c3c6 (sit) | 40,300 | [192] / [384]; vis [256] | 0.476 %-0.953 % | none | [2] = T only | none | 0.9 shuffled | 10000 | 32 | 5e-5 / 1e-4 | patch 4, hidden 256, depth 8, 4 heads, `cond_mode interp` | `sampling_N 50`, Euler; `benchmark_n_steps [4,16,32,64]` | 42 |
| same file, latent_fm block | | same | | none | | | | s1 10000 / s2 10000 | 128 / 64 | 1e-4 / 2e-4 | AE base 128, latent 128, 3 levels; FM base 128, ch_mult [1,2], 2 res, 8 heads, ema 0.999, `cond_mode image` | Euler `[2,4,8]` | 42 |
| same file, s3gm block | | same | | none | | | | 10000 | 48 | 1e-4 / 1e-6 | nf 128, ch_mult [1,2,4,8], 2 res, attn @8, dropout 0.1; sigma 0.1-7, 1000 scales | `sampling_N 200`, snr 0.128, 5 corrector, `alpha_obs 1.0` | 42 |

### C.3.2 Kolmogorov 2D (N = 65,536; 2,560 train / 640 held-out frames; launcher exports `JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0`)

Paper protocol (`main.tex:769-771`, `tab_kolm` caption): vorticity observed at **1 % = 655** sensors, eval on 50 frames of the 8 held-out trajectories, K=8, identical seeded draws; sweep {65, 164, 655, 1965, 6554} = {0.1, 0.25, 1, 3, 10} % (`eval_kolm_fleet.sh:133`).

| Config (demo) | Sensor spec (raw, per field) | Fraction of N | k / neighbour keys | Obs. fields | Noise (train) | Split | Epochs -> steps | Batch | LR / WD | Model dims | NFE / sampler (train-eval list; paper eval) | Seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **upstream** `config_pointcloud_ffm.yaml`@828c3c6 (for reference) | 192-384 of 40,300 | 0.48-0.95 % | topk 32 / local 16 | T, U_1 | none | 0.9 shuffled | 10000 | 128 | 1e-4/1e-6 | 256/128/4 | [2,4,8] | 42 |
| `kolmogorov2d/config_bench_kolm_ffm.yaml` DMF-Gen (101) | `[65]` - `[655]` | **0.099 % - 0.999 %** | **`gather_topk 16`**, `sensor_local_topk 16`, `topk_rbf_glres`, keops, learnable sigma (init 0.05) | [0] vorticity | none | 0.8 block gap 0 | 400 ep x 128 = **51,200** | 20 | 1e-4 / 1e-6 | GL_rbf_ENH 256 latent / 128 latents / 4 blocks / 8 heads; hidden 256; `n_query_points 1310` (2.0 % of N), uniform; rff ls **0.05**, 256 feats; `spectral_weight 0`; `t_sampling logit_normal`; EMA 0.9995; AMP; compile | train-eval [2,4,16]; **paper NFE 4**, K 8 | 42 (7, 1337 replicates: demos 201, 301) |
| `config_baseline_Det_kolm.yaml` Senseiver (61) | `[65]` - `[655]`; vis 655 | 0.099-0.999 % | none | [0] | none | 0.8 block | 2500 x 20 = **50,000** | 128 | 1e-4 / 1e-6 | 128 latents x **128** dim, 3x3 layers, 8 heads, field_embed 32, 32 bands, coord_dim 2; `n_query_points 4096` (6.25 % of N) | det (K=1) | 42 (161, 261) |
| `config_baseline_MLPRBF_kolm.yaml` (64) | same | same | none (full softmax RBF, sigma 0.05 fixed) | [0] | none | 0.8 block | 2500 x 20 = 50,000 | 128 | 1e-4 / 1e-6 | hidden 256, cond 128, field_embed 128, coord_dim 3, Fourier PE; `n_query_points 4096` | det | 42 (164, 264) |
| `config_baseline_GeoFNO_kolm.yaml` (65) | same | same | none (rasterised) | [0] | none | 0.8 block | 2500 x **80 = 200,000** (budget error, paper `main.tex:376-377`) | **32** | 1e-4 / 1e-6 | fno, modes 32x32, 64 ch, 4 layers | det | 42 (165, 265) |
| `kolmogorov2d_fullbudget/config_baseline_GeoFNO_kolm_matched.yaml` (265) | same | same | none | [0] | none | 0.8 block | **625 x 80 = 50,000** | 32 | same | same | det | 42 |
| `config_baseline_Gen_kolm.yaml` latent FM (60) | same | same | none (`cond_mode image`) | [0] | none | 0.8 block | s1 200 x 320 = 64,000; s2 1250 x 40 = 50,000 | 8 / 64 | 1e-4 / 1e-6; 1e-4 / 0 | AE base 64, latent 128, 3 levels; FM base 64, ch_mult [1,2], **3** res blocks, 8 heads, ema 0.999 | train-eval [2,4,8,16]; **paper NFE 4** | 42 (160, 260) |
| `config_baseline_S3GM_kolm.yaml` (63) | same | same | none | [0] | none | 0.8 block | 160 x 320 = 51,200 | 8 | 1e-4 / 1e-6 | nf 128, [1,2,4,8], 2 res, attn@8, dropout 0.1, checkpointing; sigma 0.1-7.0, 1000 scales, ema 0.999 | `sampling_N 200` + 5 corrector, snr 0.128 (paper: "S3GM at 200") | 42 (163, 263) |
| `config_baseline_SiT_kolm.yaml` (62) | same | same | none (`cond_mode interp`) | [0] | none | 0.8 block | 650 x 80 = 52,000 | 32 | 5e-5 / 1e-4 | patch **8** (32x32 tokens), hidden 256, depth 8, 4 heads, ema 0.9999, huber 0.1 | `sampling_N 50` (paper: "SiT at 50 steps") | 42 (162, 262) |
| `config_baseline_SiT_kolm_p4.yaml` (66) | same | same | none | [0] | none | 0.8 block | 650 | 32 | same | patch **4** (4096 tokens) | 50 | 42 |
| classical floors (`run_classical_kolm.sh` -> `baseline_classical_2d.py`) | 655 fixed | 1.0 % | IDW **k=8**; POD rank 80 / 20 | [0] | eval-op only | n/a | n/a | | | | det | eval seed |

### C.3.3 Cylinder 2D (mesh N = 23,800; grid N = 80,000; 1,200 train Re{60,100,150,200} / 600 held-out Re{80,250}; `train_ratio 0.6667`, block, gap 0)

Paper protocol (`main.tex:553-556`, `563-565`): (u,v) observed at **1 % = 238 per field on the mesh**, p never observed; learned rows 50 Re-stratified frames, K=8. Eval launcher `eval_kolm_fleet.sh:57-61` sets `NOBS=238 COND="0 1"` for **every** model, including the grid-trained ones (`eval_kolm_ensemble.py:50-56`: "count-matched (same n_obs) across families").

| Config (demo) | N (train file) | Sensor spec (raw, per field) | Fraction of N (train) | Eval count / fraction | k keys | Obs. fields | Noise | Epochs -> steps | Batch | LR / WD | Model dims | NFE | Seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cylinder2d/config_bench_cyl_ffm.yaml` DMF-Gen (102) | 23,800 mesh | `[24]`-`[238]` (broadcast to both fields) | **0.101 % - 1.000 %** per field; 2 fields -> 48-476 readings total | 238 / 1.0 % | `gather_topk 16`, `sensor_local_topk 16` | [0,1] Ux,Uy (p unobserved) | none | 850 x 60 = 51,000 | 20 | 1e-4 / 1e-6 | as Kolmogorov; `n_query_points 476` (2.0 %); rff ls 0.05 | [2,4,16]; paper NFE 4 | 42 (202, 302) |
| `config_baseline_Det_cyl.yaml` Senseiver (71) | 23,800 mesh | [24]-[238]; vis 238 | 0.101-1.0 % | 238 / 1.0 % | none | [0,1] | none | 5000 x ~9.4 = ~47,000 | 128 | 1e-4/1e-6 | 128x128 Perceiver, coord_dim 2; `n_query_points 4096` (**17.2 %** of N) | det | 42 (171, 271) |
| `config_baseline_MLPRBF_cyl.yaml` (72) | 23,800 mesh | same | same | 238 / 1.0 % | none (full softmax) | [0,1] | none | 5000 | 128 | 1e-4/1e-6 | coord_dim 3; `n_query_points 4096` | det | 42 (172, 272) |
| `config_baseline_GeoFNO_cyl.yaml` (75) | **80,000 grid** | `[80]`-`[800]`; vis 800 | 0.1-1.0 % of grid | **238 / 0.298 % of grid** (0.299 % of the 79,684 fluid cells) | none | [0,1] | none | 1300 x 37.5 = ~48,750 | 32 | 1e-4/1e-6 | fno, modes **32x24**, 64 ch, 4 layers | det | 42 (175, 275) |
| `config_baseline_Gen_cyl.yaml` latent FM (70) | 80,000 grid | [80]-[800] | 0.1-1.0 % of grid | 238 / 0.298 % | none | [0,1] | none | s1 350 x 150 = 52,500; s2 2600 x 18.75 = ~48,750 | 8 / 64 | 1e-4/1e-6; 1e-4/0 | AE 64/128/3; FM 64, [1,2], 3 res, 8 heads | [2,4,8,16]; paper NFE 4 | 42 (170, 270) |
| `config_baseline_S3GM_cyl.yaml` (74) | 80,000 grid | same | same | 238 / 0.298 % | none | [0,1] | none | 340 x 150 = 51,000 | 8 | 1e-4/1e-6 | as Kolmogorov, checkpointing on | 200 (+5) | 42 (174, 274) |
| `config_baseline_SiT_cyl.yaml` (73) | 80,000 grid | same | same | 238 / 0.298 % | none | [0,1] | none | 1300 x 37.5 = ~48,750 | 32 | 5e-5/1e-4 | patch 8 (25x50 = 1250 tokens), hidden 256, depth 8 | 50 | 42 (173, 273) |
| `config_baseline_SiT_cyl_p4.yaml` (76) | 80,000 grid | same | same | | none | [0,1] | none | 1300 | 32 | same | patch 4 (5000 tokens) | 50 | 42 |
| **surface** `cylinder2d_surface/config_bench_cyl_surface_ffm.yaml` (103) | 23,800 mesh, pool **360** | `sensor_pool surface`, `[32]`-`[360]` | **8.9 % - 100 % of the 360-cell ring** = 0.134 % - 1.51 % of N | eval sweep 32/64/128/360 taps per field (`eval_kolm_fleet.sh:72-77`) | topk 16 / local 16 | **[0,1,2]** all three at the taps | none | 850 | 20 | same | same | NFE 4 | 42 |
| surface `config_baseline_{Det,MLPRBF}_cyl_surface.yaml` (81, 82) | 23,800 mesh, pool 360 | [32]-[360]; vis 128 | 8.9-100 % of ring | 32/64/128/360 | none | [0,1,2] | none | 5000 | 128 | | as canonical | det | 42 |
| surface `config_baseline_{GeoFNO,Gen,S3GM,SiT}_cyl_surface.yaml` (85, 80, 84, 83) | **80,000 grid, pool 62** | [32]-[360] but **capped at 62** by `helpers.py:587` | 32 = 51.6 % of pool; anything >= 62 = the whole 62-cell pool (0.078 % of N) | 32 then 62 for 64/128/360 (launcher comment `eval_kolm_fleet.sh:81-84`: 128 and 360 reproduce 64 bit for bit) | none | [0,1,2] | none | as canonical | | | | | 42 |
| **u-only** `cylinder2d_uonly/config_bench_cyl_uonly_ffm.yaml` (104) | 23,800 mesh | [24]-[238] | 0.1-1 % of N, **one** field | 238 (`eval_kolm_fleet.sh:67-70`) | 16 / 16 | **[0]** Ux only | none | 850 | 20 | | | 4 | 42 |
| u-only `config_baseline_Det_cyl_uonly.yaml` (91) | 23,800 mesh | same | same | 238 | none | [0] | none | 5000 | 128 | | | det | 42 |
| classical (`run_classical_cyl*.sh`) | mesh | 238 fixed (surface: 32..360) | 1 % | IDW k=8; POD 80/20 | [0,1] / [0,1,2] | | | | | | det | |

### C.3.4 JHU 3D isotropic turbulence (N = 1,953,125; frames 0-149 = cubes 0-2 train, 150-199 = cube 3 held out; `train_ratio 0.75`, block, gap 0)

Paper protocol (`main.tex:242-243`, `1013-1015`; `DATA_TRANSFER_MANIFEST.md:108`): Ux and Uz observed at **19,531 sensors each (1 %)**, Uy and p never observed; K=8, NFE 4, n=50 snapshots (`eval_jhu_insample_dmf.sh:44`, `calib_nfe_array.sh:33-34`). The earlier `eval_xcube.sh:36` leg used K 8, **NFE 16**, 8 snapshots.

| Config (demo) | Sensor spec (raw, per field) | Fraction of N | k keys | Obs. fields | Noise / ops (train) | Aug (launcher env) | Split | Epochs -> steps | Batch | LR / WD | Model dims | Prior / spectral / t | NFE (train-eval) | Seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **upstream** `config_pointcloud_ffm.yaml`@828c3c6 | 192-384 of 40,300 | 0.48-0.95 % | topk 32 / local 16 | T,U_1 | none | none | 0.9 shuffled | 10000 | 128 | 1e-4/1e-6 | 256/128/4; n_query 4096 obs_mix | rff ls 0.15 / - / uniform | [2,4,8] | 42 |
| root `config_pointcloud_ffm.yaml` (3) | `[1953]`-`[19531]` | 0.1 - 1.0 % | **topk 64** / local 16 | [0,2] | none | none | **0.9 shuffled** | 10000 | 20 | 1e-4/1e-6 | 256/128/4; n_query 19531 (1 %) uniform | rff 0.15 / - / uniform | [2,4,16] | 42 |
| `config_iclr_jhu_main.yaml` (4) | same | same | topk 64 | [0,2] | none | none | 0.9 shuffled | 4000 x ~9 = ~36k | 20 | same | same | same | [2,4,16] | 42 |
| `config_iclr_jhu_robust.yaml` (5) | `[488]`-`[9765]` | **0.025 - 0.5 %** | topk 64 | **[0,1,2,3]** | noise sigma 0-0.2, slab occl p 0.3 (10-35 %), field-dropout 0.5 | none | 0.9 shuffled | 4000 | 20 | same | same | same | [2,4,16] | 42 |
| `config_iclr_jhu_xcube.yaml` (4; run dir DemoN4) | [1953]-[19531] | 0.1-1.0 % | **topk 16** / local 16 | [0,2] | none | none | 0.75 block | 6000 x 8 = 48,000 | 20 | 1e-4/1e-6 | 256/128/4 blocks/8 heads; **n_query 39062 (2.0 %)** uniform | rff 0.15 / 0 / **logit_normal**; EMA 0.9995; AMP; compile | [2,4,16] | 42 |
| `..._xcube_aug.yaml` (15) | same | same | 16 | [0,2] | none | octahedral | 0.75 block | 6000 | 20 | | same | same | | 42 |
| `..._xcube_sym.yaml` (17) | same | same | 16 | | none | octahedral,so3 | | 6000 | | | | | | 42 |
| `..._xcube_ti.yaml` (21) | same | same | 16 | | none | octahedral_proper,so3,translate | | 6000 | | | | | | 42 |
| `..._xcube_local.yaml` (20) | same | same | 16 | | none | octahedral,so3 | | 6000 | | | **16 latents**, no query-latent readout | | | 42 |
| `..._xcube_ms.yaml` (4; launcher 14) / `_ms_aug.yaml` (16) | same | same | 16 **+ coarse k 64**, sigma x4 | | none | none / octahedral | | 6000 | | | | | | 42 |
| `..._xcube_spec.yaml` (22) | same | same | 16 | | none | octahedral_proper,so3,translate | | 6000 | | | | spectral **0.05** (block 32, 12 bins) | | 42 |
| **`..._xcube_spec02.yaml` (29) = paper DMF-Gen** | `[1953]`-`[19531]` | **0.1-1.0 %** | **16 / 16** | [0,2] | none | octahedral | 0.75 block | 6000 x 8 = **48,000** (paper: 48k) | 20 | 1e-4 / 1e-6 | 256 latent / 128 latents / 4 blocks / 8 heads; 6.51 M params; n_query 39,062 (2 %) | rff 256 feats ls 0.15 / **spectral 0.02** / logit_normal; EMA 0.9995 | [2,4,16]; **paper NFE 4, K 8** | 42 |
| `..._xcube_specwin.yaml` (35) | same | same | 16 | | | octahedral | | 6000 | | | | spectral 0.02 windowed | | 42 |
| `..._xcube_cq.yaml` (33) | same | same | 16 | | | octahedral | | 6000 | | | backbone GL_rbf_CQ | spectral 0.02 | | 42 |
| `..._xcube_kprior.yaml` (34) | same | same | 16 | | | octahedral | | 6000 | | | | prior `rff_kolmogorov` 1024 feats, k 1-48, slope 1.667; spectral 0.02 | | 42 |
| `..._xcube_reg_s{42,7,1337}.yaml` (23/24/25) | same | same | 16 | [0,2] | noise 0-0.1, slab p 0.2 (10-30 %), dropout 0.3; attn/mlp/sensor-local dropout 0.1; wd **1e-2** | octahedral | 0.75 block | 3000 | 20 | 1e-4 / 1e-2 | | (args.json: spectral 0.0) | | 42 / 7 / 1337 |
| `..._xcube_sup_s{42,7,1337}.yaml` (26/27/28) | same | same | 16 | | none | octahedral | | 3000 | | | n_query **78,124 (4 %)** | | | 42/7/1337 |
| (no yaml on disk) `iclr_jhu_k32` (37) / `k64` (38) | same (args.json) | same | **topk 32 / 64** | [0,2] | | octahedral | 0.75 block | 6000 | 20 | | | spectral 0.02, logit_normal | | 42 |
| (no yaml on disk) `iclr_jhu_scale` (29, run dir `iclr_jhu_scale_DemoN29`) | same | | 16 | | | octahedral | **train_ratio 0.96** | 1600 | | | | | | 42 |
| `config_baseline_Det_xcube_aug.yaml` Senseiver (31) | [1953]-[19531]; vis [19531,19531] | 0.1-1.0 % | none | [0,2] | none | octahedral | 0.75 block | 4000 ep (paper budget table `main.tex:1050`: **79.2k** steps; not = 4000 x 150/20, so the trainer's steps/epoch is not frames/batch — see `audit_train_budgets.py`) | 20 | 1e-4/1e-6 | 128 latents x **256** dim, 3x3, 8 heads, coord_dim 3; n_query 19,531 (1 %) | det | 42 |
| `config_baseline_Gen_xcube_aug.yaml` latent FM (23) | same | same | none (`cond_mode image`) | [0,2] | none | octahedral | 0.75 block | s1 5000 ep / s2 10000 ep (`final_summary.json`: 10000 completed; paper `main.tex:1044`: **209k + 95k** steps) | 8 / 8 | 1e-4/1e-6; 2e-4/0 | AE **48/48**/3; FM base **38**, [1,2], 2 res, 8 heads, ema 0.999 (~6.53 M) | [2,4,16]; paper NFE 4 | 42 |
| `config_baseline_SiT_xcube.yaml` SiT-point (41) | [1953]-[19531]; vis [19531] | same | none (nearest-sensor fill, `cond_fill_sigma 0.05`) | [0,2] | none | octahedral | 0.75 block | 6000 ep (paper `main.tex:1045`: 38.5k steps) | 24 | 5e-5/1e-4 | pointnet tokenizer, hidden 256, **depth 6**, 4 heads, `node_subsample 8192`, huber_beta 0 (MSE) | `sampling_N 32`, `[32]` (paper: "239 sequential 8,192-token chunks at 32 steps") | 42 |
| `config_baseline_CoNFiLD_xcube.yaml` (23, arm C) | [1953]-[19531]; vis [19531] | same | none | [0,2] | none | 48 signed axis permutations (`confild_params.stage1.augmentation.groups`) | 0.75 block | s1 wall-clock 13.5 h (epochs cap 1e5), s2 5.5 h (steps cap 2e6) | 8 / 8 | dec 1e-4, latent 1e-5; s2 5e-5 | SIREN 256 hidden x 15 layers, latent 1024; UNet 32 ch, 1 res, 4 heads; 6.62 M | diffusion 1000 steps | 42 |

### C.3.5 FireBench and wing (ICLR-era; not in the PoF paper)

| Config (demo) | N | Sensor spec (raw, per field) | Fraction | k keys | Obs. fields | Ops (train) | Aug | Split | Epochs | Batch | LR / WD | n_query | Seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `config_iclr_firebench.yaml` (7) | 3,677,184 | `[1000]`-`[20000]` | 0.027 - 0.544 % | topk 64 / 16 | [0,1,2] u,v,w (theta, rho_f unobserved) | noise 0-0.2, slab p 0.4 (10-35 %), dropout 0.5 | none | 0.9 **shuffled** | 4000 | 8 | 1e-4/1e-6 | 19,531 (0.53 %) | 42 |
| `..._v2.yaml` (8) | same (dense H5) | same | same | 64 | | noise 0-0.1, p 0.2, dropout 0.3 | none | 0.9 block gap 10 | 12000 | 8 | | 19,531; rff ls 0.05 | 42 |
| `..._v3.yaml` (11) | 3,677,184 merged | `[1000]`-`[36772]` | 0.027 - **1.0 %** | **16** / 16 | | same as v2 | none | 0.9 block gap 10 | 12000 | 8 | | 36,772 (1 %); logit_normal | 42 |
| `..._v4.yaml` (18) | same | same | same | 16 | | same | reflect_y (`AUG_GRID_SHAPE=152,126,192`) | same | 3500 | 8 | | same | 42 |
| `..._v5clean.yaml` (31) | same | same | same | 16 | | **none** (`measurement_ops: null`) | reflect_y | same | 3500 | 8 | | same | 42 |
| `config_baseline_Det_firebench.yaml` (36) | same | `[1000]`-`[36772]`; vis 36772 x3 | same | none | [0,1,2] | none | reflect_y | **0.75** block gap 10 (80 train / 30 val; corrected 2026-09-22, see §D.3 — DMF-Gen above uses 0.9) | 4000 | **6** | 1e-4/1e-6 | 19,531 | 42 |
| `config_baseline_Gen_firebench.yaml` (35) | same | same | same | none | [0,1,2] | none | reflect_y | **0.75** (same as Senseiver) | s1 5000 / s2 10000 | 8 | | AE 48/48, FM 38 | 42 |
| `config_iclr_wing.yaml` (6) / `_v2` (9) | n/a (surface pool per case) | pool channels [Cp, tau_x, tau_y, tau_z]: `min [64,16,16,16]`, `max [4096,1024,1024,1024]` | n/a | topk 64 / 16 | `cond_fields [3]` "informational only in pool mode" | noise 0-0.1, **ball** occl p 0.2 (10-30 %), dropout 0.3 | none | stats.json split (600/73 cases) | 6000 / 8000 | 8 | 1e-4/1e-6 | 16,384 | 42 |
| `..._v3` (13) / `_v4` (19) | n/a | same | n/a | **16** / 16 | | same | v4: sym (save_dir) | same | 4000 | 8 | 1e-4/**3e-5**; logit_normal | 16,384 | 42 |
| `config_baseline_SiT_wing.yaml` (40) | n/a | inherits `[1953]-[19531]`? — **no**: shared cond is [0,2] / 1953-19531 copied from the JHU SiT file, which is meaningless for the wing pool (C.4-12) | | none | | none | | | 10000 | 32 | 5e-5/1e-4 | patch 4, depth 8 | 42 |

### C.3.6 What actually determines the reported numbers (evaluation side, for reconciliation with the paper)

| Regime | Eval driver | Sensor count(s) | Observed fields | K | NFE / steps | Frames | Seeds |
|---|---|---|---|---|---|---|---|
| Kolmogorov | `src/eval_kolm_fleet.sh:53-56` -> `eval_kolm_ensemble.py` | 655 (1 %); DMF-Gen canonical seed also 65/164/655/1965/6554 (`:133`) | [0] | 8 (`:162`) | DMF-Gen & latent FM `--nfe 4` (`:135-137`); SiT `sampling_N` 50; S3GM 200 | 50 evenly spaced of 640 (`--n-frames 50`) | `--seed 0 --op-seed 1000` |
| Cylinder | same, `:57-61` | 238 per field for **all** models (grid models therefore at 0.30 % of their grid) | [0,1] | 8 | same | 50, stratified over the 2 Re blocks | same |
| Cylinder surface | `:72-77` | 32 / 64 / 128 / 360 taps per field (grid models capped at 62) | [0,1,2] | 8 | same | 50 | same |
| Cylinder u-only | `:67-70` | 238 | [0] | 8 | | 50 | |
| JHU | `ensemble_eval.py` via `eval_jhu_insample_dmf.sh:44`, `calib_nfe_array.sh:33-34`, `eval_crps_all.sh:50-51` | `--n-obs 19531 19531` (1 % per field); sweep 19531 / 195312 (`calib_k32.sh`) | `--cond-fields 0 2` | 8 | 4 (earlier `eval_xcube.sh:36`: 16) | 50 (earlier: 8) | |
| Operator study (Kolmogorov) | `OPERATOR="--sensor-noise 0.1|0.3"`, `"--sensor-occlusion 0.25"` (`eval_kolm_fleet.sh:87-93`) | 655 | [0] | 8 | | 50 | op-seed 1000 |

---

## C.4 Things that look inconsistent or deserve a second look

1. **Shuffled-split configs still at the top level.** `config_iclr_jhu_main.yaml`, `config_iclr_jhu_robust.yaml`, `config_baseline_Det.yaml`, `config_baseline_Gen.yaml`, `config_pointcloud_ffm.yaml`, `config_iclr_firebench.yaml` all carry `train_ratio 0.9` and their launchers (`train_iclr_jhu_main.sh`, `train_iclr_jhu_robust.sh`, `train_iclr_firebench.sh`, `train_baseline_lfm.sh`, `train_iclr_det_senseiver.sh`) do **not** export `JHU_SPLIT_MODE`, so `helpers.py:261` falls back to `"shuffle"`. These are the leaky pre-xcube runs. `train_pointcloud_ffm.py`'s `--config` **default** is the root `config_pointcloud_ffm.yaml`, so running the trainer without `--config` reproduces a shuffled-split JHU run with `gather_topk 64`. Consider pruning or renaming them `_legacy`.

2. **k differs between the DMF-Gen generations.** Upstream: `gather_topk 32`. Root/ICLR-main/robust/firebench-v1/wing-v1: **64**. Every PoF-paper run (xcube family, Kolmogorov, cylinder, surface, u-only) and firebench v3+/wing v3+: **16**. `sensor_local_topk` is 16 everywhere. The paper text (`main.tex:512`) says "a local k-nearest-sensor retrieval per query" without stating k; the reported k is 16, and the k32/k64 ablation runs exist on disk (`iclr_jhu_k32_DemoN37`, `iclr_jhu_k64_DemoN38`, args.json verified) but their yamls (`config_iclr_jhu_k32.yaml`, `_k64.yaml`, `_scale.yaml`) are **missing from `Save_config/`** although `train_iclr_jhu_k32.sh`, `_k64.sh`, `_scale.sh` reference them. If the k-sweep is cited anywhere, the configs need to be reconstructed from the run dirs' `args.json`.

3. **Cylinder grid-trained models are trained at 80-800 sensors (0.1-1 % of 80,000) but evaluated at 238 (0.30 % of the grid)** to be count-matched with the mesh rows (`eval_kolm_ensemble.py:50-56`). The paper caption says "1 % = 238 sensors per field on the 23,800-cell mesh"; that is true for the mesh rows and count-true for the grid rows, but the grid rows see a **3.4x lower density on their own discretisation** than the density they were trained around. Worth a sentence in the paper (the note at `main.tex:598-600` only covers the pairing, not the density).

4. **Surface task: grid pool is 62 cells vs mesh pool 360.** For Geo-FNO / latent FM / S3GM / SiT the `n_obs` range `[32]-[360]` is capped to 62 (`helpers.py:587`), so their "360-tap" and "128-tap" columns are identical to the 64-tap column (acknowledged in `eval_kolm_fleet.sh:81-84`). Make sure the surface table footnotes this rather than presenting 4 distinct densities for those rows.

5. **Senseiver / MLP-RBF `n_query_points 4096` is a very different fraction per regime** (6.25 % of Kolmogorov, 17.2 % of the cylinder mesh, 1 % on JHU) while DMF-Gen uses a fixed 2 % (1310 / 476 / 39,062). The paper's "random ~2 % of points per step" (`main.tex:513`) describes DMF-Gen only; the deterministic point baselines supervise 3-9x more points per step in 2D.

6. **Senseiver capacity differs between regimes**: `latent_dim` 128 in the 2D configs (= upstream default) vs 256 on JHU. Not necessarily wrong, but the 2D Senseiver is the smaller model. Similarly latent FM is 64/128/64/3-res in 2D (from the `backup/` lineage) but 48/48/38/2-res on JHU (parity-matched); and SiT is patch-8 depth 8 in 2D vs pointnet depth 6 in 3D.

7. **Kolmogorov Geo-FNO original run = 200 k steps** (`epochs 2500` at bs 32) — already disclosed in the paper (`main.tex:376-377`, `831-833`); the reported row is the `_fullbudget` rerun (`src/fleet_select.py:33`). But: the seed replicates `kolmogorov2d_seed7/config_baseline_GeoFNO_kolm_s7.yaml` and `_seed1337/..._s1337.yaml` still carry `epochs 2500` at bs 32 (200 k steps), so the Geo-FNO seed-variance numbers are at 4x the canonical budget while the canonical row is at 1x.

8. **Fullbudget Geo-FNO: first launch had the 625 epochs on the wrong block.** Run-time snapshot `kolmogorov2d_fullbudget/det_baseline/Baseline_geofno_Stage1_DemoN265_20260909_212527.yaml` has `mlp_rbf_params.training.epochs 625` and `geofno_params.training.epochs 2500` (i.e. the edit landed in the wrong section); the relaunch `..._20260910_063432.yaml` is correct. `dump_kolm_geofno_matched.sh:24` and the eval JSONs point at the `20260910_063432` run — verify nothing in `figs_pof/` still resolves the run dir by glob and picks the earlier one (`resolve_run_dir` in `eval_kolm_fleet.sh:110-125` uses `tail -1` of an `ls`, which sorts the correct one last; fine, but fragile).

9. **Demo-number collision: `demo_num 265` is both the fullbudget Geo-FNO (`kolmogorov2d_fullbudget/`) and the seed-1337 Geo-FNO (`kolmogorov2d_seed1337/`).** Different `save_root`, so no clash on disk, but any tool that keys on `DemoN265` alone (log names, `fleet_jobs_delta.txt`) is ambiguous. Same pattern for demo 23 (`config_iclr_jhu_xcube_reg_s42` DMF-Gen, `config_baseline_Gen_xcube_aug` latent FM, `config_baseline_CoNFiLD_xcube`) and demo 4 (`config_iclr_jhu_main`, `config_iclr_jhu_xcube`, `config_iclr_jhu_xcube_ms` all `Demo_Num: 4`, with the launchers overriding to 12/14 via `--Demo-Num` but the run dirs on disk being `DemoN4`).

10. **Launchers reference 13 yamls that do not exist** (17 before the 2026-09-22 pull; verified with `[ -f ]`): `config_iclr_jhu_k32/k64/scale.yaml`, `config_baseline_fno3d_xcube.yaml` + `_smoke` (the FNO3D paper row!), `config_baseline_Senseiver_iclr.yaml` + `_smoke`, `config_baseline_DeepONet_iclr.yaml` (DeepONet paper row), `config_baseline_SiT_xcube_matched.yaml`, `config_baseline_SiT_smoke_archive.yaml`, `config_baseline_CoNFiLD_figsmoke.yaml`, `config_pointcloud_ffm_ram.yaml` (root), `pointcloud_ffm/config_pointcloud_ffm_DemoN3_20260713_081738.yaml` (`fb_det_eng.yaml`, `fb_lfm_eng.yaml`, `fb_v5clean_eng.yaml`, `jhu_temporal_eng.yaml` arrived with the 2026-09-22 pull — Section D). Some probably never left Kestrel (`*.yaml` is gitignored; only force-added files travelled). For the FNO3D and DeepONet rows the only surviving specification is the run dir's `args.json` / `run_config.yaml` under `Save_TrainedModel/JHU/baseline_fno` and `baseline_deeponetpp`.

11. **Stale header comments in the JHU DMF-Gen family.** Every `config_iclr_jhu_xcube*.yaml`, `config_iclr_firebench*.yaml` and the root `config_pointcloud_ffm.yaml` still carry "GL_rbf backbone scaled to ~20.4M ... latent_dim 256->384, num_latents 128->256, num_latent_blocks 4->8" and "gather_topk from 32 to 128" (`config_iclr_jhu_xcube_spec02.yaml:8-14`), while the bodies are 256/128/4 blocks and k 16 (confirmed in every `args.json`). A reader auditing from the comments alone would get the wrong architecture. Same for `Num_x: 403 / Num_y: 100` (combustion grid) left in all JHU/FireBench DMF-Gen yamls — inert for GL_rbf_ENH but misleading.

12. **`config_baseline_SiT_wing.yaml` conditioning block is copied from the JHU SiT file** (`cond_fields [0,2]`, `n_obs 1953-19531`, `vis_n_obs_list` null) although the wing dataset is a per-case surface pool with Cp/shear tap budgets (`config_iclr_wing_v4.yaml:75-79`). Whether `train_Gen_Baseline.py` even honours the pool for shiftwing is not visible from the yaml; legacy, but don't cite it.

13. **Keys present but never read** (for the paper configs): (a) in every DMF-Gen yaml with `backbone GL_rbf_ENH`: `Num_x`, `Num_y`, `fno_modes_x/y`, `fno_hidden_channels`, `fno_n_layers`, `condition_blur*` (fno-only; `train_pointcloud_ffm.py:1102-1136`), `query_sample_near/far/sigma_ratio` (obs_mix-only; all paper configs use `uniform`), `sigma_min` (legacy), `n_steps_generation` (fallback only when `benchmark_n_steps` is absent), `vis_cond_fields`/`vis_n_obs_list` (visualisation), `decode_chunk_size`/`gather_query_chunk_size` (memory only). (b) In every deterministic baseline yaml: the two `*_params` blocks not selected by `baseline_model`, plus `geofno_params.architecture.{latent_Nx,latent_Ny,gno_radius,gno_transform_type}` (irregular variant only, `model_baseline.py:8107-8136`), and `shared.overrides.*` (all null). (c) In every generative baseline yaml: the two unselected `*_params` blocks; `sit_params.transport.loss_weight` (null); `latent_fm_params.stage2.conditioning.pointnet_hidden_mult` (only for `cond_mode pointnet`). (d) `shared.data.num_x/num_y` in the cylinder **mesh** Det configs (comment admits "unused by the point-based adapters"). None of these affect results; they just make the yamls look more specific than the runs were.

14. **`spectral_weight` cannot act on 2D data**: `train_pointcloud_ffm.py:940-954` only builds the spectral grid when N is a perfect cube or `AUG_GRID_SHAPE` is exported; the 2D configs set 0.0 anyway (comment line 10 of `config_bench_kolm_ffm.yaml`), so the 2D DMF-Gen rows are "N29 minus the spectral term" — the paper's "as published" claim should say so if the N29 spectral term is considered part of the method.

15. **`rff_lengthscale` differs between 3D and 2D DMF-Gen** (0.15 vs 0.05, justified in comments `config_bench_kolm_ffm.yaml:88` / `config_bench_cyl_ffm.yaml:89`). This is a per-regime prior hyper-parameter that the baselines have no analogue of; disclose it alongside k if the paper claims the 2D runs are "the frozen N29 configuration re-instantiated at 2D".

16. **Training sensor range vs eval count.** All paper rows train on `U{0.1 %..1 %}` per field and are evaluated at exactly 1 % (top of the range) in 2D and 3D. The density sweep on Kolmogorov goes to 3 % and 10 % (1965, 6554), i.e. **outside the training range** for every learned model — the non-monotonic degradation of Geo-FNO / latent FM noted in commit `81561d4` is partly an extrapolation effect and the paper §density should say the models never saw >1 %.

17. **Cylinder `n_obs_min 24` is 0.1008 % and Kolmogorov `65` is 0.0992 %** — rounding of 0.1 % in different directions; immaterial, but if the text says "0.1 %" for both, fine.

18. **`config_baseline_Det_cyl.yaml` batch 128 on 1,200 train frames gives 9.375 steps/epoch**; whether the loader drops the last partial batch decides 47 k vs 50 k steps. The comment says "~10 steps/epoch -> 5000 epochs"; `audit_train_budgets.py` should be the source of truth for the paper's "within 4 % of 50 k" statement.

19. **`backup/` is mixed provenance**: `Det` and `ram` are pristine upstream, `Gen` is not (C.1.1). If `backup/` is meant to be "the upstream originals", replace `backup/config_baseline_Gen.yaml` with `git show 828c3c6:.../config_baseline_Gen.yaml` (or note that `config_baseline_Gen_Linzheng_original.yaml` is that file).

20. **Two config files per JHU baseline have no launcher that names them** (`config_baseline_CoNFiLD_xcube.yaml`; the SiT `_p4` arms; `kolmogorov2d_fullbudget/...matched.yaml`; both `cylinder2d_uonly/` files). They were launched through `CONFIG=`/`DEMO_NUM=` env overrides of the generic launchers, which is fine, but the reproduction path is only recoverable from `fleet_jobs_delta.txt` / the run-time snapshot yamls, not from a script.

21. **JHU baseline optimizer-step counts are not derivable from the yamls.** With 150 train frames, `batch_size 20` and `epochs 4000`, the Senseiver yaml implies ~32 k steps, but the paper's budget table (`main.tex:1050`) reports 79.2 k; latent FM (`epochs 5000 / 10000`, bs 8) implies ~94 k + 188 k but the paper reports 209 k + 95 k (`main.tex:1044`). The run-dir `run_config.yaml` files match the yamls exactly (checked `Baseline_senseiver_Stage1_DemoN31_20260818_083446`, `Baseline_latent_fm_Stage2_DemoN23_20260818_153527`), so the difference is in how the baseline trainers count steps per epoch (query chunking / augmentation replication), not in the configs. `src/audit_train_budgets.py` should be treated as the source of truth for any "steps" statement, and the yaml `epochs` values should not be multiplied by frames/batch for the 3D baselines.


---

# Section D — The 2026-09-22 pull: Engaging campaign configs and results (41203e0..8aa01c9)

Conventions are those of §C.0 (sensor fractions are per observed field of N; JHU N = 1,953,125; FireBench N = 3,677,184) and the key glossary of §C.3.0; they are not repeated here. All paths are relative to `0_demo_TurbulentCombustion/` unless absolute. Every number quoted from a JSON below was read from the file on this host (scripts in the session scratchpad; nothing in the repo was modified).

## D.1 What this pull is

`git log 41203e0..HEAD` is 16 commits: 14 made on the **MIT Engaging** checkout between 2026-08-31 and 2026-09-05 (Claude-Session `01W4YVafaH41jm8Af5jfCuSZ`), the merge `1052090` (2026-09-22, "Merge origin/main (DeltaAI campaign) into Engaging main", parents `36e33f0` Engaging tip + `41203e0` Delta tip; merge base `b944ecf` 2026-08-31), and `8aa01c9` (2026-09-22) which force-adds, past the `*.yaml`/`Save_TrainedModel/` ignores, "every non-checkpoint, non-image file under 1 MB" of the Engaging run directories. What it covers is the **Sept 2026 Engaging campaign** — the ICLR-2027-era work that ran in parallel with the Delta PoF fleet: a CoNFiLD capacity study on the JHU cross-cube protocol (H200 training, H100 `node2906` canonical evals), two DMF-Gen training-seed replicates of the frozen N29 config, a DMF-Gen "temporal companion" (DemoN33) on a *different* JHU dataset, and FireBench (DMF-Gen v5clean, latent FM stage 1, Senseiver). Bytes: 1,771 files, ~41 MB (`Save_TrainedModel/JHU/baseline_confild` 33 MB, `JHU/pointcloud_ffm` 5 MB, `firebench` 3.3 MB). File-type tally under `Save_TrainedModel/`: 945 in-training `euler_nfe{2,4,16}_metrics.json` (K=1 diagnostics, never a reportable number — HANDOFF correction #6), 600 `crps_canonical_{best,last}_snap*.json` + 12 `*_summary.json` (the CoNFiLD canonical DPS evals), 22 `run_config.yaml` + 22 `run_metadata.json`, 14 `history.jsonl`, 11 `final_summary.json`, 11 `instrumentation*.json`, 10 `stage1_auto_decode.json` (frozen-decoder oracles), 9 `window_manifest.json`, 8 `args.json`, 8 `senseiver_metrics.json`, 2 `canonical_all50_nfe{4,2}_K8.json` (DemoN33), plus loss CSVs. **Not included**: every `*.pt` checkpoint and `*.png` (by design of `8aa01c9` and the new `.gitignore` lines 41-43), every `src/*.log` the launchers write, the Engaging datasets (`Dataset/JHU_TurbulenceDataset.h5` — the 617-frame single-cutout file DemoN33 trains on — is **not on this host**; `ls Dataset/` shows only `download_firebench.py`, `README`), and the Engaging venv itself (`requirements_engaging.txt`, 96 lines, is its frozen pip list). Also new: `HANDOFF.md` (451 lines) — this is the **2026-08-30 ICLR handoff** that the Delta side rewrote on 09-03 (`b3b4163`, "HANDOFF records PoF pivot") and deleted on 09-08 (`464c72e`, "Removed old MD files"); the merge kept the Engaging copy because its appended progress logs (lines 158-451: 08-31 launch, 09-01 environment incident, 09-02..05 results) are the only narrative record of these runs. Its header still says "updated 2026-08-30" and its §"Where the paper stands" describes `Paper/iclr2027/main.tex`; the live PoF handoff remains `HANDOFF_2026-09-12.md`.

**Previously-missing configs (§A.0.3, §C.4-10), re-checked with `[ -f Save_config/<f> ]` after the pull:**

| File | Status |
|---|---|
| `config_baseline_Senseiver_iclr.yaml`, `_smoke` | **still MISSING** |
| `config_baseline_S3GM_xcube.yaml` | **still MISSING** |
| `config_baseline_DeepONet_iclr.yaml`, `config_baseline_DeepONetPP_iclr.yaml` | **still MISSING** |
| `config_baseline_fno3d_xcube.yaml`, `_smoke` | **still MISSING** |
| `config_baseline_SiT_xcube_matched.yaml`, `config_baseline_SiT_smoke_archive.yaml` | **still MISSING** |
| `config_baseline_CoNFiLD_figsmoke.yaml` | **still MISSING** |
| `config_iclr_jhu_k32.yaml`, `_k64`, `_scale` | **still MISSING** |
| `config_pointcloud_ffm_ram.yaml` (root; the `.gitignore` even whitelists it, line 38) | **still MISSING** |
| `pointcloud_ffm/config_pointcloud_ffm_DemoN3_20260713_081738.yaml` | still MISSING |
| `fb_det_eng.yaml`, `fb_lfm_eng.yaml`, `fb_v5clean_eng.yaml`, `jhu_temporal_eng.yaml` | **now PRESENT** (+ `jhu_seedrep1379_eng.yaml`, `jhu_seedrep2718_eng.yaml`, which no launcher-reference scan had flagged) |

`find Save_TrainedModel -maxdepth 4 -name run_config.yaml -o -name args.json` under `Save_TrainedModel/JHU/` still returns **only** `baseline_confild/` and `pointcloud_ffm/` — there is still no `run_config.yaml`/`args.json` for the JHU Senseiver, FNO3D, DeepONet(++), S3GM or SiT rows, and the CoNFiLD arms **P** (`unified_published_prior`) and **F** (`unified_faithful384`) are still absent (their save_roots are not among the new `baseline_confild/*` directories). So the consequence stated in §A.0.3 stands for every paper row except CoNFiLD; for CoNFiLD the pull adds *new* Engaging arms, not the origin P/C/F run dirs. The `DATA_TRANSFER_TIER2.md:67` verification line ("expect latsweep_2048 / 4096 / strict2048 / improve" under `baseline_confild/`) refers to a *different*, origin-side set of directories that has not arrived either (see D.6-1).

## D.2 CoNFiLD capacity sweep

### D.2.1 Design, in the words of the commits and configs

Origin (Delta/Kestrel) arm **C** = `config_baseline_CoNFiLD_xcube.yaml` (§A.7 checklist #13): SIREN hidden 256 × 15 layers, latent 1024, UNet prior `num_channels 32` = 1,441,217 params, total 6,624,453 (+1.82 % vs the 6,506,253 comparison model; yaml lines 9-10, 92-94, 117). The Engaging study asks two questions in sequence (HANDOFF.md:211-216, 353-357):

1. **Stage 1 only (2026-08-31, Nick-approved):** is C's held-out codec ceiling latent-capacity-bound? Four arms hold everything of C fixed except `latent_dim` (and, for the strict arm, `hidden_features`), plus the `model_image_size` / `attention_resolutions` bookkeeping that must follow the latent (the attention string is image-size-relative: `"32,16,8"`@1024 ≡ `"64,32,16"`@2048 ≡ `"128,64,32"`@4096 ≡ `"256,128,64"`@8192, all giving `attention_ds [32,64,128]`; an unscaled string at 2048+ "silently loses ALL attention", HANDOFF.md:229-232). A fifth arm, `sweep8192`, was added on 09-02 (`5517b0b`). "**strict**" = the only arm that stays inside the ±10 % parameter budget: latent 2048 with `hidden_features` **144** instead of 256 "compensates the FiLM term's linear growth in latent_dim: decoder 5,032,948 + UNet prior 1,441,217 = 6,474,165 inference params (−0.49 % vs the 6,506,253 comparison model)" (`config_baseline_CoNFiLD_xcube_strict2048.yaml:2-7`). The width-256 arms at 2048/4096/8192 are +66 % / +195 % / +453 % over budget and "can NEVER be promoted to stage 2 (parameter_budget.enforce would refuse)" (`config_baseline_CoNFiLD_xcube_sweep2048.yaml:4`) — unless the gate is switched off, which is what happened next.
2. **Stage 2 with the budget explicitly lifted (Nick, 2026-09-02; HANDOFF.md:355-357):** "the ±10 % matched question was already answered (C = agg 0.871), so the question became CoNFiLD's UNCONSTRAINED ceiling." Commit `86eff67` found that "the stage-2 UNet is 1-D conv, so its size depends on num_channels/channel_mult and NOT on model_image_size. Every sweep arm therefore ran the SAME 1,441,217 parameter prior while the latent it models grew 1024 → 8192", added a `PRIOR_CH` axis to the stage-2 launcher, and measured the menu "at latent 2048: ch32=1,441,217 ch64=5,743,489 ch96=12,906,817 ch128=22,931,201 ch192=51,563,137 ch256=91,639,297". Stage 2 then ran at ch64/128/256 on the `sweep2048` stage 1 and at ch128/256 on the `sweep8192` stage 1, with `parameter_budget.enforce: false` (`BUDGET_ENFORCE=0`, `train_confild_stage2_engaging.sh:121-124`).

**What the two audit scripts check (docstrings quoted):**

* `src/check_confild_sweep_params.py:1-16` — "Pre-launch smoke test for the CoNFiLD stage-1 latent-dim sweep (Engaging). Instantiates, for each sweep config, exactly what the trainer will build -- the upstream SIRENAutodecoder_film decoder (confild_upstream_training.py:455-461), the 7200 x D latent table (:463), and the stage-2 diffusion prior via upstream create_model (:738-746) -- and asserts parameter counts to the digit, forward tensor shapes, and the architecture invariants (model_image_size == latent_dim, divisibility by the UNet downsample factor, attention_ds identical across arms so the deepest attention level is preserved at every latent size). [...] Exits nonzero on any mismatch. Training must not launch until this passes." Expected counts are hard-coded at lines 35-45 and cross-checked against a closed-form decoder formula (lines 50-54: `(3H+H) + L(H²+H) + (H·4+4) + (L+1)·D·H`). Its recorded output is `src/confild_smoke_21693407.out` (63 lines, "ALL CHECKS PASSED", four arms — `sweep8192` was added after this run and has no recorded smoke output; commit `5517b0b` quotes "decoder 34,543,364, latent table 58,982,400, prior 1,441,217" for it, and `final_summary.json` of the 8192 run confirms 34,543,364).
* `src/check_confild_prior_sizes.py:1-14` — "Report CoNFiLD stage-2 prior (UNet) parameter count vs num_channels. Context (2026-09-02): the prior was held at 1,441,217 parameters in EVERY sweep arm, because it is a 1-D conv UNet whose size depends on num_channels and channel_mult but NOT on model_image_size. So as the latent grew 1024 -> 8192, the prior modelling it never gained capacity. That is a prime suspect for the canonical result where strict2048 -- best codec of any arm -- produced the WORST conditional reconstruction (agg 1.316 vs CoNFiLD C's 0.871, pressure 2.818). This prints the menu of prior sizes so a scaled-prior run can be chosen against measured numbers rather than a guess." It instantiates `create_model` from `config_baseline_CoNFiLD_xcube_sweep2048.yaml` for `CHANNELS = [32, 64, 96, 128, 192, 256]` (lines 28-29). Neither script touches training code; both import the same upstream constructors the trainer uses (`confild_upstream_core.import_upstream_decoder`, `confild_upstream_training._import_upstream_diffusion`).

**Launch mechanics you need to know to read the run dirs** (`src/engaging/train_confild_sweep_engaging.sh`, `train_confild_stage2_engaging.sh`):
* Tracked "source" configs are `Save_config/config_baseline_CoNFiLD_xcube_{sweep1024,sweep2048,sweep4096,sweep8192,strict2048}.yaml` (Engaging paths, budgets 48600/19800). At each (re)launch the launcher `sed`s a **job copy** — `Save_config/confild_<ARM>_eng.yaml` for stage 1 (`train_confild_sweep_engaging.sh:104-108`), `confild_<ARM>[_pch<CH>]_s2_eng.yaml` for stage 2 (`train_confild_stage2_engaging.sh:112-134`) — replacing `data_path` with the node-local `/tmp/ntricard/confild[_s2]_<jobid>/JHU_4cubes_stride100.h5` staging copy and `wallclock_budget_s` with the **remaining** budget (total minus per-process maxima of `elapsed_seconds` in `history.jsonl`, lines 55-86 / 73-102). The `*_eng.yaml` files in the pull are therefore **the last overwrite**, not a per-run record; the per-run record is each run dir's `run_config.yaml`. Training is `train_Gen_Baseline.py --config <job copy> --training-stage {1,2} --reload` (lines 110-111 / 136-137), so every as-run `run_config.yaml` has `shared.reload: true` and stage 2's has `training_stage: 2` (the CLI flag overrides the yaml's `training_stage: 1`).
* Stage 1 ran on `mit_preemptable` (`--time=14:00:00`, `gpu:h200:1`, lines 3-8) as one job per arm after the chained-6h design and the 09-01 environment incident (HANDOFF.md:238-257, 259-307; jobs 21761937-40, 21818868). Stage 2: `--time=08:00:00`, same partition. Preemption/requeue keeps the Slurm job id, which is why one job id can own several run dirs (below).
* Stage-2 checkpoint selection: PRIOR_CH runs pin `stage2.stage1_checkpoint` to the stage-1 `last.pt` (`train_confild_stage2_engaging.sh:69, 132`; honoured by `model_baseline.py:8221-8227` ahead of the save_root glob); the strict run (no `PRIOR_CH`) went through the glob and took `best.pt` (`model_baseline.py:8230-8235`; `final_summary.json` of the strict Stage 2 records `.../Stage1_DemoN23_20260901_174028/best.pt`). For strict, `best.pt` and `last.pt` are the same epoch (2189), so this is a paper-trail difference only.
* Evals: frozen-decoder oracle `src/engaging/eval_confild_stage1_engaging.sh` → `evaluate_confild_stage1.py` on `last.pt` (primary, budget-matched) and `best.pt`, "snaps 150 151 153 162, 3000 steps, 3 restarts, seed 123" (line 14; args lines 79-86), gated on ≥ 48,300 s consumed (lines 45-75) → `<stage1 run>/Evaluation/stage1_oracle_{last,best}/stage1_auto_decode.json`. Canonical DPS eval `src/engaging/eval_confild_canonical_engaging.sh` → `confild_eval_unified.py --seed 0 --op-seed 1000 --n-snapshots 50 --K 8 --cond-fields 0 2 --n-obs 19531 19531 --dps-scale 1.0 --steps 1000 --row-chunk 4` (lines 61-69), pinned to `node2906` (`gpu:h100:1`, lines 7-8) because the canonical sensor draw is H100-SXM-bound (`ensemble_eval.py:41` `CANONICAL_FP = {snap 29, sensors 39062, idx_sum 37987162596}`), `JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0` (line 28), `--data $DEMO/Dataset/JHU_4cubes_stride100.h5` (the same 4-cube file as §C.3.4). It runs twice — `SEL=last` pairs stage-1 `last.pt` with stage-2 `last.pt`, `SEL=best` pairs the two `best.pt` — and writes into the **stage-2** run dir: `Evaluation/crps_canonical_{last,best}_summary.json` (50 per-snapshot aggregates + `mean` + `spectral_bands_mean` + protocol keys `n_scored 50, n_sensor_total 39062, window_length 32, windows [0,18], K 8, seed 0, op_seed 1000, smoke false`) and `crps_canonical_{last,best}_snap{0..49}.json` (each: `per_field {Ux,Uy,Uz,p} × {rel_l2_mean, rel_l2_single, crps, rmse, spread, spread_error_ratio, coverage_50, coverage_90}`, `aggregate`, `rank_hist`). That is the "~100 files" per Evaluation dir (102 = 2 × 51). The summary JSON has **no per-field means**; the per-field columns below are means over the 50 snap files.

### D.2.2 The arms

Common to every arm (verified identical in all 22 as-run `run_config.yaml`s and in the tracked yamls; only the keys in the next table differ from arm C): `seed 42`, `demo_num 23`, `train_ratio 0.75` (frames 0-149 train, 150-199 held out), `cond_fields [0,2]`, `n_obs [1953]-[19531]` = **0.1-1.0 % of N per observed field**, `vis [19531]`; stage 1 `epochs 100000` (cap), `wallclock_budget_s 48600`, `batch_size 8`, `points_per_item 16384`, decoder lr 1e-4 / latent lr 1e-5, `layers 15`, `tie_latent_to_hidden false`, `augmentation.groups 48` (→ 150 × 48 = 7,200 latent-table items; 900 latent steps per epoch at batch 8); stage 2 `steps 2000000` (cap), `wallclock_budget_s 19800`, `batch_size 8`, lr 5e-5, `cube_length 50`, `window_length 32`, `window_stride 1`, `diffusion_steps 1000`, `num_res_blocks 1`, `num_heads 4`, `num_head_channels -1`, `channel_mult "1,1,1,1,2,2"`, `ema 0.9999`; `parameter_budget {comparison_model 6506253, relative_tolerance 0.10}`.

**Tracked `*_eng.yaml` vs arm C (`config_baseline_CoNFiLD_xcube.yaml`), semantic key-by-key (67 keys each; comment-only differences ignored):**

| Job copy | Keys that differ from arm C (besides `upstream_root` → `/orcd/scratch/orcd/002/ntricard/baselines/CoNFiLD`, `data_path` → `/tmp/...`, `save_root`) |
|---|---|
| `confild_sweep1024_eng.yaml` (job 21761937) | none — an Engaging re-train of C's stage 1 |
| `confild_sweep2048_eng.yaml` (21761938) | `latent_dim 2048`, `model_image_size 2048`, `attention_resolutions "64,32,16"` |
| `confild_sweep4096_eng.yaml` (21761939) | `latent_dim 4096`, `model_image_size 4096`, `attention "128,64,32"`, **`stage1.wallclock_budget_s 47655`** (= 48600 − 945 s consumed before a preemption; the resume copy) |
| `confild_sweep8192_eng.yaml` (21818868) | `latent_dim 8192`, `model_image_size 8192`, `attention "256,128,64"` |
| `confild_strict2048_eng.yaml` (21761940) | `hidden_features 144`, `latent_dim 2048`, `model_image_size 2048`, `attention "64,32,16"` |
| `confild_strict2048_s2_eng.yaml` (21818806) | same four as strict; `num_channels` stays 32, `enforce` stays true |
| `confild_sweep2048_s2_eng.yaml` (21940482) | sweep2048 keys + `enforce false`, **`num_channels 256`**, `stage1_checkpoint .../sweep_ld2048_hf256/Baseline_confild_Stage1_DemoN23_20260901_173934/last.pt`, `save_root ..._pch256` — i.e. this file is the **pch256** copy; the pch64 and pch128 runs of the same arm used the same filename and were overwritten (see as-run diffs) |
| `confild_sweep8192_s2_eng.yaml` (21962017) | sweep8192 keys + `enforce false`, `num_channels 128`, `stage1_checkpoint .../sweep_ld8192_hf256/..._20260902_103121/last.pt`, `save_root ..._pch128`, budget 19800 — **the second-collision artifact** (`f55854e`); no surviving run dir corresponds to job 21962017 |
| `confild_sweep8192_pch128_s2_eng.yaml` (21972566) | as above, `wallclock_budget_s 8557` (resume copy) |
| `confild_sweep8192_pch256_s2_eng.yaml` (21972571) | sweep8192 keys + `enforce false`, `num_channels 256`, `save_root ..._pch256`, `wallclock_budget_s 8165` (resume copy) |

**As-run `run_config.yaml` vs the tracked job copy named in its `run_metadata.json:config_path`:** identical except `shared.reload false→true` and (stage 2) `training_stage 1→2`, **except** the three ld2048 prior variants, which all name `confild_sweep2048_s2_eng.yaml`: pch64's as-run has `num_channels 64`, `save_root ..._pch64`, `data_path .../confild_s2_21866077/...`; pch128's has `num_channels 128`, `..._pch128`, `.../confild_s2_21866078/...`, `stage2.wallclock_budget_s 17970` (= 19800 − 1830, its one resume); pch256's matches the tracked file. And the two aborted 8192 pch128 dirs carry budgets 19800 and 19647 (= 19800 − 153) against the tracked 8557. In other words: **trust `run_config.yaml`, not `*_s2_eng.yaml`, for `num_channels`/`save_root`/budget of any stage-2 run.**

**Run-dir table.** "Final" stage-2 dir = the one the eval launcher's `ls -d ... | tail -1` picks (`eval_confild_canonical_engaging.sh:50`), which is also the only one with `Evaluation/` and `final_summary.json`. Steps and epochs from `final_summary.json`; segments from `history.jsonl` (`elapsed_seconds` resets per process). Params: decoder / prior / combined-at-inference from `final_summary.json` (the latent table, 7,200 × D, is training-only).

| Arm (save_root under `Save_TrainedModel/JHU/baseline_confild/`) | Config file → run dir | Latent D / hidden H / layers | Decoder params | Stage-1 as run (budget 48,600 s) | Stage-1 oracle `last` (mean rel-L2 over frames 150,151,153,162; per-channel Ux/Uy/Uz/p) | oracle `best` |
|---|---|---|---|---|---|---|
| `sweep_ld1024_hf256` (= C's stage 1, Engaging retrain) | `config_baseline_CoNFiLD_xcube_sweep1024.yaml` → `Baseline_confild_Stage1_DemoN23_20260901_150532` (job 21761937, H200) | 1024 / 256 / 15 | 5,183,236 (+ prior 1,441,217 = 6,624,453) | 1,412 epochs, 48,607 s, 1 segment; 1,411 decoder steps, 1,270,800 latent steps; `best_train_rel_l2 0.2287` | **0.3333** (0.376 / 0.326 / 0.399 / 0.301), epoch 1411 | 0.3378 @ epoch 1399 |
| `sweep_ld2048_hf256` | `..._sweep2048.yaml` → `Stage1_DemoN23_20260901_173934` (21761938) | 2048 / 256 / 15 | 9,377,540 (10,818,757 with C prior, +66 %) | 1,423 ep, 48,619 s, 1 seg; 1,280,700 latent steps; 0.2023 | **0.2948** (0.332 / 0.288 / 0.360 / 0.258), ep 1422 | 0.2948 @ 1389 |
| `sweep_ld4096_hf256` | `..._sweep4096.yaml` → `Stage1_DemoN23_20260901_173934` (21761939; same second as the 2048 dir — different save_root, no clash) | 4096 / 256 / 15 | 17,766,148 (19,207,365, +195 %) | 1,504 ep, 945 + 47,665 = 48,611 s, **2 segments** (preempted once; `run_metadata.started_at 20260902_045454` is the resume); 1,335,600 latent steps; 0.1868 | **0.2702** (0.307 / 0.264 / 0.330 / 0.234), ep 1503 | 0.2706 @ 1489 |
| `sweep_ld8192_hf256` | `..._sweep8192.yaml` → `Stage1_DemoN23_20260902_103121` (21818868) | 8192 / 256 / 15 | 34,543,364 (35,984,581, +453 %) | 1,456 ep, 48,632 s, 1 seg; 1,310,400 latent steps; 0.1749 | **0.2462** (0.280 / 0.240 / 0.302 / 0.213), ep 1455 | 0.2476 @ 1399 |
| `strict_ld2048_hf144` | `..._strict2048.yaml` → `Stage1_DemoN23_20260901_174028` (21761940) | 2048 / **144** / 15 | 5,032,948 (6,474,165, −0.49 %) | **2,190 ep** (cheaper decoder ⇒ more epochs in the same wall-clock), 48,608 s, 1 seg; 1,971,000 latent steps; 0.1965 | **0.2836** (0.323 / 0.282 / 0.344 / 0.242), ep 2189 | same checkpoint (best = last = 2189) |

| Stage-2 arm (save_root) | Stage-1 ckpt used | Prior `num_channels` / params / combined | Stage-2 dirs (all `Baseline_confild_Stage2_DemoN23_*`) → final | Steps completed / wall-clock / segments | Canonical `last` (agg rel-L2-mean; Ux / Uy* / Uz / p*; CRPS; cov90) | Canonical `best` |
|---|---|---|---|---|---|---|
| `strict_ld2048_hf144` | strict `best.pt` (= last, ep 2189), via glob | **32** / 1,441,217 / 6,474,165 (`enforce true` passes at −0.49 %) | `20260902_102838` (job 21818806) — single dir | 630,377 / 19,794 s / 1 | **1.3157**; 0.688 / 1.030 / 0.727 / **2.818**; 0.782; 0.391 | 1.3645; 0.686 / 1.072 / 0.735 / 2.965; 0.809; 0.382 |
| `sweep_ld2048_hf256_pch64` | sweep2048 `last.pt` (ep 1422), pinned | 64 / 5,743,489 / 15,121,029 | `20260903_045138` (21866077) | 471,398 / 19,791 s / 1 | **0.8581**; 0.448 / 1.152 / 0.485 / 1.347; 0.423; 0.605 | 0.8505; 0.439 / 1.135 / 0.480 / 1.348; 0.431; 0.611 |
| `sweep_ld2048_hf256_pch128` | same | 128 / 22,931,201 / 32,308,741 | `20260903_045830` (21866078; `started_at 20260903_081949` = resume) | 204,378 / 1,830 + 17,953 = 19,783 s / 2 | **0.7992**; 0.434 / 1.151 / 0.474 / 1.138; 0.404; 0.583 | 0.8055; 0.439 / 1.157 / 0.485 / 1.141; 0.413; 0.567 |
| `sweep_ld2048_hf256_pch256` | same | 256 / 91,639,297 / 101,016,837 | `20260903_234224` (21940482) | 105,379 / 19,768 s / 1 | **0.8050**; 0.440 / 1.108 / 0.484 / 1.188; 0.403; 0.596 | 0.8148; 0.442 / 1.150 / 0.489 / 1.178; 0.413; 0.584 |
| `sweep_ld8192_hf256_pch128` | sweep8192 `last.pt` (ep 1455), pinned | 128 / 22,931,201 / 57,474,565 | `20260904_112551` (400 steps, 153 s, no ckpt) and `20260904_120442` (3,200 steps, 1,061 s, no ckpt) are **dead starts** of requeued job 21972566 (`save_every 5000`, so `--reload` found nothing and opened a new dir); final **`20260905_004554`** | 55,073 / 10,814 + 429 + 8,534 = 19,777 s / 3 | **0.7379**; 0.370 / 1.159 / 0.395 / 1.028; 0.351; 0.607 | 0.7331; 0.373 / 1.120 / 0.390 / 1.049; 0.349; 0.617 |
| `sweep_ld8192_hf256_pch256` | same | 256 / 91,639,297 / 126,182,661 | `20260904_113315` (1 record, 2 s — dead start of 21972571); final **`20260905_003109`** | 24,743 / 11,635 + 8,067 = 19,702 s / 2 (peak train GPU mem 85 GB) | **0.7554**; 0.382 / 1.143 / 0.407 / 1.090; 0.365; 0.626 | 0.7591; 0.382 / 1.121 / 0.409 / 1.124; 0.367; 0.632 |
| `sweep_ld1024_hf256`, `sweep_ld4096_hf256` | — | — | **no stage 2** in this pull | — | — | — |

(`*` = unobserved channel. Stage-2 GPU: all H200 except pch128@2048's resume segment on an "H200 NVL". Stage-2 `history.jsonl` records only `step, loss, elapsed_seconds`.)

### D.2.3 What the commit body concludes, and what the files support

Commit `bcd98e5` ("CoNFiLD capacity-study results: prior is the lever, and it saturates"): "Four-point prior sweep at fixed stage 1: 1.4M -> 22.9M cut the conditional aggregate 1.316 -> 0.799, and the next 4x to 91.6M bought nothing (0.805). ch256 still falls ~30% short of CoNFiLD P at comparable prior scale, and since C and P share a bit-identical stage 1 and differ only in the prior, P's edge is architectural rather than capacity [...] Also records the stage-1 latent curve (monotonic to 8192, ~8-9%/doubling, nearly free in compute because FiLM is per-item), the warning that the in-training proxy inverts rankings vs the oracle for large latents, and that Uy never responds to prior capacity -- pointing at DPS guidance as the remaining untested lever." The HANDOFF.md:389-395 table (ch32 1.316 / ch64 0.858 / ch128 0.799 / ch256 0.805 / "P (fleet)" 0.635) reproduces the **`canonical_last`** means above to three decimals, and the stage-1 table (HANDOFF.md:361-366: 0.3333 / 0.2948 / 0.2702 / 0.2462) the `stage1_oracle_last` means. Per-doubling stage-1 gains from the files: −11.6 %, −8.3 %, −8.9 %.

Two things the reader must add:

* **The "fixed stage 1" claim does not hold for the ch32 point.** There is no ch32 stage 2 on the `sweep_ld2048_hf256` codec in this pull (that save_root holds only Stage 1). The ch32 row's numbers (0.688 / 1.030 / 0.727 / 2.818 / 1.316 / 0.782 / 0.391) are exactly `strict_ld2048_hf144/.../Evaluation/crps_canonical_last_summary.json`, i.e. the **hidden-144** decoder (5.03 M) — a different stage 1 from the hidden-256 (9.38 M) codec under ch64/128/256. So "1.4M -> 22.9M cut 1.316 -> 0.799" mixes a decoder change with the prior change; the clean prior-only comparison in these files is ch64 → ch128 → ch256 at fixed sweep2048 stage 1 (0.858 → 0.799 → 0.805), plus origin C (ld1024/hf256/ch32, 0.871) as an off-host anchor. Note also that strict's *codec* is better than sweep2048's (oracle 0.2836 vs 0.2948) while its *pipeline* is far worse — consistent with the "prior/DPS is the bottleneck" reading, but it is the hf144 decoder's pipeline, not hf256's, that scored 1.316.
* **Relating to §A.7 arms P/C/F:** `sweep1024` is C's stage 1 re-trained on Engaging (bit-identical config; only site paths differ) but has **no** Engaging stage 2, so C's 0.871 is not reproduced here; none of the new arms is P (118.9 M published prior) or F (384-d); every new stage-2 arm is over the ±10 % budget except `strict2048` (ch32). The PoF table (`Paper/pof2026/main.tex:991-993`) still reports P/C/F from origin; the capacity study is not in the PoF text (grep for `91.6`, `22.9M`, `ch256` finds nothing), and its natural home is the ICLR draft (`Paper/iclr2027/`). The origin numbers cited by the commits/HANDOFF for C (agg 0.871; Uy 1.089 / p 1.487 in `train_confild_stage2_engaging.sh:14-19`) and P (agg 0.635) match `main.tex:991-992` but cannot be re-derived on this host (no origin CoNFiLD run dirs).

**The PRIOR_CH collision bugs (two, one layer apart):**
1. `1b23ddf` (09-03) — two `PRIOR_CH` runs of the same arm shared a `save_root`; `find_latest_run_dir` (`model_baseline.py:4834`) globs by save_root+stage+demo_num, so the second job's `--reload` loaded the first's checkpoint and died on `size mismatch for input_blocks.0.0.weight: checkpoint torch.Size([64,1,3,3]) vs current torch.Size([128,1,3,3])` (ch128 loading ch64). Fix: `save_root <root>_pch<CH>` plus explicit `stage2.stage1_checkpoint` (`train_confild_stage2_engaging.sh:58-71, 129-134`). The dirs on disk for ld2048 pch64/128/256 (jobs 21866077/78, 21940482) all post-date the fix; the crashed pair's dirs were not kept.
2. `f55854e` (09-04) — the fix above separated the *output* dir but both jobs still wrote the *same* generated config `confild_<ARM>_s2_eng.yaml` and raced: "a ch256 job silently trained a ch128 prior because the other job overwrote the file between sed and python start [...] Detected because both logs reported diffusion_parameters=22,931,201 despite one being launched at ch256." Fix: `CFG=.../confild_${ARM}${PRIOR_CH:+_pch$PRIOR_CH}_s2_eng.yaml` (line 117). Superseded: the run dirs of the colliding pair (jobs 21962017 and its sibling) are **not** in the pull; what survives is the stale job copy `confild_sweep8192_s2_eng.yaml` (num_channels 128, job 21962017) and the post-fix runs 21972566 (pch128) / 21972571 (pch256), whose two/one dead-start dirs are preemption restarts, not collisions.

Two further caveats recorded by the authors: the 8192 stage-2 arms are **step-starved** (55 k and 25 k steps vs 204 k / 105 k at 2048, because the 1-D conv prior scales with latent length — HANDOFF.md:415-421), so their better aggregates (0.738 / 0.755) are not budget-comparable to the 2048 arms; and the Engaging H200 buys more epochs per wall-clock hour than the origin H100, so "comparison to origin C numbers is indicative only" (HANDOFF.md:349-351).

## D.3 FireBench Engaging runs

Three trainings, all on the merged u10+u12 file (`FireBench_u10u12_merged.h5`, N = 3,677,184, 120 frames, 5 fields u,v,w,theta,rho_f) built on Engaging by `src/engaging/merge_firebench_cases.py` from the two `*_dense.h5` sources (`patch_firebench_merge_time.py:33-36`). Launcher env for all three: `JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10 JHU_AUGMENT=reflect_y AUG_GRID_SHAPE=152,126,192` (`train_fb_baseline_engaging.sh:17`, `train_fb_v5clean_engaging.sh:15`); H200, `mit_normal_gpu`, 6 h segments chained with `--reload`/`--RELOAD`.

**Config diffs (semantic).** Each `fb_*_eng.yaml` is a `sed` of its tracked ancestor that replaces only the data path (`train_fb_baseline_engaging.sh:40, 46`; `train_fb_v5clean_engaging.sh:36-37`):

| Job copy | Ancestor | Differing keys |
|---|---|---|
| `fb_det_eng.yaml` (job 21886438) | `config_baseline_Det_firebench.yaml` (Senseiver, demo 36) | `shared.paths.data_path` only (77 keys) |
| `fb_lfm_eng.yaml` (21886435) | `config_baseline_Gen_firebench.yaml` (latent FM, demo 35) | `data_path` only (99 keys) |
| `fb_v5clean_eng.yaml` (22000143) | `config_iclr_firebench_v5clean.yaml` (DMF-Gen, demo 31) | `data` only (79 keys) |

As-run vs tracked: Senseiver/LFM `run_config.yaml`s add `shared.data.field_names: null` (trainer default) and, for LFM, `vis_n_obs_list null → [36772]`; the older dirs carry the earlier jobs' `/tmp` paths (21770410-13 v5clean, 21770414-16 LFM, 21770417/18/20 Senseiver). v5clean `args.json` vs yaml: `vis_cond_fields None→[0,1,2]`, `vis_n_obs_list None→[36772]`, `RELOAD true` — nothing substantive.

**Sensor fractions and split, per model (from the tracked yamls; these are the same three configs §C.3.5 lists):**

| Model (demo) | `cond_fields` | `n_obs` per field | Fraction of 3,677,184 | `train_ratio` | Block split at gap 10 over 120 frames (`helpers.py:261-283`) | Budget |
|---|---|---|---|---|---|---|
| DMF-Gen v5clean (31) | [0,1,2] u,v,w | 1000-36772 | 0.027-1.0 % | **0.9** (`config_iclr_firebench_v5clean.yaml:27`) | n_val = 12, n_train = 98 → train frames 0-97, val 108-119 | 3500 ep × ⌈98/8⌉=13 ≈ 45 k steps; `n_query 36772` (1 %), `measurement_ops null`, topk 16 |
| Senseiver (36) | [0,1,2] | 1000-36772 | same | **0.75** (`config_baseline_Det_firebench.yaml:36`) | n_val = 30, n_train = 80 → train 0-79, val 90-119 | 4000 ep, bs 6 (`:110-111`); `n_query 19531` |
| Latent FM (35) | [0,1,2] | 1000-36772 | same | **0.75** (`config_baseline_Gen_firebench.yaml:33`) | same as Senseiver | s1 5000 ep / s2 10000 ep, bs 8 (`:91-92, 108-109`) |

**Correction to §C.3.5:** that table gives "0.9 block gap 10" for `config_baseline_Det_firebench.yaml` and "same" for `Gen_firebench`; the files say `train_ratio: 0.75`. The consequence is a **protocol mismatch inside FireBench**: DMF-Gen trains on 98 frames (including u12 frames 60-97) and is validated on 12, the two baselines train on 80 and validate on 30 (u12 frames 90-119). Commit `4df3d70`'s "Verified: loader builds train n=80 / val n=30" is the *baseline* split. Any FireBench comparison table must state this, or the ratios must be aligned before the evals run (none has run yet — see below).

**Why several Stage-1 dirs with the same DemoN exist.** Commit `4df3d70` (09-03): "All 13 FireBench jobs (v5clean, LFM stage 1+2, Senseiver) died on startup with KeyError f["time"] at helpers.py:230 [now line 237], which reads `time` unconditionally for every dataset. merge_firebench_cases.py wrote only `coordinates` and `fields`; both sources carry time[60] and field_names[5] and the merge dropped them. Because the chains link with afterany, each failure released the next segment and all 13 burned through in seconds." The 10 empty dirs (`run_config.yaml` + `run_metadata.json` only; `losses.csv` with header only) timestamped 2026-09-03 10:01-11:05 are exactly those chain segments: v5clean `100111/100206/100246/102358` (jobs 21770410-13), LFM `100140/102554/102634` (21770414-16), Senseiver `102715/110508/110554` (21770417/18/20). The fix has two parts: `merge_firebench_cases.py:61-73, 83-85` now reads `time` (per input, sliced by `t_idx`) and `field_names` from the sources and writes both; and `src/engaging/patch_firebench_merge_time.py` appends them **in place** to the already-merged 8.9 GB file ("fields/coordinates are correct and verified, so no re-merge"), taking `time` "verbatim from the sources — nothing synthesised", with the documented consequence that `time` restarts at index 60 (two independent cases; splitting is index-based so unaffected; `patch_firebench_merge_time.py:15-19`). The Delta copy of the merged file (`/work/hdd/bilr/ntricard/datasets/FireBench_u10u12_merged.h5`, dated Aug 16) already contains `coordinates, field_names, fields, time` and was never affected. The surviving trainings are the post-fix relaunches:

| Run dir | Job | State at snapshot |
|---|---|---|
| `pointcloud_ffm/iclr_firebench_v5clean_DemoN31_20260904_111804` | 22000143 chain | **complete**: `Loss_*_bk..bk3` = epochs 1-1181, 1181-2360, 2361-3489, 3486-3500 (4 segments); final train 0.387 / val 0.784 at epoch 3500; `Recon/` in-training K=1 metrics; **no `Evaluation/`** |
| `baseline_senseiver/Baseline_senseiver_Stage1_DemoN36_20260905_011115` | 21886438 (resumed: `started_at 20260905_153802`) | **complete**: `det_baseline_history.csv` to step 4000; `Evaluation/epoch_{0500..4000}/senseiver_metrics.json` are in-training single-snapshot checks at `n_obs [36772]×3`; `cost_train.json` is all zeros (written by the resume process) |
| `baseline_latent_fm/Baseline_latent_fm_Stage1_DemoN35_20260904_190906` | 21886435 | **incomplete**: 1,996 of 5,000 AE epochs; **no stage 2** anywhere in the pull |

So the FireBench part of the campaign has produced **no evaluated number**; only the DMF-Gen and Senseiver trainings finished. The Senseiver in-training metrics are labelled with the combustion names hard-coded in `model_baseline.py:30` / `helpers_baseline.py:33` (`FIELD_NAMES = ("CH4","CO","T","U_1","p")`) — read CH4→u, CO→v, T→w, U_1→theta, p→rho_f — and look pathological (epoch 4000: u 0.158, **v 1.366, w 1.270**, theta 0.052, rho_f 0.516, with `obs_rel_l2_SenConsis` for v/w > 1, i.e. the model cannot fit its own sensors on v,w; `val_loss` rose from 1.52 at step 10 to 1.92 at step 4000). Whether that is the model or the diagnostic is not decidable from the pull.

## D.4 JHU DMF-Gen seed replicates (1379, 2718) and the DemoN33 temporal companion

**Configs.** All three job copies are `sed`s of the frozen N29 file `config_iclr_jhu_xcube_spec02.yaml` (81 keys):

| Job copy | Launcher sed (`train_jhu_xcube_seedrep_engaging.sh:42-46`; `train_jhu_temporal_engaging.sh:46-51`) | Differing keys vs spec02 |
|---|---|---|
| `jhu_seedrep1379_eng.yaml` (job 21770427) | `data`, `seed`, `save_dir` | `data → /tmp/ntricard/jhu4c_21770427/JHU_4cubes_stride100.h5`, `seed 42→1379`, `save_dir → .../iclr_jhu_xcube_spec02_seedrep1379` — **3 keys** |
| `jhu_seedrep2718_eng.yaml` (21770433) | same | `data`, `seed 42→2718`, `save_dir → ..._seedrep2718` — 3 keys |
| `jhu_temporal_eng.yaml` (21770409) | `data`, `save_dir`, `epochs`, `Demo_Num` | `data → /tmp/ntricard/jhu_21770409/JHU_TurbulenceDataset.h5` (**a different file**), `save_dir → .../iclr_jhu_temporal_spec02`, `epochs 6000→2500`, `Demo_Num 29→33` — 4 keys |

**`args.json` vs spec02** (99 keys vs 81): beyond the keys above, the extra 18 are argparse defaults not present in the yaml (`RELOAD true`, `config <job copy path>`, `dataset "jhu"`, `cond_field 2`, `n_obs_min/max 64/256`, `prior_k_min/max/slope`, `gather_multiscale false`, `gather_topk_coarse 64`, `gather_coarse_sigma_scale 4.0`, `fno_modes_z 16`, `fno_domain_padding null`, `Num_z null`, `measurement_ops null`, `spectral_window false`, `processed_root` (wing path)) and `vis_cond_fields None→[0,2]`, `vis_n_obs_list None→[19531]` (trainer fills the visualisation defaults). None of these touches training (the `n_obs_min/max` scalars are superseded by the list keys; `gather_multiscale` is false). So: **seed replicates = N29 with only seed and save_dir changed; DemoN33 = N29 with dataset, epochs and demo number changed.** `Demo_Num` stays 29 for the replicates by intent ("same protocol", `train_jhu_xcube_seedrep_engaging.sh:13-14`), which creates a `DemoN29` name collision with the canonical run (D.6-6). Launcher env: `JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 JHU_AUGMENT=octahedral` for the replicates (`:21`), `JHU_SPLIT_GAP=100` for DemoN33 (`train_jhu_temporal_engaging.sh:20`) — the gap is not a yaml key, so it is invisible in `args.json`.

**What "temporal companion" means.** `train_jhu_temporal_engaging.sh:11-18`: "Temporally-blocked SAME-REGION companion training (paper pending item vi): N29 architecture config (spec02) retrained on the single-cutout 617-frame consecutive JHU dataset with a time-forward block split (last 25% = val, 100-frame guard gap; frame correlation ~0.67 at separation 100, so the residual correlation must be disclosed with the result). Budget-matched to the canonical 48k steps: 363 train frames / batch 20 -> ~19 steps/epoch, epochs 2500." The dataset is `Dataset/JHU_TurbulenceDataset.h5` (HANDOFF.md:161-165: 18.4 GB, "single-cutout, 617 CONSECUTIVE frames of isotropic1024coarse (cube 125^3, start_ijk [228,51,563]) — this is the temporally-blocked same-region protocol dataset, NOT the 4-cube cross-cube file"), so N is still 125³ = 1,953,125 and 19,531 sensors is still 1 %. Split with `helpers.py:266-271`: n_val = round(617 × 0.25) = 154, n_train = 617 − 154 − 100 = 363 (train frames 0-362, gap 363-462 discarded, val 463-616). Same architecture, augmentation, sensors and query budget as N29; it answers "does the headline number depend on the cross-cube split?", not "is this a stronger held-out test?" — the authors are explicit that ~0.67 residual correlation makes it *weaker* (commit `36e33f0`; HANDOFF.md:439-443).

**Training state.** DemoN33: 3 segments (`Loss_*_bk` 1-1020, `_bk1` 1021-2044, `_bk2` 2046-2500), 2,500/2,500 epochs, final val loss 1.371 (HANDOFF quotes the best 1.2579). Seed 1379: 3 segments (1-2149, 2146-4482, 4481-6000), **6,000/6,000**, final val 0.919. Seed 2718: 4 segments (1-1406, 1406-2888, 2886-4897, 4896-6000), **6,000/6,000**, final val 0.977. (Epoch overlaps at segment boundaries are the resume re-running the last unsaved epoch.)

**Recorded results.** Only DemoN33 was evaluated: `src/engaging/eval_jhu_temporal_engaging.sh` → `ensemble_eval.py --run-dir <RD> --ckpt best.pt --data $DEMO/Dataset/JHU_TurbulenceDataset.h5 --K 8 --n-steps {4,2} --n-snapshots 50 --cond-fields 0 2 --n-obs 19531 19531 --seed 0 --op-seed 1000` (lines 44-50), `JHU_SPLIT_MODE=block JHU_SPLIT_GAP=100` (line 29, "the gap defines the val block, so evaluating at another gap scores a different split"), pinned to `node2906` h100 (lines 7-8; the canonical fingerprint at snap 29 still fires because the draw depends on N and seed, not on the field values — `ensemble_eval.py:72-95`). Output `Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_temporal_spec02_DemoN33_20260903_100111/Evaluation/canonical_all50_nfe{4,2}_K8.json` (keys `run_dir, ckpt, backbone GL_rbf_ENH, K, n_steps, cond_fields, n_obs, noise_sigma 0.0, summary, snapshots[50] {per_field, aggregate, rank_hist, snapshot}, figure_seconds`; the 50 snapshot indices are val-block indices 0..153 in seed-0 order, e.g. `8,145,114,89,...`):

| DemoN33 (temporal, same region) | NFE 4 (primary) | NFE 2 |
|---|---|---|
| `summary.rel_l2_mean` (agg) | **0.6027** | 0.5967 |
| per-field rel_l2_mean Ux / Uy* / Uz / p* (means over the 50 snap entries) | 0.199 / 1.097 / 0.187 / 0.928 | 0.192 / 1.089 / 0.184 / 0.921 |
| `rel_l2_single` | 0.692 | 0.643 |
| CRPS | 0.402 | 0.427 |
| spread / error | 0.673 | 0.452 |
| cov50 / cov90 | 0.292 / 0.566 | 0.182 / 0.378 |

Against the paper N29 row (`Paper/pof2026/main.tex:987`: 0.176 / 1.048 / 0.182 / 0.964, agg 0.593, CRPS 0.291, Spr/Err 0.63, Cov90 0.54): aggregate within 0.01, observed channels ~0.02 worse, Uy/p within 0.05, **CRPS 0.40 vs 0.29** is the one column that moved. Commit `36e33f0` reads it as "the headline number does not depend on the cross-cube split", with the NFE-2 calibration collapse (cov90 0.566 → 0.378 at equal accuracy) as the operating-point-sensitivity example; framing "left open for Nick" (`main.tex:142` of the **ICLR** draft, not the PoF paper, which does not mention DemoN33 or the replicates). **The two seed replicates have no `Evaluation/` directory** — trained to completion, never scored; their only numbers are K=1 in-training `Recon/.../Epoch_6000/euler_nfe4_metrics.json` (snapshot 0: 1379 → Ux 0.247 / Uy 1.257 / Uz 0.285 / p 0.856; 2718 → 0.259 / 1.259 / 0.264 / 0.924), which are not reportable.

## D.5 Code changes in this pull

Only two Python files under `src/` that affect evaluation changed, plus two launcher-side files and the merge script (D.3). `git log 41203e0..HEAD -- src/model_baseline.py src/confild_upstream_training.py src/train_Gen_Baseline.py src/train_pointcloud_ffm.py src/helpers.py src/evaluate_confild_stage1.py` is **empty**: no trainer, model, loader or metric code moved.

**`src/confild_eval_unified.py` (+6 −1; commit `ec47358` "Port CoNFiLD canonical DPS eval to Engaging; make CONFILD_ROOT env-overridable").** One hunk, lines 40-45: `CONFILD_ROOT = "/work/hdd/bilr/ntricard/datasets/baselines/CoNFiLD"` → `CONFILD_ROOT = os.environ.get("CONFILD_ROOT", "/work/hdd/bilr/ntricard/datasets/baselines/CoNFiLD")`, still followed by `sys.path.insert(0, CONFILD_ROOT)`. Reason (commit body): the module "hardcoded the origin /projects CoNFiLD checkout and inserted it on sys.path at import time, so it could not even be imported here [Engaging]. Now reads $CONFILD_ROOT, defaulting to the origin path so origin behaviour is unchanged." The merge (`1052090`) kept the Delta default. `DEFAULT_DATA` (`--data` default, line 253) is untouched and still the Delta path; the Engaging launcher passes `--data` explicitly. **Canonical numbers unchanged**: with the env var unset the resolved path is byte-identical to before, and the variable only selects *which checkout* of the upstream code is imported — the checkout itself is pinned at the same commit on both sites (`449835e`, §A.7; HANDOFF.md:341-343). §A.7's line cites shift by +4 in this file (`load_stage1` now 198, `load_stage2` 221, argparse `--dps-scale/--steps` 261-262, `create_sampler` 316, `get_noise` 320, `WindowSensorOperator` 72). **Note** `.gitignore` now ignores `Save_TrainedModel/` and `*.out` (lines 41-43) — future run-dir metadata will need force-adds like `8aa01c9`.

**`src/ensemble_eval.py` (+13 −4; commit `948b085` "Add --data override to ensemble_eval; DemoN33 canonical eval wrapper").** Three hunks: (1) `load_run(...)` signature gains `data_override: str | None = None` (line 101); (2) `data_path = cfg["data"]` → `data_path = data_override or cfg["data"]` (line 111) with the comment "Engaging launchers stage the H5 to node-local /tmp for I/O, so args.json records a path like /tmp/<user>/jhu_<jobid>/... that is gone once the job ends. That path is a transient staging artifact, not provenance"; (3) argparse `--data` (default `None`, lines 317-319) and the call site `load_run(args.run_dir, args.ckpt, args.device, split=args.split, data_override=args.data)` (lines 365-366). The merge resolved a conflict here: Delta had added `split=`, Engaging `data_override=`; both are kept and both are passed. **Canonical numbers unchanged**: with `--data` omitted `data_override` is `None`, `None or cfg["data"]` is `cfg["data"]`, and the rest of `load_run` (abs-path resolution, dataset construction) is untouched; none of the 20+ Delta eval scripts that call `ensemble_eval.py` (`grep -l ensemble_eval.py src/*.sh`: `eval_jhu_insample_dmf.sh`, `calib_nfe_array.sh`, `eval_crps_all.sh`, ...) passes `--data`. The only thing `--data` can change is *which H5 file is opened*; the sensor draw, fingerprint check and metrics do not read the path.

**Launchers (+2 each; commit `90c911a`):** `train_fb_baseline_engaging.sh:20-21`, `train_fb_v5clean_engaging.sh:18-19`, `train_jhu_temporal_engaging.sh:23-24`, `train_jhu_xcube_seedrep_engaging.sh:24-25` each gain `source ~/envs/phycoflow` (the dedicated venv that replaced the broken anaconda+`--user` stack: scipy 1.16 vs numpy 1.24, missing `neuralop`, KeOps unable to link `-lnvrtc` because the cuda module sets only `LD_LIBRARY_PATH` — HANDOFF.md:259-307). `src/engaging/check_env.sh` (new, two-tier: `bash` = login-safe `pip check`; `sbatch` = real imports incl. the CoNFiLD eval entry points + a KeOps `argKmin` vs `torch.cdist` cross-check) and `requirements_engaging.txt` are the guard and the freeze. None of this runs on Delta.

## D.6 Things that look inconsistent or deserve a second look

1. **Two different "CoNFiLD strict-2048" arms exist, and the PoF paper cites the one that is *not* in this pull.** `Paper/pof2026/main.tex:1103-1106`: "A fourth CoNFiLD arm, with strict 2048-point conditioning, is omitted altogether: its files aggregate to 0.941 relative L2 but carry no protocol stamp" (also `DELTA_STATUS_2026-09-08.md:275, 421, 887-888`: "CoNFiLD strict at 0.90 (files give 0.941, unstamped)"). That number matches nothing in `strict_ld2048_hf144/` (canonical last 1.316, best 1.364, `rel_l2_single` 1.401/1.448) and the `DATA_TRANSFER_TIER2.md:67` expectation ("latsweep_2048 / 4096 / strict2048 / improve") points at an origin-side directory set that never arrived. The Engaging strict arm *is* stamped (`crps_canonical_*_summary.json` records seed 0, op_seed 1000, n_sensor_total 39062, K 8, windows [0,18]) but is a separate H200 training with a 6.47 M budget-compliant model. Also, "2048-point conditioning" misdescribes the arm: strict2048 = 2048-scalar latent code at hidden width 144; conditioning is the standard 19,531 + 19,531 sensors. If the footnote is kept, either the description or the number needs fixing, and the two arms must not be conflated.
2. **The "fixed stage 1" prior sweep is not fixed at its ch32 point** (D.2.3): the 1.316 anchor is the hf144 decoder, the ch64-256 points are hf256. The HANDOFF.md:382-398 wording ("Stage 1 held fixed (sweep2048); only prior num_channels varied") and the `bcd98e5` commit body should be read with that substitution; a ch32 stage 2 on `sweep_ld2048_hf256` would close the gap (≈5.5 h H200).
3. **Tracked `confild_sweep2048_s2_eng.yaml` and `confild_sweep8192_s2_eng.yaml` do not describe the runs that cite them.** The former is the pch256 overwrite while `run_metadata.json` of the pch64 and pch128 runs names it as their config; the latter is the second-collision job copy (job 21962017, num_channels 128) with no surviving run. Anyone reconstructing the prior sweep from `Save_config/` alone would get ch256 for all three ld2048 runs. The as-run `run_config.yaml`s are correct (D.2.2). Same class of issue as §C.4-8's launcher snapshots, one layer down.
4. **Canonical `best` pairs a stage-2 prior with a decoder it was not trained on** for the sweep2048 prior variants: stage 2 was pinned to stage-1 `last.pt` (epoch 1422), but `SEL=best` evaluates stage-1 `best.pt` (epoch 1389) with stage-2 `best.pt` (`eval_confild_canonical_engaging.sh:58-64`). The differences are small (0.799 vs 0.806 for ch128) and the HANDOFF quotes `last` throughout, but do not quote `best` for these arms without saying so. For sweep8192 (best 1399 vs last 1455) the same applies.
5. **Dead-start run dirs under `sweep_ld8192_hf256_pch128/` and `_pch256/`** (`20260904_112551`, `20260904_120442`, `20260904_113315`: 400 / 3,200 / 0 steps, no checkpoint, no Evaluation) are preemption restarts that `--reload` could not resume because `save_every 5000` had not fired. They are harmless for the launchers (`tail -1` picks the newest) but any glob that takes the *first* `Stage2_DemoN23_*` dir, or counts dirs, will be wrong. Their `run_config.yaml`s also carry `data_path` of the *final* job id (the requeue kept the id), which looks like a resume that never happened.
6. **Demo-number collisions, JHU/pointcloud_ffm (extends §C.4-9).** `iclr_jhu_xcube_spec02_seedrep{1379,2718}_DemoN29_*` deliberately reuse `DemoN29` (the canonical run's number); `eval_jhu_temporal_engaging.sh:33` and every Delta script that resolves a run by `*_DemoN29_*` glob must use the save_dir prefix, never the demo number. CoNFiLD keeps `demo_num 23` across all ten Engaging arms (and shares it with `config_iclr_jhu_xcube_reg_s42`, `config_baseline_Gen_xcube_aug`, arm C — §C.4-9); the per-arm `save_root` is the only disambiguator, and stage-2 `find_latest_run_dir` relies on it (this is exactly what bit in `1b23ddf`).
7. **FireBench `train_ratio` mismatch and a §C.3.5 error** (D.3): DMF-Gen 0.9 (98/12 frames) vs Senseiver and latent FM 0.75 (80/30). §C.3.5 currently prints 0.9 for the baselines; the yamls say 0.75. Nothing has been evaluated on FireBench yet, so this can still be fixed before any table exists.
8. **Trained but never scored, or unfinished:** seed replicates 1379 and 2718 (6000/6000 epochs, no `Evaluation/`); FireBench DMF-Gen v5clean (3500/3500, no `Evaluation/`); FireBench Senseiver (4000/4000, only in-training checks, and those look broken — v,w sensor-consistency > 1, val loss rising); FireBench latent FM stage 1 at 1996/5000 epochs, no stage 2. The HANDOFF's "seed noise is the open rigor item" (line 89-90) is therefore still open; the seedrep evals need the h100 node and `--data` (D.5).
9. **`HANDOFF.md` is a resurrected ICLR-era document.** Header "updated 2026-08-30", paper = `Paper/iclr2027/main.tex`, "LOCUS → DMF-Gen-3D"; the Delta side had replaced it with the PoF pivot (`b3b4163`) and then removed it (`464c72e`). Keep it for the 08-31..09-05 logs, but it should not be mistaken for the current handoff (`HANDOFF_2026-09-12.md`). Its instruction "do NOT remove the JHU material from the main text yet" (line 50) is about the ICLR draft.
10. **Senseiver FireBench metric labels are combustion names** (`CH4, CO, T, U_1, p` from `model_baseline.py:30`) because `config_baseline_Det_firebench.yaml` has no `field_names` and the baseline trainers never read the H5 `field_names` dataset; the DMF-Gen yaml (`config_iclr_firebench_v5clean.yaml:30`) does name `[u, v, w, theta, rho_f]`. Cosmetic, but any FireBench baseline table generated from these JSONs will be mislabelled.
11. **`config_pointcloud_ffm_ram.yaml` is whitelisted in `.gitignore` (line 38) but still absent from `Save_config/`** — the whitelist predates the pull and the file was never committed; §C.2.4/§C.4-10 stand.
12. **Comment drift in the sweep configs:** every `config_baseline_CoNFiLD_xcube_sweep*.yaml` header says stage-2 is refused by the budget gate "by design", but the stage-2 launcher's `BUDGET_ENFORCE=0` path and all five prior-variant `run_config.yaml`s have `enforce: false`. The `check_confild_sweep_params.py` docstring likewise describes a four-arm sweep whose expected prior is "identical across arms" (1,441,217) — true for stage 1, no longer true for the stage-2 study. Neither affects a number; both would mislead a reader auditing from the yaml headers (same pattern as §C.4-11).
13. **Wall-clock accounting is per-launcher, not per-trainer**, and the trainer's own `wallclock_budget_s` is still process-relative (`confild_upstream_training.py:485, 655` / `760, 896`). The resume-aware subtraction lives only in the three Engaging shell scripts; a Delta relaunch of any of these arms through `train_confild_unified.slurm` would grant a fresh full budget on resume. Also, dead-start segments (D.6-5) are never charged, so `sweep_ld8192_hf256_pch128` consumed 19,777 + 1,214 s of GPU time against a 19,800 s protocol — immaterial to the result (no checkpoint carried over), but the "budget-matched" label is launcher-enforced, not file-enforced.
