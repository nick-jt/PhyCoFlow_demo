# Per-method reconstructed-field dumps (fixed qualitative snapshot)

Produced by `src/dump_fields_baseline.py` via `src/dump_fields_pof.sh`
(this worktree). Purpose: cross-method qualitative figure panels on the SAME
snapshot and the SAME sensor set as the existing qualitative dumps
`$MAIN/Paper/iclr2027/figures/qual_jhu.npz` (DMF-Gen/ours) and
`qual_firebench.npz`.

## Files

| file | dataset | method | checkpoint (under $MAIN/Save_TrainedModel) | K | NFE |
|---|---|---|---|---|---|
| `jhu_sit.npz` | JHU 125^3 | SiT-point (generative) | `JHU/baseline_sit/matched/Baseline_sit_Stage1_DemoN41_20260828_181938/best.pt` | 8 | 32 (run sampling_N) |
| `jhu_senseiver.npz` | JHU 125^3 | Senseiver (deterministic) | `JHU/baseline_senseiver/Baseline_senseiver_Stage1_DemoN43_20260828_180959/best.pt` | 1 | - |
| `firebench_latent_fm.npz` | FireBench 152x126x192 | latent-FM (generative) | `firebench/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN35_20260823_234826/best.pt` | 8 | 4 |
| `firebench_senseiver.npz` | FireBench | Senseiver (deterministic) | `firebench/baseline_senseiver/Baseline_senseiver_Stage1_DemoN36_20260823_153857/best.pt` | 1 | - |

Existing companion dumps (ours + JHU latent-FM) already live in
`$MAIN/Paper/iclr2027/figures/`: `qual_jhu.npz`, `qual_firebench.npz`
(truth/mean/std/sample0 in STANDARDIZED units), `ens_latentfm_s{0,12}.npz`.

## Snapshot / sensor protocol (matches the qual dumps bit-for-bit)

* the SAME PHYSICAL SNAPSHOT as the qual dumps, pinned by ABSOLUTE frame
  index (`--frame`), split env exactly as the qual launchers:
  * JHU (`JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0`, all runs ratio 0.75):
    qual val[3] = frame **153** = val[3] for the baselines too.
  * FireBench (`JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10`, merged u10+u12 h5,
    120 frames): ours trained ratio 0.9 -> qual val[3] = frame **112**;
    the FB baselines trained ratio 0.75, where frame 112 is val index 22.
    NOTE this ratio mismatch means earlier launchers that passed ours' val
    indices straight to `evaluate_*_Baseline --snapshot-index` on FireBench
    (e.g. eval_firebench_ops.sh) were comparing different physical frames.
  * DemoN43 (JHU Senseiver) has a dead node-local `/tmp` data_path in its
    run_config; the dump overrides it to the shared
    `outputfiles_diverse/JHU_4cubes_stride100.h5` (same file all JHU runs
    staged from).
* sensors drawn with the CANONICAL `helpers.build_sparse_condition`
  (not the `helpers_baseline` variant — the two are not RNG-equivalent) under
  `torch.manual_seed(seed)` on an H100-SXM compute node:
  * JHU: seed `100+3=103`, `cond_fields=[0,2]` (Ux, Uz), `n_obs=[19531]`
    (1% per observed channel) — identical to `qualitative_jhu.py`.
  * FireBench: seed `1000+3=1003`, `cond_fields=[0,1,2]` (u,v,w),
    `n_obs=[36772]` — identical to `qualitative_firebench.py`.
* each npz records two verification results in `meta` (also printed in the
  job log): `truth_corr` (~1.0 iff same physical snapshot) and
  `dist_max_absdiff` (0 iff the identical sensor set), both checked against
  the corresponding qual npz.
* generative samplers are each baseline's OWN sampler (SiT:
  `sit_conditional_sample_points_chunked`; latent-FM: `model.sample` on the
  obs grid), K=8, member k seeded `torch.manual_seed(sensor_seed*131*10000+k)`.

## Keys (all float32; fields in PHYSICAL units)

* `truth` `[N, C]` — ground truth.
* `pred_mean` `[N, C]` — ensemble mean; for Senseiver, the single
  deterministic prediction.
* `pred_sample` `[N, C]` — first ensemble member; for Senseiver, same as
  `pred_mean`.
* `pred_std` `[N, C]` — ensemble std (physical units); zeros for Senseiver.
* `coords` `[N, 3]` — normalized coordinates; `coords_raw` when the dataset
  provides them. Grids: JHU reshape `(125,125,125)`; FireBench reshape
  `(152,126,192)` = (NX, NY, NZ), same convention as the qual scripts
  (`a[:, c].reshape(...)`).
* `sensor_indices`, `sensor_field_ids` — the drawn sensor set (for overlays).
* `names` `[C]` — field names (JHU: Ux,Uy,Uz,p; FireBench: u,v,w,theta,rho_f).
* `norm_mean`, `norm_std` `[C]` — the run's standardization;
  standardized = (physical - norm_mean) / norm_std. NOTE: the qual npz files
  are in standardized units of OUR run's stats — compare in physical units,
  or restandardize with each side's own stats.
* `meta` — JSON string: checkpoint, epoch, seeds, NFE, K, sensor fingerprint
  (count + idx_sum), GPU, env, rel-L2, and the qual-match check results.
