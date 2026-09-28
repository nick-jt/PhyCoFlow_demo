# `src/` layout (reorganised 2026-09-27)

This directory was flattened for a year and had grown to ~640 files. It was
reorganised on 2026-09-27 **purely for readability**: every file was moved with
`git mv`, no numerical code was changed, and every embedded path was repointed
so the tree behaves exactly as before. The old -> new map for every file is at
the bottom of this page.

## What lives where

```
src/
  <main entry points>          DMF-Gen train/eval + every baseline's canonical train/eval
                               (train_pointcloud_ffm, ensemble_eval, evaluate_ffm,
                               train_/evaluate_{Gen,Det}_Baseline, eval_kolm_ensemble,
                               eval_*_ensemble, eval_*_iclr, train_deeponet*, train_s3gm3d,
                               eval_s3gm3d, confild_eval_unified, baseline_classical_*, ...)
  _srcpaths.py                 sys.path bootstrap that makes the sub-directories importable
  README.md                    this file

  core/                        DMF-Gen library: Model, helpers, losses, priors, operators,
                               backbones (FNO-3D, CQ adapter), augmentations, wing dataset,
                               fine-tuning helpers, artifact_guard / fleet_select
  baselines/                   shared baseline library (model_baseline, helpers_baseline,
                               loss_plot, lfm_fixes, dataset_wing_baseline, sensor_fingerprints)
  baselines/confild/           CoNFiLD: upstream adapters, legacy 3-stage pipeline, stage-A/B
                               fix campaign, selection / summary tools
  baselines/s3gm/              S3GM 3-D adaptation, normalised guidance, tuning, watchdog
  baselines/deeponet/          DeepONet / DeepONet++ model definitions
  baselines/senseiver/         Senseiver improvement-arm patches, sweep wrappers, diagnostics
  data_prep/                   dataset conversion / extraction / inspection (OpenFOAM cylinder,
                               Kolmogorov, FireBench, exports, H5 viewers, frame audits)
  calibration/                 uncertainty calibration campaign (probes, checkpoint-window
                               variance, conformal / spread recalibration, calib dumps)
  spectra/                     spectral diagnostics and the paper spectra figure
  figures/                     qualitative figures, re-plots, field dumps for galleries,
                               JHU panels, the z-collapse figure guard
  analysis/                    CPU post-hoc aggregation: tables, paired CIs, seed variance,
                               cross-Re breakdown, budget audits
  experiments/                 one-off experiment drivers (lit-protocol ablation, LDW-FFM
                               blend, K ablation, interpolation floors, FireBench field probe)
  benchmarks/                  cost / memory / profiling instrumentation
  ops/                         job-side utilities (checkpoint archiving)
  temp/                        temporary checks: test_*, check_*, probes, seed checks and the
                               one-off shell probes that drove them
  submissions/                 every SLURM / launcher script, by purpose (see below)
  logs/                        the old *.log / *.out job logs that used to sit here, plus
                               fleet_jobs_delta.txt (the 2D fleet job map)

  engaging/ figs_pof/ phycoflow_pointcloud/ sit_transport/   unchanged
```

`submissions/`:

```
  dmfgen/train                 DMF-Gen training launchers (JHU, FireBench, wing, FNO-3D)
  dmfgen/eval                  DMF-Gen evaluation launchers (matched evals, sweeps, wing plots)
  eval_all_methods             evaluation matrices that score DMF-Gen AND the baselines
  baselines/                   unified baseline trainers (train_baseline_*, train_{kolm,cyl}_baseline)
  baselines/{confild,s3gm,senseiver,deeponet,sit,gen4turb,classical}   per-family launchers
  fleet_2d                     the 2D Kolmogorov / cylinder fleet: submit chains, fleet evals,
                               DMF-Gen 2D trainers, post-transfer resume
  figures  spectra  calibration  benchmarks  experiments  data  ops     as for the .py dirs
  smoke                        smoke gates (short end-to-end runs that guard a launch)
```

## Conventions (unchanged, now written down)

* **Run and submit from `src/`.** Nearly every launcher does `cd $SLURM_SUBMIT_DIR`
  and then calls scripts by a path relative to `src/`, e.g.
  `sbatch submissions/dmfgen/train/train_iclr_jhu_xcube.sh` or
  `bash submissions/fleet_2d/submit_2d_fleet_delta.sh`. Submitting from
  another directory was never supported and still is not.
* **Module names did not change.** `import helpers`, `import model_baseline`,
  `from Model import ...` all still work, so checkpoints that pickle class
  references keep loading. What changed is only *where* the file sits.
  `src/_srcpaths.py` puts every library directory on `sys.path`; each entry
  point (root or sub-directory) starts with a short bootstrap block that
  imports it. From an interactive session in `src/`: `import _srcpaths` first.
* **A new script** goes in the directory matching its purpose, with the
  bootstrap block copied from any neighbour (it resolves `src/` from
  `__file__`, so the `".."` count must match the depth). A new submission
  script goes under `submissions/<purpose>/` and refers to `.py` files by
  their `src/`-relative path.
* **Log destinations are unchanged.** `#SBATCH --output=...log` lines still
  write next to wherever the job was submitted from (i.e. `src/`). Only the
  pre-existing logs were moved into `logs/`. If you want new logs out of the
  root too, change those lines to `logs/<name>_%j.log`.
* The S3GM tuning scripts (`baselines/s3gm/tune_norm_guidance*.py`,
  `merge_winner.py`, `run_final_eval.py`) read/write `results/` and
  `results2/` next to themselves; those directories now live under
  `baselines/s3gm/` (they were never present in this checkout).

## Old -> new map

| file (was at `src/`) | now |
|---|---|
| `add_grid_surface_indices.py` | `data_prep/add_grid_surface_indices.py` |
| `archive_checkpoints.py` | `ops/archive_checkpoints.py` |
| `archive_checkpoints.sh` | `submissions/ops/archive_checkpoints.sh` |
| `artifact_guard.py` | `core/artifact_guard.py` |
| `assemble_baseline_table.py` | `analysis/assemble_baseline_table.py` |
| `audit_fb_frames.py` | `data_prep/audit_fb_frames.py` |
| `audit_train_budgets.py` | `analysis/audit_train_budgets.py` |
| `augment_octahedral.py` | `core/augment_octahedral.py` |
| `augment_symmetry.py` | `core/augment_symmetry.py` |
| `baseline_classical_figs.py` | `figures/baseline_classical_figs.py` |
| `bench_backbones.py` | `benchmarks/bench_backbones.py` |
| `bench_baseline_sample.sh` | `submissions/eval_all_methods/bench_baseline_sample.sh` |
| `bench_cost.sh` | `submissions/benchmarks/bench_cost.sh` |
| `bench_fno3d.py` | `benchmarks/bench_fno3d.py` |
| `bench_loader.sh` | `submissions/benchmarks/bench_loader.sh` |
| `bench_loader_abc.py` | `benchmarks/bench_loader_abc.py` |
| `bench_prior.sh` | `temp/bench_prior.sh` |
| `benchmark_cost.py` | `benchmarks/benchmark_cost.py` |
| `benchmark_scaling_baselines.py` | `benchmarks/benchmark_scaling_baselines.py` |
| `calib_archive_window.py` | `calibration/calib_archive_window.py` |
| `calib_ckptvar_analyze.py` | `calibration/calib_ckptvar_analyze.py` |
| `calib_ckptvar_eval.sh` | `submissions/calibration/calib_ckptvar_eval.sh` |
| `calib_ckptvar_smoke.sh` | `submissions/smoke/calib_ckptvar_smoke.sh` |
| `calib_ckptvar_train.sh` | `submissions/calibration/calib_ckptvar_train.sh` |
| `calib_k32.sh` | `submissions/calibration/calib_k32.sh` |
| `calib_nfe16_hi.sh` | `submissions/calibration/calib_nfe16_hi.sh` |
| `calib_nfe_array.sh` | `submissions/calibration/calib_nfe_array.sh` |
| `calib_probe.py` | `calibration/calib_probe.py` |
| `calib_probe.sh` | `submissions/calibration/calib_probe.sh` |
| `calib_probe2.py` | `calibration/calib_probe2.py` |
| `calib_probe2.sh` | `submissions/calibration/calib_probe2.sh` |
| `calib_step_bench.py` | `benchmarks/calib_step_bench.py` |
| `calib_step_bench.sh` | `submissions/benchmarks/calib_step_bench.sh` |
| `calib_table.py` | `calibration/calib_table.py` |
| `calib_truth_spec.py` | `calibration/calib_truth_spec.py` |
| `check_confild_prior_sizes.py` | `temp/check_confild_prior_sizes.py` |
| `check_confild_sweep_params.py` | `temp/check_confild_sweep_params.py` |
| `check_no_zcollapse.py` | `figures/check_no_zcollapse.py` |
| `check_strouhal.py` | `data_prep/check_strouhal.py` |
| `check_sweep_params_and_idw.py` | `temp/check_sweep_params_and_idw.py` |
| `check_xattn.py` | `temp/check_xattn.py` |
| `coherence_dist.py` | `core/coherence_dist.py` |
| `compare_ckpts.py` | `temp/compare_ckpts.py` |
| `compare_ckpts.sh` | `temp/compare_ckpts.sh` |
| `compare_spectra.py` | `spectra/compare_spectra.py` |
| `compare_spectra_n22.py` | `spectra/compare_spectra_n22.py` |
| `compare_spectra_n29.py` | `spectra/compare_spectra_n29.py` |
| `confild_accum_sweep.sh` | `submissions/baselines/confild/confild_accum_sweep.sh` |
| `confild_baseline.py` | `baselines/confild/confild_baseline.py` |
| `confild_cap1024.sh` | `submissions/baselines/confild/confild_cap1024.sh` |
| `confild_cond_smoke.sh` | `submissions/smoke/confild_cond_smoke.sh` |
| `confild_cond_smokeb.sh` | `submissions/smoke/confild_cond_smokeb.sh` |
| `confild_conditional.py` | `baselines/confild/confild_conditional.py` |
| `confild_eval_canonical.slurm` | `submissions/baselines/confild/confild_eval_canonical.slurm` |
| `confild_eval_full.sh` | `submissions/baselines/confild/confild_eval_full.sh` |
| `confild_eval_unified.py.orig_20260829` | `temp/confild_eval_unified.py.orig_20260829` |
| `confild_eval_unified2.py` | `baselines/confild/confild_eval_unified2.py` |
| `confild_figsmoke.sh` | `submissions/smoke/confild_figsmoke.sh` |
| `confild_final.sh` | `submissions/baselines/confild/confild_final.sh` |
| `confild_pick_best.py` | `baselines/confild/confild_pick_best.py` |
| `confild_seedcheck.sh` | `temp/confild_seedcheck.sh` |
| `confild_select_and_final.py` | `baselines/confild/confild_select_and_final.py` |
| `confild_smoke_21693407.out` | `logs/confild_smoke_21693407.out` |
| `confild_smoke_eval.sh` | `submissions/smoke/confild_smoke_eval.sh` |
| `confild_split_summary.py` | `baselines/confild/confild_split_summary.py` |
| `confild_stage1.sh` | `submissions/baselines/confild/confild_stage1.sh` |
| `confild_stage1_plateau.py` | `baselines/confild/confild_stage1_plateau.py` |
| `confild_stage1_smoke.sh` | `submissions/smoke/confild_stage1_smoke.sh` |
| `confild_stage1b.sh` | `submissions/baselines/confild/confild_stage1b.sh` |
| `confild_stage1c.sh` | `submissions/baselines/confild/confild_stage1c.sh` |
| `confild_stage1d.sh` | `submissions/baselines/confild/confild_stage1d.sh` |
| `confild_stage2.py` | `baselines/confild/confild_stage2.py` |
| `confild_stage2.sh` | `submissions/baselines/confild/confild_stage2.sh` |
| `confild_stage2_ckpts.py` | `baselines/confild/confild_stage2_ckpts.py` |
| `confild_stage2b.sh` | `submissions/baselines/confild/confild_stage2b.sh` |
| `confild_stageA_fit.py` | `baselines/confild/confild_stageA_fit.py` |
| `confild_upstream_core.py` | `baselines/confild/confild_upstream_core.py` |
| `confild_upstream_training.py` | `baselines/confild/confild_upstream_training.py` |
| `conformal_recalib.py` | `calibration/conformal_recalib.py` |
| `convert_cylinder.py` | `data_prep/convert_cylinder.py` |
| `convert_cylinder.sh` | `submissions/data/convert_cylinder.sh` |
| `convert_kolmogorov.py` | `data_prep/convert_kolmogorov.py` |
| `convert_kolmogorov.sh` | `submissions/data/convert_kolmogorov.sh` |
| `cq_flash_patch.py` | `core/cq_flash_patch.py` |
| `cross_re_cylinder.py` | `analysis/cross_re_cylinder.py` |
| `dataset_shiftwing.py` | `core/dataset_shiftwing.py` |
| `dataset_wing_baseline.py` | `baselines/dataset_wing_baseline.py` |
| `deeponet_baseline.py` | `baselines/deeponet/deeponet_baseline.py` |
| `deeponetpp.py` | `baselines/deeponet/deeponetpp.py` |
| `diag_senseiver_gradnorm.py` | `baselines/senseiver/diag_senseiver_gradnorm.py` |
| `diag_senseiver_gradnorm.sh` | `submissions/baselines/senseiver/diag_senseiver_gradnorm.sh` |
| `diagnose_sample_spectra.py` | `spectra/diagnose_sample_spectra.py` |
| `diagnose_spectra.py` | `spectra/diagnose_spectra.py` |
| `dump_calib_latentfm.sh` | `submissions/calibration/dump_calib_latentfm.sh` |
| `dump_calib_points.py` | `calibration/dump_calib_points.py` |
| `dump_calib_points.sh` | `submissions/calibration/dump_calib_points.sh` |
| `dump_classical_gallery.py` | `figures/dump_classical_gallery.py` |
| `dump_classical_gallery.sh` | `submissions/figures/dump_classical_gallery.sh` |
| `dump_cyl_inject.sh` | `submissions/figures/dump_cyl_inject.sh` |
| `dump_fields_baseline.py` | `figures/dump_fields_baseline.py` |
| `dump_fields_pof.sh` | `submissions/figures/dump_fields_pof.sh` |
| `dump_gallery_fill.sh` | `submissions/figures/dump_gallery_fill.sh` |
| `dump_jhu_gallery.py` | `figures/dump_jhu_gallery.py` |
| `dump_jhu_gallery.sh` | `submissions/figures/dump_jhu_gallery.sh` |
| `dump_kolm_extra.sh` | `submissions/figures/dump_kolm_extra.sh` |
| `dump_kolm_gallery.sh` | `submissions/figures/dump_kolm_gallery.sh` |
| `dump_kolm_geofno_matched.sh` | `submissions/figures/dump_kolm_geofno_matched.sh` |
| `dump_kolm_nfe.sh` | `submissions/figures/dump_kolm_nfe.sh` |
| `dump_kolm_nfe_delta.sh` | `submissions/figures/dump_kolm_nfe_delta.sh` |
| `dump_surface_gallery.sh` | `submissions/figures/dump_surface_gallery.sh` |
| `envtest_delta.sh` | `temp/envtest_delta.sh` |
| `eval_crps_all.sh` | `submissions/eval_all_methods/eval_crps_all.sh` |
| `eval_deeponet.sh` | `submissions/baselines/deeponet/eval_deeponet.sh` |
| `eval_deeponetpp.sh` | `submissions/baselines/deeponet/eval_deeponetpp.sh` |
| `eval_ffm.sh` | `submissions/dmfgen/eval/eval_ffm.sh` |
| `eval_firebench_ops.sh` | `submissions/eval_all_methods/eval_firebench_ops.sh` |
| `eval_firebench_ops2.sh` | `submissions/eval_all_methods/eval_firebench_ops2.sh` |
| `eval_firebench_ops_matched.sh` | `submissions/eval_all_methods/eval_firebench_ops_matched.sh` |
| `eval_fno3d.sh` | `submissions/dmfgen/eval/eval_fno3d.sh` |
| `eval_fno3d_smoke.sh` | `submissions/smoke/eval_fno3d_smoke.sh` |
| `eval_jhu_insample.sh` | `submissions/eval_all_methods/eval_jhu_insample.sh` |
| `eval_jhu_insample_dmf.sh` | `submissions/dmfgen/eval/eval_jhu_insample_dmf.sh` |
| `eval_k32.sh` | `submissions/dmfgen/eval/eval_k32.sh` |
| `eval_kolm_fleet.sh` | `submissions/fleet_2d/eval_kolm_fleet.sh` |
| `eval_kolm_ksweep.sh` | `submissions/fleet_2d/eval_kolm_ksweep.sh` |
| `eval_kolm_litprotocol.py` | `experiments/eval_kolm_litprotocol.py` |
| `eval_kolm_litprotocol.sh` | `submissions/experiments/eval_kolm_litprotocol.sh` |
| `eval_kolm_sensorsweep_fleet.sh` | `submissions/fleet_2d/eval_kolm_sensorsweep_fleet.sh` |
| `eval_kolm_sweep_rest.sh` | `submissions/fleet_2d/eval_kolm_sweep_rest.sh` |
| `eval_latentfm_canonical.sh` | `submissions/baselines/eval_latentfm_canonical.sh` |
| `eval_n22_matched.sh` | `submissions/dmfgen/eval/eval_n22_matched.sh` |
| `eval_n29_matched.sh` | `submissions/dmfgen/eval/eval_n29_matched.sh` |
| `eval_n31_ops.sh` | `submissions/dmfgen/eval/eval_n31_ops.sh` |
| `eval_n33_matched.sh` | `submissions/dmfgen/eval/eval_n33_matched.sh` |
| `eval_n34n35.sh` | `submissions/dmfgen/eval/eval_n34n35.sh` |
| `eval_n36_scale.sh` | `submissions/dmfgen/eval/eval_n36_scale.sh` |
| `eval_nfe_sweep.sh` | `submissions/dmfgen/eval/eval_nfe_sweep.sh` |
| `eval_noclamp.sh` | `submissions/dmfgen/eval/eval_noclamp.sh` |
| `eval_senseiver_iclr.sh` | `submissions/baselines/senseiver/eval_senseiver_iclr.sh` |
| `eval_senseiver_smoke.sh` | `submissions/smoke/eval_senseiver_smoke.sh` |
| `eval_senseiver_sweep.sh` | `submissions/baselines/senseiver/eval_senseiver_sweep.sh` |
| `eval_senseiver_sweep_wrap.py` | `baselines/senseiver/eval_senseiver_sweep_wrap.py` |
| `eval_sensor_sweep.sh` | `submissions/dmfgen/eval/eval_sensor_sweep.sh` |
| `eval_sensor_sweep_hi.sh` | `submissions/dmfgen/eval/eval_sensor_sweep_hi.sh` |
| `eval_sit_matched_array.sh` | `submissions/baselines/sit/eval_sit_matched_array.sh` |
| `eval_sit_matched_seeded.sh` | `submissions/baselines/sit/eval_sit_matched_seeded.sh` |
| `eval_sit_seeded_array.sh` | `submissions/baselines/sit/eval_sit_seeded_array.sh` |
| `eval_sit_xcube.sh` | `submissions/baselines/sit/eval_sit_xcube.sh` |
| `eval_sit_xcube_array.sh` | `submissions/baselines/sit/eval_sit_xcube_array.sh` |
| `eval_stats_expand_base.sh` | `submissions/eval_all_methods/eval_stats_expand_base.sh` |
| `eval_stats_expand_ours.sh` | `submissions/dmfgen/eval/eval_stats_expand_ours.sh` |
| `eval_wing_plots.sh` | `submissions/dmfgen/eval/eval_wing_plots.sh` |
| `eval_xcube.sh` | `submissions/eval_all_methods/eval_xcube.sh` |
| `eval_xcube_matched.sh` | `submissions/eval_all_methods/eval_xcube_matched.sh` |
| `evaltest_delta.sh` | `temp/evaltest_delta.sh` |
| `evaluate_confild_conditional.py` | `baselines/confild/evaluate_confild_conditional.py` |
| `evaluate_confild_stage1.py` | `baselines/confild/evaluate_confild_stage1.py` |
| `export_for_baselines.py` | `data_prep/export_for_baselines.py` |
| `extract_firebench.py` | `data_prep/extract_firebench.py` |
| `final_eval_norm.sh` | `submissions/baselines/s3gm/final_eval_norm.sh` |
| `fleet_jobs_delta.txt` | `logs/fleet_jobs_delta.txt` |
| `fleet_select.py` | `core/fleet_select.py` |
| `floor_variants.py` | `experiments/floor_variants.py` |
| `floor_variants.sh` | `submissions/experiments/floor_variants.sh` |
| `fno3d_backbone.py` | `core/fno3d_backbone.py` |
| `gen_firebench_field.py` | `experiments/gen_firebench_field.py` |
| `gen_firebench_field.sh` | `submissions/experiments/gen_firebench_field.sh` |
| `geofno_matched_sweeps.sh` | `submissions/fleet_2d/geofno_matched_sweeps.sh` |
| `helpers.py` | `core/helpers.py` |
| `helpers_baseline.py` | `baselines/helpers_baseline.py` |
| `instr_sit.sh` | `submissions/benchmarks/instr_sit.sh` |
| `instrument_sit.py` | `benchmarks/instrument_sit.py` |
| `k_ablation.py` | `experiments/k_ablation.py` |
| `k_ablation.sh` | `submissions/experiments/k_ablation.sh` |
| `keops_gputest.sh` | `temp/keops_gputest.sh` |
| `ldw_ffm_stage1.py` | `experiments/ldw_ffm_stage1.py` |
| `ldw_ffm_stage1.sh` | `submissions/experiments/ldw_ffm_stage1.sh` |
| `ldw_ffm_stage1_analyze.py` | `analysis/ldw_ffm_stage1_analyze.py` |
| `lfm_fixes.py` | `baselines/lfm_fixes.py` |
| `Load_Check.py` | `data_prep/Load_Check.py` |
| `lora_finetune.py` | `core/lora_finetune.py` |
| `loss_plot.py` | `baselines/loss_plot.py` |
| `measurement_ops.py` | `core/measurement_ops.py` |
| `merge_winner.py` | `baselines/s3gm/merge_winner.py` |
| `Model.py` | `core/Model.py` |
| `model_baseline.py` | `baselines/model_baseline.py` |
| `model_cq.py` | `core/model_cq.py` |
| `model_ema.py` | `core/model_ema.py` |
| `model_finetune.py` | `core/model_finetune.py` |
| `obs_consistency.py` | `core/obs_consistency.py` |
| `paired_ci_2d.py` | `analysis/paired_ci_2d.py` |
| `paired_ci_jhu.py` | `analysis/paired_ci_jhu.py` |
| `paper_jhu_panels.py` | `figures/paper_jhu_panels.py` |
| `persistent_topk_geometry_cache.py` | `core/persistent_topk_geometry_cache.py` |
| `preprocess_wing_v3.sh` | `submissions/data/preprocess_wing_v3.sh` |
| `probe_s3gm.sh` | `temp/probe_s3gm.sh` |
| `probe_s3gm2.sh` | `temp/probe_s3gm2.sh` |
| `profile_io.sh` | `submissions/benchmarks/profile_io.sh` |
| `profile_io_sit.py` | `benchmarks/profile_io_sit.py` |
| `profile_solver.sh` | `submissions/spectra/profile_solver.sh` |
| `profile_solver_sections.py` | `benchmarks/profile_solver_sections.py` |
| `profile_step.py` | `benchmarks/profile_step.py` |
| `profile_step.sh` | `submissions/benchmarks/profile_step.sh` |
| `qualitative_firebench.py` | `figures/qualitative_firebench.py` |
| `qualitative_jhu.py` | `figures/qualitative_jhu.py` |
| `qualitative_wing.py` | `figures/qualitative_wing.py` |
| `recalibrate_spread.py` | `calibration/recalibrate_spread.py` |
| `regen_figs.py` | `baselines/s3gm/regen_figs.py` |
| `regen_figs.sh` | `submissions/baselines/s3gm/regen_figs.sh` |
| `replot_firebench.py` | `figures/replot_firebench.py` |
| `replot_jhu.py` | `figures/replot_jhu.py` |
| `replot_spectra.py` | `spectra/replot_spectra.py` |
| `replot_wing.py` | `figures/replot_wing.py` |
| `rerun_dmfgen_sweep.sh` | `submissions/fleet_2d/rerun_dmfgen_sweep.sh` |
| `rescore_cyl_mesh_gh200.sh` | `submissions/fleet_2d/rescore_cyl_mesh_gh200.sh` |
| `resume_after_transfer.sh` | `submissions/fleet_2d/resume_after_transfer.sh` |
| `rng_parity_probe.py` | `temp/rng_parity_probe.py` |
| `rng_probe.sh` | `temp/rng_probe.sh` |
| `run_bench_backbones.sh` | `submissions/benchmarks/run_bench_backbones.sh` |
| `run_cache_upgrade_test.sh` | `temp/run_cache_upgrade_test.sh` |
| `run_classical.sh` | `submissions/baselines/classical/run_classical.sh` |
| `run_classical_cyl.sh` | `submissions/baselines/classical/run_classical_cyl.sh` |
| `run_classical_cyl_surface.sh` | `submissions/baselines/classical/run_classical_cyl_surface.sh` |
| `run_classical_cyl_uonly.sh` | `submissions/baselines/classical/run_classical_cyl_uonly.sh` |
| `run_classical_figs.sh` | `submissions/figures/run_classical_figs.sh` |
| `run_classical_kolm.sh` | `submissions/baselines/classical/run_classical_kolm.sh` |
| `run_classical_kolm_sweep.sh` | `submissions/baselines/classical/run_classical_kolm_sweep.sh` |
| `run_conformal_all.sh` | `submissions/calibration/run_conformal_all.sh` |
| `run_cq_flash.sh` | `temp/run_cq_flash.sh` |
| `run_diag_spectra.sh` | `submissions/spectra/run_diag_spectra.sh` |
| `run_final_eval.py` | `baselines/s3gm/run_final_eval.py` |
| `run_gappy_pod_wing.sh` | `submissions/baselines/classical/run_gappy_pod_wing.sh` |
| `run_gen4turb_all50.sh` | `submissions/baselines/gen4turb/run_gen4turb_all50.sh` |
| `run_gen4turb_eval.sh` | `submissions/baselines/gen4turb/run_gen4turb_eval.sh` |
| `run_gen4turb_uxuz_eval.sh` | `submissions/baselines/gen4turb/run_gen4turb_uxuz_eval.sh` |
| `run_paper_jhu_panels.sh` | `submissions/figures/run_paper_jhu_panels.sh` |
| `run_qualitative.sh` | `submissions/figures/run_qualitative.sh` |
| `run_qualitative_fb.sh` | `submissions/figures/run_qualitative_fb.sh` |
| `run_qualitative_wing.sh` | `submissions/figures/run_qualitative_wing.sh` |
| `run_scaling_baselines.sh` | `submissions/benchmarks/run_scaling_baselines.sh` |
| `run_scaling_bench.sh` | `submissions/benchmarks/run_scaling_bench.sh` |
| `run_scaling_fixup.sh` | `submissions/benchmarks/run_scaling_fixup.sh` |
| `run_spectra_arms.sh` | `submissions/spectra/run_spectra_arms.sh` |
| `run_spectra_n29.sh` | `submissions/spectra/run_spectra_n29.sh` |
| `run_spectra_paper.sh` | `submissions/spectra/run_spectra_paper.sh` |
| `s3gm3d.py` | `baselines/s3gm/s3gm3d.py` |
| `s3gm_eval.sh` | `submissions/baselines/s3gm/s3gm_eval.sh` |
| `s3gm_evalsmoke.sh` | `submissions/smoke/s3gm_evalsmoke.sh` |
| `s3gm_finalize.sh` | `submissions/baselines/s3gm/s3gm_finalize.sh` |
| `s3gm_improve_common.py` | `baselines/s3gm/s3gm_improve_common.py` |
| `s3gm_isolate.py` | `baselines/s3gm/s3gm_isolate.py` |
| `s3gm_isolate.sh` | `submissions/baselines/s3gm/s3gm_isolate.sh` |
| `s3gm_norm_guidance.py` | `baselines/s3gm/s3gm_norm_guidance.py` |
| `s3gm_smoke.sh` | `submissions/smoke/s3gm_smoke.sh` |
| `s3gm_train.sh` | `submissions/baselines/s3gm/s3gm_train.sh` |
| `s3gm_watchdog.py` | `baselines/s3gm/s3gm_watchdog.py` |
| `s3gm_watchdog.sh` | `submissions/baselines/s3gm/s3gm_watchdog.sh` |
| `sA1_fit.sh` | `submissions/baselines/confild/sA1_fit.sh` |
| `sA_sweep.sh` | `submissions/baselines/confild/sA_sweep.sh` |
| `sB1a_s2ckpts.sh` | `submissions/baselines/confild/sB1a_s2ckpts.sh` |
| `sB1b_select.sh` | `submissions/baselines/confild/sB1b_select.sh` |
| `sB2_plateau.sh` | `submissions/baselines/confild/sB2_plateau.sh` |
| `sB3_correct.sh` | `submissions/baselines/confild/sB3_correct.sh` |
| `sB4_window1.sh` | `submissions/baselines/confild/sB4_window1.sh` |
| `seed_variance_2d.py` | `analysis/seed_variance_2d.py` |
| `sen_local_xattn.py` | `baselines/senseiver/sen_local_xattn.py` |
| `sen_sweep_fixes.py` | `baselines/senseiver/sen_sweep_fixes.py` |
| `sensor_fingerprints.py` | `baselines/sensor_fingerprints.py` |
| `smoke_archive.sh` | `submissions/smoke/smoke_archive.sh` |
| `smoke_cyl2d.sh` | `submissions/smoke/smoke_cyl2d.sh` |
| `smoke_cyl_surface.sh` | `submissions/smoke/smoke_cyl_surface.sh` |
| `smoke_deeponet.sh` | `submissions/smoke/smoke_deeponet.sh` |
| `smoke_deeponetpp.sh` | `submissions/smoke/smoke_deeponetpp.sh` |
| `smoke_det_bench.sh` | `submissions/smoke/smoke_det_bench.sh` |
| `smoke_eval_sit.sh` | `submissions/smoke/smoke_eval_sit.sh` |
| `smoke_guard.sh` | `submissions/smoke/smoke_guard.sh` |
| `smoke_kolm2d.sh` | `submissions/smoke/smoke_kolm2d.sh` |
| `smoke_kprior.sh` | `submissions/smoke/smoke_kprior.sh` |
| `smoke_operator.sh` | `submissions/smoke/smoke_operator.sh` |
| `smoke_senseiver_iclr.sh` | `submissions/smoke/smoke_senseiver_iclr.sh` |
| `smoke_senseiver_local.sh` | `submissions/smoke/smoke_senseiver_local.sh` |
| `smoke_senseiver_xattn.sh` | `submissions/smoke/smoke_senseiver_xattn.sh` |
| `smoke_spec.sh` | `submissions/smoke/smoke_spec.sh` |
| `spectra_arms.py` | `spectra/spectra_arms.py` |
| `spectra_paper.py` | `spectra/spectra_paper.py` |
| `spectral_loss.py` | `core/spectral_loss.py` |
| `spectral_prior.py` | `core/spectral_prior.py` |
| `spectral_utils.py` | `core/spectral_utils.py` |
| `stage_h5.sh` | `submissions/data/stage_h5.sh` |
| `submit_2d_fleet_delta.sh` | `submissions/fleet_2d/submit_2d_fleet_delta.sh` |
| `submit_2d_seed_replicates_delta.sh` | `submissions/fleet_2d/submit_2d_seed_replicates_delta.sh` |
| `submit_cyl_surface_fleet_delta.sh` | `submissions/fleet_2d/submit_cyl_surface_fleet_delta.sh` |
| `summarize_2d_fleet.py` | `analysis/summarize_2d_fleet.py` |
| `test_artifact_guard.py` | `temp/test_artifact_guard.py` |
| `test_attn_compile.py` | `temp/test_attn_compile.py` |
| `test_attn_fix.py` | `temp/test_attn_fix.py` |
| `test_attn_mechanism.py` | `temp/test_attn_mechanism.py` |
| `test_cq_adapter.py` | `temp/test_cq_adapter.py` |
| `test_cq_flash.py` | `temp/test_cq_flash.py` |
| `test_ode_cache.py` | `temp/test_ode_cache.py` |
| `test_persistent_workers.py` | `temp/test_persistent_workers.py` |
| `test_pw.sh` | `temp/test_pw.sh` |
| `train_baseline_det_firebench.sh` | `submissions/baselines/train_baseline_det_firebench.sh` |
| `train_baseline_det_xcube.sh` | `submissions/baselines/train_baseline_det_xcube.sh` |
| `train_baseline_det_xcube_aug.sh` | `submissions/baselines/train_baseline_det_xcube_aug.sh` |
| `train_baseline_lfm.sh` | `submissions/baselines/train_baseline_lfm.sh` |
| `train_baseline_lfm_firebench.sh` | `submissions/baselines/train_baseline_lfm_firebench.sh` |
| `train_baseline_lfm_xcube.sh` | `submissions/baselines/train_baseline_lfm_xcube.sh` |
| `train_baseline_lfm_xcube_aug.sh` | `submissions/baselines/train_baseline_lfm_xcube_aug.sh` |
| `train_bench_cyl_ffm.sh` | `submissions/fleet_2d/train_bench_cyl_ffm.sh` |
| `train_bench_kolm_ffm.sh` | `submissions/fleet_2d/train_bench_kolm_ffm.sh` |
| `train_confild_unified.slurm` | `submissions/baselines/confild/train_confild_unified.slurm` |
| `train_cyl_baseline.sh` | `submissions/baselines/train_cyl_baseline.sh` |
| `train_deeponet.sh` | `submissions/baselines/deeponet/train_deeponet.sh` |
| `train_deeponetpp.sh` | `submissions/baselines/deeponet/train_deeponetpp.sh` |
| `train_det_sweep.py` | `baselines/senseiver/train_det_sweep.py` |
| `train_ffm.sh` | `submissions/dmfgen/train/train_ffm.sh` |
| `train_fno3d_iotest.sh` | `submissions/dmfgen/train/train_fno3d_iotest.sh` |
| `train_fno3d_matched.sh` | `submissions/dmfgen/train/train_fno3d_matched.sh` |
| `train_fno3d_smoke.sh` | `submissions/smoke/train_fno3d_smoke.sh` |
| `train_gen4turb_jhu.sh` | `submissions/baselines/gen4turb/train_gen4turb_jhu.sh` |
| `train_gen4turb_uxuz.sh` | `submissions/baselines/gen4turb/train_gen4turb_uxuz.sh` |
| `train_iclr_det_senseiver.sh` | `submissions/baselines/senseiver/train_iclr_det_senseiver.sh` |
| `train_iclr_firebench.sh` | `submissions/dmfgen/train/train_iclr_firebench.sh` |
| `train_iclr_firebench_v2.sh` | `submissions/dmfgen/train/train_iclr_firebench_v2.sh` |
| `train_iclr_firebench_v3.sh` | `submissions/dmfgen/train/train_iclr_firebench_v3.sh` |
| `train_iclr_firebench_v4.sh` | `submissions/dmfgen/train/train_iclr_firebench_v4.sh` |
| `train_iclr_firebench_v5clean.sh` | `submissions/dmfgen/train/train_iclr_firebench_v5clean.sh` |
| `train_iclr_jhu_k32.sh` | `submissions/dmfgen/train/train_iclr_jhu_k32.sh` |
| `train_iclr_jhu_k64.sh` | `submissions/dmfgen/train/train_iclr_jhu_k64.sh` |
| `train_iclr_jhu_main.sh` | `submissions/dmfgen/train/train_iclr_jhu_main.sh` |
| `train_iclr_jhu_robust.sh` | `submissions/dmfgen/train/train_iclr_jhu_robust.sh` |
| `train_iclr_jhu_scale.sh` | `submissions/dmfgen/train/train_iclr_jhu_scale.sh` |
| `train_iclr_jhu_xcube.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube.sh` |
| `train_iclr_jhu_xcube_aug.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_aug.sh` |
| `train_iclr_jhu_xcube_cq.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_cq.sh` |
| `train_iclr_jhu_xcube_kprior.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_kprior.sh` |
| `train_iclr_jhu_xcube_local.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_local.sh` |
| `train_iclr_jhu_xcube_ms.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_ms.sh` |
| `train_iclr_jhu_xcube_ms_aug.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_ms_aug.sh` |
| `train_iclr_jhu_xcube_reg_s1337.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_reg_s1337.sh` |
| `train_iclr_jhu_xcube_reg_s42.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_reg_s42.sh` |
| `train_iclr_jhu_xcube_reg_s7.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_reg_s7.sh` |
| `train_iclr_jhu_xcube_spec.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_spec.sh` |
| `train_iclr_jhu_xcube_spec02.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_spec02.sh` |
| `train_iclr_jhu_xcube_specwin.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_specwin.sh` |
| `train_iclr_jhu_xcube_sup_s1337.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_sup_s1337.sh` |
| `train_iclr_jhu_xcube_sup_s42.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_sup_s42.sh` |
| `train_iclr_jhu_xcube_sup_s7.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_sup_s7.sh` |
| `train_iclr_jhu_xcube_sym.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_sym.sh` |
| `train_iclr_jhu_xcube_ti.sh` | `submissions/dmfgen/train/train_iclr_jhu_xcube_ti.sh` |
| `train_iclr_wing.sh` | `submissions/dmfgen/train/train_iclr_wing.sh` |
| `train_iclr_wing_v2.sh` | `submissions/dmfgen/train/train_iclr_wing_v2.sh` |
| `train_iclr_wing_v3.sh` | `submissions/dmfgen/train/train_iclr_wing_v3.sh` |
| `train_iclr_wing_v4.sh` | `submissions/dmfgen/train/train_iclr_wing_v4.sh` |
| `train_kolm_baseline.sh` | `submissions/baselines/train_kolm_baseline.sh` |
| `train_senseiver_iclr.sh` | `submissions/baselines/senseiver/train_senseiver_iclr.sh` |
| `train_senseiver_sweep.sh` | `submissions/baselines/senseiver/train_senseiver_sweep.sh` |
| `train_sit_wing.sh` | `submissions/baselines/sit/train_sit_wing.sh` |
| `train_sit_xcube.sh` | `submissions/baselines/sit/train_sit_xcube.sh` |
| `train_sit_xcube_matched.sh` | `submissions/baselines/sit/train_sit_xcube_matched.sh` |
| `train_sit_xcube_smoke.sh` | `submissions/smoke/train_sit_xcube_smoke.sh` |
| `tune_norm.sh` | `submissions/baselines/s3gm/tune_norm.sh` |
| `tune_norm2.sh` | `submissions/baselines/s3gm/tune_norm2.sh` |
| `tune_norm_guidance.py` | `baselines/s3gm/tune_norm_guidance.py` |
| `tune_norm_guidance2.py` | `baselines/s3gm/tune_norm_guidance2.py` |
| `turbulence_stats.py` | `core/turbulence_stats.py` |
| `verify_seedcheck.py` | `temp/verify_seedcheck.py` |
| `verify_seedcheck.sh` | `temp/verify_seedcheck.sh` |
| `View_Dataset.py` | `data_prep/View_Dataset.py` |
| `_watch_dump_job.sh` | `temp/_watch_dump_job.sh` |
| `baseline_classical_wing.sh` | `submissions/baselines/classical/baseline_classical_wing.sh` |
| `eval_wing_fleet.sh` | `submissions/eval_all_methods/eval_wing_fleet.sh` |
| `helpers_wing_baseline.py` | `baselines/helpers_wing_baseline.py` |
| `smoke_wing_baselines.sh` | `submissions/smoke/smoke_wing_baselines.sh` |
| `train_wing_baseline.sh` | `submissions/baselines/train_wing_baseline.sh` |
| `wing_baseline_preflight.py` | `temp/wing_baseline_preflight.py` |
| `wing_param_probe.py` | `temp/wing_param_probe.py` |
| `probe_wing_progress.sh` | `submissions/ops/probe_wing_progress.sh` |

All `*.log` / `*.out` job logs and `fleet_jobs_delta.txt` moved to `logs/`. Everything not listed stayed at `src/`.
