# Cylinder surface-to-field task — pre-registration (2026-09-08, before any run)

**Task.** Reconstruct the full 2D wake (Ux, Uy, p on all 23,800 mesh cells /
80,000 ROI grid cells) from sensors confined to the cylinder surface: the 360
wall-adjacent cells (`surface_indices` in `Cylinder2D_mesh.h5`). All three
fields are observed at the taps. Tap budgets 32 / 64 / 128 / 360 per field.
Held-out Reynolds numbers {80, 250}; 50 Re-stratified frames; K=8; canonical
seeded draws (`helpers.build_sparse_condition(valid_mask=surface)` under
`torch.manual_seed(seed*777+snap)`); training draws U{32..360} taps/field.

**Grid-family discretization.** Grid-locked methods (latent-FM, SiT, S3GM,
GeoFNO) train and evaluate on `Cylinder2D_grid.h5`; their pool is the unique
nearest fluid grid cell of each mesh surface point = **62 cells** (grid spacing
0.05 vs. the mesh's 360-cell ring). Counts above 62 are capped, so the 128 and
360 budgets degenerate to the full 62-cell pool for that family. This collapse
is the rasterization cost the task is designed to expose and is reported, not
hidden. Sensor layouts are bit-identical within a discretization family and
count-matched across families, exactly as in the canonical cylinder protocol.

**Pre-registered predictions** (report whichever way they land):
1. DMF-Gen leads on CRPS and far-wake accuracy; its gap to grid methods grows
   as the tap count falls.
2. Interpolation (IDW / kd-tree) collapses toward the train-mean floor: it
   cannot extrapolate off the sensor manifold.
3. Honest risk: gappy POD may still win (rank-20 = 99.7 % energy). If it does,
   report it as the headline of the section — it sharpens the regime-dependence
   claim rather than weakening the paper.

**Fixed before running:** configs `Save_config/cylinder2d_surface/*`, eval
`eval_kolm_fleet.sh DATASET=cylinder2d_surface` (`--cond-source surface`),
classical `run_classical_cyl_surface.sh`. Model selection on TUNE (odd indices),
reporting on TEST (even) as everywhere else.
