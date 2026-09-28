# 2D laminar cylinder wake, Re = 60–250 (OpenFOAM 9) — sparse-reconstruction benchmark

Unsteady incompressible flow past a circular cylinder at six Reynolds numbers,
300 snapshots each of `(Ux, Uy, p)` in the periodic vortex-shedding regime.
**We generated this dataset ourselves** with OpenFOAM 9 (`pimpleFoam`); the
complete case setup is in `openfoam/` and summarized below. This is the exact
data the PhyCoFlow / PoF-2026 sparse-reconstruction benchmark trains and
evaluates on, reorganized so that Reynolds number is an explicit axis.

Everything is nondimensional: cylinder diameter `D = 1` centred at the origin,
free stream `U∞ = 1` along `+x`, `ρ = 1`, `ν = 1/Re`. Time is in `D/U∞`.
`p` is OpenFOAM's kinematic pressure `p/ρ`, referenced to `p = 0` at the
outlet (so `Cp = 2p`).

## Contents

| File | Shape / type | Meaning |
|---|---|---|
| `fields_grid.npy` | `(6, 300, 200, 400, 3)` float32, 1.73 GB | `[Re, frame, iy, ix, channel]` on a uniform grid — **the image-like array most users want** |
| `fields_mesh.npy` | `(6, 300, 23800, 3)` float32, 514 MB | `[Re, frame, cell, channel]` at the native finite-volume cell centres — the solver's actual output |
| `reynolds.npy` | `(6,)` int32 | `[60, 80, 100, 150, 200, 250]` — labels of axis 0 |
| `time.npy` | `(300,)` float32 | solver time of each frame: `120.5, 121.0, …, 270.0` (identical for every Re) |
| `grid_x.npy`, `grid_y.npy` | `(400,)`, `(200,)` | grid coordinates: `linspace(-3, 17, 400)`, `linspace(-5, 5, 200)` (endpoints included; Δx = 0.0501, Δy = 0.0503) |
| `grid_fluid_mask.npy` | `(200, 400)` bool | `True` = fluid; `False` = the 316 grid nodes inside the cylinder (`r ≤ 0.5`), where fields are stored as 0 |
| `grid_surface_indices.npy` | `(62,)` int64 | near-wall sensor pool on the grid, flat index `iy*400 + ix` |
| `mesh_xy.npy` | `(23800, 2)` float32 | cell-centre coordinates (same mesh for every Re) |
| `mesh_surface_indices.npy` | `(360,)` int64 | near-wall sensor pool on the mesh **as used by the benchmark** (see caveat below) |
| `mesh_wall_adjacent_indices.npy` | `(160,)` int64 | cells that truly touch the wall; provided for reference, *not* used by the benchmark |
| `metadata.json` | — | shapes, split, normalization stats, H5 index mapping, St/Cd validation |
| `openfoam/` | — | full solver setup, generated mesh, per-Re as-run dictionaries and force histories |
| `scripts/` | — | loader example + the converters/validation that produced these arrays |
| `provenance/cylinder2d_manifest.json` | — | manifest written by the converter (paths are from the original host) |
| `MD5SUMS` | — | checksums of every file in the package |

Channels (last axis) are always `[Ux, Uy, p]`. There is no `Uz` (2D case).

```python
import numpy as np
g   = np.load("fields_grid.npy", mmap_mode="r")    # (Re, frame, y, x, channel)
res = np.load("reynolds.npy").tolist()
ux  = g[res.index(100), 0, :, :, 0]                # Ux(y, x), Re = 100, t = 120.5
```

## Organization: by Reynolds number

Axis 0 is the flow condition, in **ascending Re**; axis 1 is time within that
run. The only thing that differs between conditions is `ν = 1/Re` — mesh,
boundary conditions, numerics and sampling window are identical.

| axis-0 index | Re | ν | role in benchmark | St (lift FFT) | mean C_d | C_l amplitude |
|---|---|---|---|---|---|---|
| 0 | 60  | 0.016667 | train | 0.140 | 1.459 | 0.158 |
| 1 | 80  | 0.0125   | **held out** (interpolative) | 0.160 | 1.447 | 0.282 |
| 2 | 100 | 0.01     | train | 0.173 | 1.418 | 0.382 |
| 3 | 150 | 0.006667 | train | 0.187 | 1.398 | 0.588 |
| 4 | 200 | 0.005    | train | 0.200 | 1.408 | 0.750 |
| 5 | 250 | 0.004    | **held out** (mild extrapolation) | 0.207 | 1.424 | 0.878 |

Split = Reynolds-number holdout: train on {60, 100, 150, 200}, evaluate on
{80, 250}. Normalization is per-channel mean/std over **train-Re frames
only**, computed on the mesh array (values in `metadata.json`): Ux 0.91781 ±
0.29020, Uy 1.2e−05 ± 0.17777, p −0.03803 ± 0.16819.

## The two spatial representations

**`fields_mesh.npy` — native.** Cell-centre values exactly as written by
OpenFOAM (ASCII, 8 significant digits, cast to float32). The packaging script
re-parsed raw `U`/`p` files for 3 frames × 6 Re and confirmed exact equality.
Cells are unstructured from the user's point of view (12-block structured
mesh, OpenFOAM cell numbering); use `mesh_xy.npy` for positions. The mesh
covers the full domain `x ∈ [-8, 25]`, `y ∈ [-8, 8]`.

**`fields_grid.npy` — derived.** The mesh fields linearly interpolated
(Delaunay triangulation of the cell centres, `scipy.interpolate.LinearNDInterpolator`)
onto a 400 × 200 uniform grid over the region of interest `x ∈ [-3, 17]`,
`y ∈ [-5, 5]`. Nodes with `r ≤ 0.5` are set to 0 and flagged in
`grid_fluid_mask.npy`. This is an interpolation of a second-order FV solution,
not a re-simulation: it is not exactly divergence-free, and the near-wall
boundary layer (first cell height ≈ 0.011 D) is under-resolved at Δx ≈ 0.05.
Use the mesh array for anything wall-sensitive.

### Near-wall sensor pools — caveat

The benchmark's surface-sensing task draws sensors from
`mesh_surface_indices.npy`, defined as *all cells with centre radius
`r < 0.53`* — 360 cells. The O-grid has 160 cells around the circumference, so
this pool is **not** a single wall-adjacent ring: it is the wall-adjacent layer
(160 cells, `r` = 0.5055–0.5084), the full second layer (160 cells), and the 40
third-layer cells nearest the axes (max `r` = 0.5297). Every pool cell is within
0.03 D of the wall. The converter's comment and some descriptions call this
"the first O-ring cell layer"; that wording is inaccurate, the definition
above is what was actually used. The strictly wall-adjacent set, from the mesh
topology (owners of the `cylinder` patch faces), is provided separately as
`mesh_wall_adjacent_indices.npy`.

`grid_surface_indices.npy` is the grid counterpart: for each of the 360 mesh
pool cells, the nearest grid node outside the body; unique nodes only → 62.

## How the data was generated (OpenFOAM setup)

Solver: **OpenFOAM 9** (openfoam.org, build `9-d87800e1bde0`), `pimpleFoam`,
serial (`nProcs 1`), run 2026-09-06 on NREL Kestrel. Six independent runs, one
per Re, from `openfoam/submit_all.sh` → `openfoam/run_case.sh`.

| Aspect | Setting | File |
|---|---|---|
| Equations | incompressible Navier–Stokes, `simulationType laminar`, Newtonian, `ν = 1/Re` (substituted for `NU_VALUE` per case) | `constant/momentumTransport`, `constant/transportProperties` |
| Domain | `x ∈ [-8, 25]`, `y ∈ [-8, 8]`, cylinder `r = 0.5` at origin; one cell thick in `z` (−0.05…0.05), `frontAndBack` = `empty` → strictly 2D. Blockage D/H = 6.25 % | `system/blockMeshDict` |
| Mesh | `blockMesh`, 12 hex blocks, **23 800 cells**: 4 O-grid quadrants 40 × 40 around the cylinder out to the square `±2` (radial expansion 8 → first cell ≈ 0.011 D), wake band 120 × 40 (expansion 4), top/bottom/left bands and 4 corner blocks graded away from the body. `checkMesh`: OK; max non-orthogonality 43.9° (mean 10.1°), max skewness 0.49, max aspect ratio 4.2 | `system/blockMeshDict`, `generated_mesh/` |
| Inlet (`x = −8`) | `U = (1 0 0)` fixedValue; `p` zeroGradient | `0/U`, `0/p` |
| Outlet (`x = 25`) | `U` inletOutlet (inletValue 0); `p = 0` fixedValue | 〃 |
| Top / bottom (`y = ±8`) | `symmetryPlane` (free-slip) | 〃 |
| Cylinder | `U` noSlip; `p` zeroGradient | 〃 |
| Initial condition | uniform `U = (1 0 0)`, `p = 0`; no perturbation — shedding develops from numerical asymmetry during the discarded transient | 〃 |
| Time scheme | `backward` (2nd order implicit) | `system/fvSchemes` |
| Convection | `Gauss linearUpwind grad(U)` (2nd order upwind) | 〃 |
| Gradients / Laplacian / snGrad | `Gauss linear` / `Gauss linear corrected` / `corrected` | 〃 |
| Pressure–velocity coupling | PIMPLE with `nOuterCorrectors 1`, `nCorrectors 2`, `nNonOrthogonalCorrectors 1` (i.e. PISO mode) | `system/fvSolution` |
| Linear solvers | `p`: GAMG + GaussSeidel, tol 1e−6, relTol 0.01 (`pFinal` relTol 0); `U`: smoothSolver + symGaussSeidel, tol 1e−8 | 〃 |
| Time step | adaptive, `maxCo 0.9`, `maxDeltaT 0.02`, initial `deltaT 0.002` | `system/controlDict` |
| Output | ASCII, `writePrecision 8` | 〃 |
| Forces | `forceCoeffs` function object on the cylinder patch every 5 steps (`magUInf 1`, `lRef 1`, `Aref 0.1` = D × z-thickness) | 〃 |

**Two-phase schedule** (`run_case.sh`): phase 1 integrates `t = 0 → 120` and
writes only the final state (start-up transient, discarded); the script then
edits `controlDict` to `endTime 270`, `writeInterval 0.5` and restarts from
`latestTime`, giving **300 snapshots at t = 120.5 … 270.0** — about 25
shedding cycles at Re = 100, ≈ 11.6 frames per cycle there. Cell centres are
written once with `postProcess -func writeCellCentres -time 0`.

`openfoam/` layout:

- `case_template/` — the case as submitted (`NU_VALUE` placeholder, phase-1
  `controlDict`).
- `run_case.sh`, `submit_all.sh`, `env_test.sh` — launchers. Note: the SLURM
  header in `submit_all.sh` was later ported to another cluster, while the
  OpenFOAM `source`/`LD_LIBRARY_PATH` lines are still the Kestrel paths the
  data was generated with, and `env_test.sh` (a pre-flight smoke test) has
  hard-coded paths from the original host. Adapt these to your site; the case
  itself is not machine-specific.
- `generated_mesh/` — the `polyMesh` actually produced by `blockMesh` plus
  `log.blockMesh`, `log.checkMesh`, `log.cellCentres`.
- `as_run/Re*/` — per-case `transportProperties` (with the substituted ν),
  `controlDict` (in its final, phase-2 state), the first 60 lines of the
  phase-2 solver log (build id, host, date), and the full force-coefficient
  histories `forceCoeffs_from_t0.dat` / `forceCoeffs_from_t120.dat`.

To regenerate from scratch: `bash openfoam/run_case.sh <Re> <dest_dir>` in an
OpenFOAM-9 environment, then `python scripts/convert_cylinder.py --runs
<dest_dir> --out-dir <out> --verify` (writes the benchmark H5 pair; needs
`h5py`, `scipy`), then `scripts/add_grid_surface_indices.py`.

Not shipped: the raw OpenFOAM time directories (≈ 0.9 GB per Re of ASCII,
including the `phi` flux fields and old-time-level fields) and the full solver
logs. `fields_mesh.npy` carries all of the `U` and `p` content of those
directories for the sampled window.

## Validation

Strouhal number from the FFT of the lift coefficient over `t ≥ 120`
(`scripts/check_strouhal.py`, Hann window), and mean drag, are tabulated above
and in `metadata.json`. Reference values (Williamson 1996; Norberg 2003):
St ≈ 0.135, 0.164, 0.183, 0.196 at Re = 60, 100, 150, 200; C_d(100) ≈ 1.33.
Ours are 3–5 % high in St and ≈ 7 % high in C_d at Re = 100, in the direction
expected from the 6.25 % blockage with slip walls and 2D simulation; trend and
spacing with Re are reproduced and lift amplitude grows monotonically. Two
limitations worth knowing: the FFT bin width over the 150-time-unit window is
≈ 0.0067, so St is quantized at about that level; and Re = 200–250 is past the
onset of three-dimensional wake instability (Re ≈ 190) in experiments, so
those cases are 2D idealizations, not physical wakes.

## Mapping to the benchmark H5 files

The training code reads `Cylinder2D_mesh.h5` / `Cylinder2D_grid.h5` with
`fields (1, 1800, N, 1, 1, 3)` whose frames are ordered **train Re first**:
`[60, 100, 150, 200, 80, 250]` × 300, so that a contiguous block split at
`train_ratio = 2/3` reproduces the holdout. This package re-sorts to ascending
Re. Hence

    H5 frame index = h5_start[Re] + frame,
    h5_start = {60: 0, 100: 300, 150: 600, 200: 900, 80: 1200, 250: 1500}

Grid points in the H5 are flattened row-major (`iy*400 + ix`); the H5
`body_mask` dataset is this package's `grid_fluid_mask.npy` flattened
(`True` = fluid, despite the name).

## Citation / license

Generated by the authors of the PhyCoFlow sparse-reconstruction benchmark;
please cite the accompanying paper when using it. Redistribution terms are
set by the authors; ask them if no license file accompanies this package.
