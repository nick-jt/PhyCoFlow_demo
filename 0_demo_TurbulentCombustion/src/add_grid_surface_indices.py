#!/usr/bin/env python3
"""Add a `surface_indices` dataset to Cylinder2D_grid.h5 (the 400x200 ROI
companion export) so grid-locked methods can run the surface-to-field task.

Definition: for each of the 360 wall-adjacent mesh cells listed in
Cylinder2D_mesh.h5/surface_indices, take the nearest grid cell that is NOT
inside the body mask; the pool is the sorted set of unique such cells. The
pool is smaller than 360 because a thin curved ring aliases onto a Cartesian
grid whose spacing (0.05) is coarser than the mesh's wall-ring spacing --
that collapse is exactly the rasterization cost the surface task measures,
and it is recorded (n_mesh_surface, n_grid_pool) in the manifest.
Idempotent: re-running overwrites the dataset with identical content.
"""
import json, sys
import numpy as np, h5py
from scipy.spatial import cKDTree

D = "/work/hdd/bilr/ntricard/datasets/cylinder2d"
mesh, grid, man = f"{D}/Cylinder2D_mesh.h5", f"{D}/Cylinder2D_grid.h5", f"{D}/cylinder2d_manifest.json"
with h5py.File(mesh, "r") as f:
    si = f["surface_indices"][:]
    sc = f["coordinates"][si, 0, 0, :2]
with h5py.File(grid, "r") as g:
    gc = g["coordinates"][:, 0, 0, :2]
    bm = g["body_mask"][:].astype(bool)
# body_mask semantics are inferred, not assumed: the body interior is the
# minority of the 80000 ROI cells and sits at r < 0.5.
r_all = np.hypot(gc[:, 0], gc[:, 1])
body = bm if bm.sum() < bm.size / 2 else ~bm
assert r_all[body].max() < 0.5 + 0.05 and r_all[~body].min() > 0.45, "body_mask semantics unclear"
print(f"body_mask stored as {'body=True' if body is bm else 'fluid=True'}; body cells {body.sum()}")
free = np.where(~body)[0]
_, nn = cKDTree(gc[free]).query(sc, k=1)
pool = np.unique(free[nn]).astype(np.int64)
r = np.hypot(gc[pool, 0], gc[pool, 1])
DEFN = ("unique nearest non-body grid cell of each Cylinder2D_mesh.h5 "
        "surface_indices point (first wall-adjacent O-ring layer, r<0.53)")
side = grid.replace(".h5", ".surface_indices.npy")
np.save(side, pool)   # sidecar: what the dataset classes fall back to
print(f"sidecar written: {side}")
try:
    with h5py.File(grid, "r+") as g:
        if "surface_indices" in g:
            del g["surface_indices"]
        ds = g.create_dataset("surface_indices", data=pool)
        ds.attrs["definition"] = DEFN
        ds.attrs["n_mesh_surface"] = int(len(si))
    print("surface_indices written into the grid H5")
except BlockingIOError:
    print("grid H5 is locked by running readers; H5 dataset NOT written -- the "
          "sidecar is authoritative until this script is re-run on an idle file")
print(f"mesh surface pts {len(si)} -> grid pool {len(pool)} cells; "
      f"r range [{r.min():.3f}, {r.max():.3f}]")
m = json.load(open(man))
m["surface_pool"] = {"mesh_h5": int(len(si)), "grid_h5": int(len(pool)),
                     "grid_definition": "unique nearest non-body cell per mesh surface point"}
json.dump(m, open(man, "w"), indent=2)
print("manifest updated")
