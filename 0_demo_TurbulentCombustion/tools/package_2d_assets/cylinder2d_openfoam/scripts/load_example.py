#!/usr/bin/env python3
"""Minimal loader for the packaged 2D cylinder dataset (numpy only)."""

import json
import os

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta = json.load(open(os.path.join(root, "metadata.json")))
res = np.load(os.path.join(root, "reynolds.npy")).tolist()   # [60, 80, 100, 150, 200, 250]
t = np.load(os.path.join(root, "time.npy"))                  # 120.5 ... 270.0

# Uniform-grid export: (Re, frame, y, x, [Ux, Uy, p]). mmap -> lazy reads.
g = np.load(os.path.join(root, "fields_grid.npy"), mmap_mode="r")
gx = np.load(os.path.join(root, "grid_x.npy"))
gy = np.load(os.path.join(root, "grid_y.npy"))
fluid = np.load(os.path.join(root, "grid_fluid_mask.npy"))   # (200, 400) True = fluid
print("grid", g.shape, "| Re", res, "| t", t[0], "->", t[-1])

i = res.index(100)
ux = np.array(g[i, 0, :, :, 0])          # Ux at Re=100, first sampled frame
ux[~fluid] = np.nan                      # body interior is stored as 0; mask it for plots
print(f"Re=100 Ux range in fluid: {np.nanmin(ux):.3f} .. {np.nanmax(ux):.3f}")

# Native finite-volume cells: (Re, frame, cell, [Ux, Uy, p]) at mesh_xy.
m = np.load(os.path.join(root, "fields_mesh.npy"), mmap_mode="r")
xy = np.load(os.path.join(root, "mesh_xy.npy"))              # (23800, 2)
wall = np.load(os.path.join(root, "mesh_surface_indices.npy"))  # 360 wall-adjacent cells
p_wall = m[i, 0, wall, 2]
print("mesh", m.shape, "| wall-ring pressure:", p_wall.shape,
      f"stagnation-side max {float(p_wall.max()):.3f}")

# Reynolds-number holdout used by the benchmark.
train = [res.index(r) for r in meta["split"]["train_res"]]   # 60, 100, 150, 200
test = [res.index(r) for r in meta["split"]["test_res"]]     # 80, 250
print("train Re blocks", train, "| test Re blocks", test)

# Train-only per-channel normalization.
st = meta["train_only_normalization"]
mean = np.array([st[c]["mean"] for c in ("Ux", "Uy", "p")], dtype=np.float32)
std = np.array([st[c]["std"] for c in ("Ux", "Uy", "p")], dtype=np.float32)
z = (m[train[0], :10] - mean) / std
print("normalized sample", z.shape)
