#!/usr/bin/env python3
"""Minimal loader for the packaged Kolmogorov benchmark (numpy only)."""

import json
import os

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta = json.load(open(os.path.join(root, "metadata.json")))

# (trajectory, frame, y, x); mmap so nothing is read until it is indexed.
w = np.load(os.path.join(root, "vorticity.npy"), mmap_mode="r")
x = np.load(os.path.join(root, "x.npy"))
y = np.load(os.path.join(root, "y.npy"))
print("vorticity", w.shape, w.dtype, "| x", x.shape, "| y", y.shape)

train_ids = meta["split"]["train_trajs"]   # 32 trajectories
test_ids = meta["split"]["test_trajs"]     # 8 held-out trajectories
train, test = w[train_ids], w[test_ids]    # fancy-indexing copies into RAM
print("train", train.shape, "test", test.shape)

# Normalization used by the benchmark: statistics of TRAIN frames only.
st = meta["train_only_normalization"]
train_n = (train - st["mean"]) / st["std"]
print(f"normalized train: mean {train_n.mean():+.2e}, std {train_n.std():.4f}")

# One field: trajectory 5 (held out), frame 10 -> w[5, 10] is (256, 256),
# indexed [iy, ix]; its location is (x[ix], y[iy]).
snap = w[5, 10]
print("snapshot", snap.shape, "min/max", float(snap.min()), float(snap.max()))
