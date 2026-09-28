#!/usr/bin/env python3
"""Package the 2D Kolmogorov and 2D cylinder benchmarks as self-contained,
shareable directories (zipped separately by the caller).

Both packages are built from the canonical H5 files the training/eval code
reads, re-organized from the H5's train-block-first frame order into an
explicit leading condition axis:

  kolmogorov2d_shu/vorticity.npy        (traj, frame, y, x)
  cylinder2d_openfoam/fields_grid.npy   (Re, frame, y, x, [Ux,Uy,p])
  cylinder2d_openfoam/fields_mesh.npy   (Re, frame, cell, [Ux,Uy,p])

Every reorganization is verified against the raw source (the Shu et al. npy;
the OpenFOAM ASCII time directories) before the package is declared good.
READMEs and package-only helper scripts live in tools/package_2d_assets/ and
are copied in (followed by MD5SUMS) as the last step.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
import _srcpaths  # noqa: E402,F401  (src/ library dirs, 2026-09-27 layout)

DATA = "/work/hdd/bilr/ntricard/datasets"
KOLM_H5 = f"{DATA}/kolmogorov2d/Kolmogorov2D_shu_stride4.h5"
KOLM_MAN = f"{DATA}/kolmogorov2d/kolmogorov2d_manifest.json"
KOLM_RAW = f"{DATA}/baselines/sparse-reconstruction/data/kolmogorov_shu.npy"
CYL_DIR = f"{DATA}/cylinder2d"
CYL_RUNS = f"{CYL_DIR}/runs"


def md5(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_checksums(root: str) -> None:
    rows = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn == "MD5SUMS":
                continue
            p = os.path.join(dp, fn)
            rows.append(f"{md5(p)}  {os.path.relpath(p, root)}")
    with open(os.path.join(root, "MD5SUMS"), "w") as fh:
        fh.write("\n".join(sorted(rows, key=lambda r: r.split("  ", 1)[1])) + "\n")


def copy_into(dst_dir: str, *paths: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for p in paths:
        shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))


# --------------------------------------------------------------------------
# Kolmogorov
# --------------------------------------------------------------------------
def build_kolmogorov(out_root: str) -> None:
    out = os.path.join(out_root, "kolmogorov2d_shu")
    os.makedirs(out, exist_ok=True)
    man = json.load(open(KOLM_MAN))
    with h5py.File(KOLM_H5, "r") as f:
        ny, nx = (int(v) for v in f.attrs["grid_shape"])
        fpt = int(f.attrs["frames_per_traj"])
        stride = int(f.attrs["stride"])
        train = [int(t) for t in f.attrs["train_trajs"]]
        test = [int(t) for t in f.attrs["test_trajs"]]
        seq = train + test  # H5 block order
        n_traj = len(seq)
        assert f["fields"].shape == (1, n_traj * fpt, ny * nx, 1, 1, 1)
        coords = f["coordinates"][:, 0, 0, :2].reshape(ny, nx, 2)

        arr = np.lib.format.open_memmap(
            os.path.join(out, "vorticity.npy"), mode="w+", dtype=np.float32,
            shape=(n_traj, fpt, ny, nx))
        h5_block_of_traj = {}
        for b, tj in enumerate(seq):
            blk = f["fields"][0, b * fpt:(b + 1) * fpt, :, 0, 0, 0]
            arr[tj] = blk.reshape(fpt, ny, nx)  # row-major: y slowest, x fastest
            h5_block_of_traj[tj] = b
        arr.flush()

    x = coords[0, :, 0].astype(np.float32)
    y = coords[:, 0, 1].astype(np.float32)
    assert np.allclose(coords[..., 0], x[None, :]) and np.allclose(coords[..., 1], y[:, None])
    np.save(os.path.join(out, "x.npy"), x)
    np.save(os.path.join(out, "y.npy"), y)
    src_frames = np.arange(0, fpt * stride, stride, dtype=np.int32)
    np.save(os.path.join(out, "source_frame_index.npy"), src_frames)

    # Verify against the raw upstream npy: package[traj, k] == raw[traj, k*stride].
    raw_checked = False
    if os.path.exists(KOLM_RAW):
        raw = np.load(KOLM_RAW, mmap_mode="r")
        assert raw.shape == (n_traj, fpt * stride, ny, nx), raw.shape
        for tj in range(n_traj):
            assert np.array_equal(arr[tj], np.asarray(raw[tj, ::stride], dtype=np.float32)), tj
        raw_checked = True
        print(f"[kolm] verified bit-identical to raw npy[:, ::{stride}] for all {n_traj} trajectories")

    # Recompute train-only stats from the package and compare to the manifest.
    tr = np.asarray(arr[train], dtype=np.float64)
    mean, std = float(tr.mean()), float(tr.std())
    mm = man["train_stats"]["vorticity"]
    assert abs(mean - mm["mean"]) < 1e-6 and abs(std - mm["std"]) < 1e-6, (mean, std, mm)

    meta = {
        "name": "kolmogorov2d_shu (stride-4 benchmark cut)",
        "file": "vorticity.npy",
        "shape": [n_traj, fpt, ny, nx],
        "axes": ["trajectory", "frame", "y", "x"],
        "dtype": "float32",
        "field": "vorticity (single scalar channel; the source ships no velocity)",
        "domain": "[0, 2pi)^2, doubly periodic, uniform 256x256, x = 2pi*i/256",
        "trajectory_index": "identical to the upstream npy's axis 0",
        "temporal_stride": stride,
        "source_frame_index": "frame k of this file = upstream frame 4k (k = 0..79)",
        "split": {"kind": "trajectory holdout", "split_seed": int(man["split_seed"]),
                  "train_trajs": train, "test_trajs": test},
        "train_only_normalization": {"mean": mm["mean"], "std": mm["std"]},
        "h5_mapping": {
            "h5": os.path.basename(KOLM_H5),
            "rule": "H5 frame index = h5_block_of_traj[traj] * 80 + frame",
            "h5_block_of_traj": {str(k): v for k, v in sorted(h5_block_of_traj.items())},
        },
        "upstream_npy": {"name": "kolmogorov_shu.npy", "shape": [40, 320, 256, 256],
                         "md5": md5(KOLM_RAW) if raw_checked else None,
                         "verified_bit_identical_to_upstream_strided": raw_checked},
    }
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2)
    copy_into(os.path.join(out, "scripts"),
              os.path.join(REPO, "src", "data_prep", "convert_kolmogorov.py"),
              os.path.join(REPO, "src", "submissions", "data", "convert_kolmogorov.sh"))
    copy_into(os.path.join(out, "provenance"), KOLM_MAN)
    print(f"[kolm] wrote {out}/vorticity.npy {arr.shape}")


# --------------------------------------------------------------------------
# Cylinder
# --------------------------------------------------------------------------
def wall_adjacent_cells(poly_mesh: str, patch: str) -> np.ndarray:
    """Owner cells of a boundary patch's faces (OpenFOAM ASCII polyMesh)."""
    import re

    btxt = open(os.path.join(poly_mesh, "boundary")).read()
    m = re.search(patch + r"\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);", btxt, re.S)
    n_faces, start = int(m.group(1)), int(m.group(2))
    otxt = open(os.path.join(poly_mesh, "owner")).read()
    body = otxt[otxt.index("\n(") + 2:otxt.rindex(")")]
    owner = np.array(body.split(), dtype=np.int64)
    return np.unique(owner[start:start + n_faces])



def build_cylinder(out_root: str) -> None:
    from convert_cylinder import read_of_internal_field, time_dirs

    out = os.path.join(out_root, "cylinder2d_openfoam")
    os.makedirs(out, exist_ok=True)
    man = json.load(open(f"{CYL_DIR}/cylinder2d_manifest.json"))
    mesh_h5, grid_h5 = f"{CYL_DIR}/Cylinder2D_mesh.h5", f"{CYL_DIR}/Cylinder2D_grid.h5"

    with h5py.File(mesh_h5, "r") as fm, h5py.File(grid_h5, "r") as fg:
        frame_re = fm["frame_re"][:]
        assert np.array_equal(frame_re, fg["frame_re"][:])
        h5_order = list(dict.fromkeys(int(r) for r in frame_re))  # train-first
        res = sorted(h5_order)                                    # package order
        T = int((frame_re == res[0]).sum())
        assert all(int((frame_re == r).sum()) == T for r in res)
        n_pts = fm["coordinates"].shape[0]
        gny, gnx = (int(v) for v in fg.attrs["grid_shape"])

        mesh = np.lib.format.open_memmap(
            os.path.join(out, "fields_mesh.npy"), mode="w+", dtype=np.float32,
            shape=(len(res), T, n_pts, 3))
        grid = np.lib.format.open_memmap(
            os.path.join(out, "fields_grid.npy"), mode="w+", dtype=np.float32,
            shape=(len(res), T, gny, gnx, 3))
        h5_start = {}
        for i, Re in enumerate(res):
            idx = np.where(frame_re == Re)[0]
            assert np.array_equal(idx, np.arange(idx[0], idx[0] + T)), "Re block not contiguous"
            h5_start[Re] = int(idx[0])
            mesh[i] = fm["fields"][0, idx[0]:idx[0] + T, :, 0, 0, :]
            grid[i] = fg["fields"][0, idx[0]:idx[0] + T, :, 0, 0, :].reshape(T, gny, gnx, 3)
            print(f"[cyl] Re={Re}: H5 frames [{idx[0]}, {idx[0] + T})")
        mesh.flush()
        grid.flush()

        mesh_xy = fm["coordinates"][:, 0, 0, :2].astype(np.float32)
        mesh_surf = fm["surface_indices"][:].astype(np.int64)
        gxy = fg["coordinates"][:, 0, 0, :2].reshape(gny, gnx, 2)
        fluid = fg["body_mask"][:].astype(bool).reshape(gny, gnx)  # True = fluid

    gx, gy = gxy[0, :, 0].astype(np.float32), gxy[:, 0, 1].astype(np.float32)
    assert np.allclose(gxy[..., 0], gx[None, :]) and np.allclose(gxy[..., 1], gy[:, None])
    rr = np.hypot(gxy[..., 0], gxy[..., 1])
    assert (rr[~fluid] <= 0.5 + 1e-6).all() and (rr[fluid] > 0.5 - 1e-6).all()
    assert not grid[:, :, ~fluid, :].any(), "body interior should be zero"
    grid_surf = np.load(f"{CYL_DIR}/Cylinder2D_grid.surface_indices.npy").astype(np.int64)

    # Physical time: the H5 only stores a global frame counter; recover the
    # solver times from the run directories and require all Re to agree.
    times = None
    for Re in res:
        t = np.array([float(d) for d in time_dirs(f"{CYL_RUNS}/Re{Re}", 120.0, 270.0)])
        assert len(t) == T
        assert times is None or np.array_equal(times, t)
        times = t
    assert np.allclose(np.diff(times), 0.5)

    # End-to-end provenance check: re-parse raw OpenFOAM output for a few
    # (Re, frame) pairs and require exact equality with the packaged mesh array.
    dirs = {Re: time_dirs(f"{CYL_RUNS}/Re{Re}", 120.0, 270.0) for Re in res}
    for i, Re in enumerate(res):
        for k in (0, T // 2, T - 1):
            case = f"{CYL_RUNS}/Re{Re}/{dirs[Re][k]}"
            u = read_of_internal_field(f"{case}/U", n_pts)
            p = read_of_internal_field(f"{case}/p", n_pts)
            assert np.array_equal(mesh[i, k, :, 0], u[:, 0]), (Re, k)
            assert np.array_equal(mesh[i, k, :, 1], u[:, 1]), (Re, k)
            assert np.array_equal(mesh[i, k, :, 2], p), (Re, k)
    print(f"[cyl] verified mesh array == raw OpenFOAM U/p at 3 frames x {len(res)} Re")
    cc = read_of_internal_field(f"{CYL_RUNS}/Re{res[0]}/0/C")
    assert np.array_equal(cc[:, :2].astype(np.float32), mesh_xy)

    # Train-only stats recomputed from the package vs. the manifest.
    tr_i = [res.index(r) for r in man["train_res"]]
    flat = np.asarray(mesh[tr_i], dtype=np.float64).reshape(-1, 3)
    for c, name in enumerate(["Ux", "Uy", "p"]):
        mm = man["train_stats"][name]
        assert abs(flat[:, c].mean() - mm["mean"]) < 1e-6, name
        assert abs(flat[:, c].std() - mm["std"]) < 1e-6, name

    # Cells that actually own a face on the `cylinder` wall patch, from the mesh
    # topology. The benchmark's surface pool (r < 0.53) is a superset of these:
    # shipped alongside it so the difference is explicit, not used by any model.
    wall_adj = wall_adjacent_cells(f"{CYL_RUNS}/Re{res[0]}/constant/polyMesh", "cylinder")
    assert np.isin(wall_adj, mesh_surf).all()
    r_cell = np.hypot(mesh_xy[:, 0], mesh_xy[:, 1])
    print(f"[cyl] wall-adjacent cells {len(wall_adj)} (r {r_cell[wall_adj].min():.4f}-"
          f"{r_cell[wall_adj].max():.4f}); benchmark surface pool {len(mesh_surf)} "
          f"(r {r_cell[mesh_surf].min():.4f}-{r_cell[mesh_surf].max():.4f})")
    np.save(os.path.join(out, "mesh_wall_adjacent_indices.npy"), wall_adj)

    np.save(os.path.join(out, "reynolds.npy"), np.array(res, dtype=np.int32))
    np.save(os.path.join(out, "time.npy"), times.astype(np.float32))
    np.save(os.path.join(out, "grid_x.npy"), gx)
    np.save(os.path.join(out, "grid_y.npy"), gy)
    np.save(os.path.join(out, "grid_fluid_mask.npy"), fluid)
    np.save(os.path.join(out, "grid_surface_indices.npy"), grid_surf)
    np.save(os.path.join(out, "mesh_xy.npy"), mesh_xy)
    np.save(os.path.join(out, "mesh_surface_indices.npy"), mesh_surf)

    # Strouhal / drag V&V from the solver's own forceCoeffs output.
    from check_strouhal import strouhal
    vv = {}
    for Re in res:
        st, cd, cl = strouhal(f"{CYL_RUNS}/Re{Re}", 120.0)
        vv[str(Re)] = {"St": round(float(st), 4), "mean_Cd": round(cd, 4), "Cl_amplitude": round(cl, 4)}
        print(f"[cyl] Re={Re}: St={st:.4f} Cd={cd:.3f} |Cl|={cl:.3f}")

    meta = {
        "name": "cylinder2d_openfoam",
        "files": {
            "fields_grid.npy": {"shape": list(grid.shape), "axes": ["Re", "frame", "y", "x", "channel"]},
            "fields_mesh.npy": {"shape": list(mesh.shape), "axes": ["Re", "frame", "cell", "channel"]},
        },
        "channels": ["Ux", "Uy", "p"],
        "dtype": "float32",
        "reynolds": res,
        "nondimensionalization": "D = 1, U_inf = 1, rho = 1, nu = 1/Re; p is kinematic (p/rho), outlet p = 0",
        "time": {"first": float(times[0]), "last": float(times[-1]), "dt": 0.5, "n": T,
                 "note": "convective units D/U_inf; t<=120 is discarded start-up transient"},
        "grid": {"roi": man["roi"], "shape_yx": [gny, gnx], "spacing": "linspace endpoints inclusive",
                 "body_cells": int((~fluid).sum()), "surface_pool": int(len(grid_surf)),
                 "flat_index": "iy * 400 + ix (row-major, y slowest)"},
        "mesh": {"n_cells": int(n_pts), "surface_cells": int(len(mesh_surf)),
                 "surface_rule": "cell centres with r < 0.53: the benchmark's surface-sensor pool "
                                 "(wall-adjacent layer + the layer above + part of the third)",
                 "wall_adjacent_cells": int(len(wall_adj)),
                 "wall_adjacent_rule": "cells owning a face of the `cylinder` wall patch (not used by the benchmark)"},
        "split": {"kind": "Reynolds-number holdout", "train_res": man["train_res"], "test_res": man["test_res"]},
        "train_only_normalization": man["train_stats"],
        "h5_mapping": {
            "h5_block_order": h5_order,
            "rule": "H5 frame index = h5_start[Re] + frame",
            "h5_start": {str(k): v for k, v in sorted(h5_start.items())},
        },
        "validation": vv,
    }
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2)

    # OpenFOAM setup, as run.
    of_src = os.path.join(REPO, "openfoam", "cylinder2d")
    of_dst = os.path.join(out, "openfoam")
    if os.path.exists(of_dst):
        shutil.rmtree(of_dst)
    shutil.copytree(of_src, of_dst)
    ref = f"{CYL_RUNS}/Re{res[0]}"
    shutil.copytree(f"{ref}/constant/polyMesh", os.path.join(of_dst, "generated_mesh", "polyMesh"))
    copy_into(os.path.join(of_dst, "generated_mesh"),
              f"{ref}/log.blockMesh", f"{ref}/log.checkMesh", f"{ref}/log.cellCentres")
    for Re in res:
        d = os.path.join(of_dst, "as_run", f"Re{Re}")
        copy_into(d, f"{CYL_RUNS}/Re{Re}/constant/transportProperties",
                  f"{CYL_RUNS}/Re{Re}/system/controlDict")
        for seg in ("0", "120"):
            src = f"{CYL_RUNS}/Re{Re}/postProcessing/forceCoeffs1/{seg}/forceCoeffs.dat"
            shutil.copy2(src, os.path.join(d, f"forceCoeffs_from_t{seg}.dat"))
        # Solver log header (build id, host, date) without the 18 MB of iterations.
        with open(f"{CYL_RUNS}/Re{Re}/log.phase2") as fh, open(os.path.join(d, "log.phase2.head"), "w") as oh:
            for n, line in enumerate(fh):
                if n >= 60:
                    break
                oh.write(line)

    copy_into(os.path.join(out, "scripts"),
              os.path.join(REPO, "src", "data_prep", "convert_cylinder.py"),
              os.path.join(REPO, "src", "submissions", "data", "convert_cylinder.sh"),
              os.path.join(REPO, "src", "data_prep", "add_grid_surface_indices.py"),
              os.path.join(REPO, "src", "data_prep", "check_strouhal.py"))
    copy_into(os.path.join(out, "provenance"), f"{CYL_DIR}/cylinder2d_manifest.json")
    print(f"[cyl] wrote {out}/fields_grid.npy {grid.shape} and fields_mesh.npy {mesh.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{DATA}/shareable")
    ap.add_argument("--only", choices=["kolmogorov", "cylinder"])
    ap.add_argument("--finalize-only", action="store_true",
                    help="skip the array builds; only refresh READMEs/scripts and MD5SUMS")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.finalize_only:
        if args.only in (None, "kolmogorov"):
            build_kolmogorov(args.out)
        if args.only in (None, "cylinder"):
            build_cylinder(args.out)
    # READMEs and package-only helper scripts live next to this file.
    for d in ("kolmogorov2d_shu", "cylinder2d_openfoam"):
        pkg = os.path.join(args.out, d)
        if os.path.isdir(pkg):
            shutil.copytree(os.path.join(HERE, "package_2d_assets", d), pkg, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
            write_checksums(pkg)
            print(f"[finalize] {d}: assets copied, MD5SUMS written")


if __name__ == "__main__":
    main()
