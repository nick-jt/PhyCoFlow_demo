"""Classical anchor baselines for the 2D canonical-H5 reconstruction protocol.

2D port of ``baseline_classical_jhu.py`` (read them side by side; the
structure, function names, and protocol conventions are mirrored on purpose so
a reviewer can diff the two files).  Four CPU-only, training-free (or SVD-only)
estimators that a reviewer will ask for as the floor of the table:

  1. ``kdtree``   nearest-neighbour interpolation from the sensors
                  (``scipy.spatial.cKDTree``; periodic wrap only with
                  ``--periodic``).
  2. ``idw``      inverse-distance-weighted interpolation over k neighbours.
  3. ``gappy_pod`` Everson & Sirovich (1995) gappy POD: a POD basis built from
                  the TRAINING frames only, modal coefficients recovered by
                  least squares on the sparse sensor readings, ALL channels
                  reconstructed from the basis.
  4. ``constant`` the trivial floor: predict the TRAIN-split per-channel mean
                  everywhere.

Data layout
-----------
Canonical 2D H5: ``coordinates (N,1,1,3)`` with a constant z column (column 2
is ignored), ``fields (1,T,N,1,1,F)``.  Frames ``[0, --train-frames)`` are the
train split, the rest are the test split.  A sidecar manifest
(``<stem>_manifest.json`` or ``kolmogorov2d_manifest.json`` next to the H5) is
consulted for the train/test boundary when ``--train-frames`` is not given.

Unobserved channels
-------------------
Only the ``--cond-fields`` channels carry sensors.  A nearest-neighbour or IDW
interpolator has literally nothing to interpolate for the other channels, so
it predicts the train-split mean there, which in z-score units is exactly 0.
That is the honest choice: any other value would be smuggling in information
no sensor supplied.  Gappy POD is different -- it can in principle infer the
unobserved channels through the cross-channel correlations captured by the POD
basis -- so it reconstructs all channels from the basis.  (Identical to the
3D script's treatment of Uy and p.)

The constant-predictor floor
----------------------------
The fields are z-scored with TRAIN-split statistics, so the train-split mean
is identically 0 in the evaluation units and the constant predictor scores
relative L2 == 1.0 in every channel by construction.  1.0 is the "I know
nothing" line, and CRPS = E|y| is the corresponding probabilistic line.
Predicting the *test*-split mean instead would score slightly BELOW 1.0 while
quietly using held-out information, so it is reported here only as a labelled
contrast (``constant_test_mean_CONTRAST``), never as the floor.

Periodicity
-----------
DEFAULT IS NON-PERIODIC (a past bug came from wrongly-periodic floors on a
non-periodic cutout; audit 2026-08-28 s26).  Pass ``--periodic`` only for
genuinely periodic domains (e.g. Kolmogorov flow on [0,2pi)^2): distances then
wrap dx and dy by the domain period, implemented by rescaling the grid to the
unit box (exactly as the 3D script does) and handing ``boxsize=1`` to cKDTree.

Seeding / fairness
------------------
The sensor draws are produced by calling ``helpers.build_sparse_condition``
itself, under the same ``torch.manual_seed(seed * 777 + snapshot_index)``
convention that ``ensemble_eval.main`` uses (snapshot_index = 0-based position
within the evaluated split), with the tensors on ``--sensor-device``.
``torch.randperm`` on CUDA and on CPU give different permutations from the
same seed, so canonical draws are done on CUDA compute nodes (as
``ensemble_eval`` does) and the resulting indices are then used by these CPU
estimators.  Any model eval that draws under the same seed/device therefore
sees bit-identical sensor sets.

Metrics come from ``ensemble_eval.ensemble_metrics`` so the JSON schema
matches every other method.  A deterministic estimator is passed as a
two-member ensemble of identical members: the fair-CRPS estimator then returns
exactly the MAE (the K -> 1 limit is 0/0 and cannot be evaluated directly),
and the spread is exactly 0.

Usage:
    python baseline_classical_2d.py --methods kdtree idw gappy_pod constant \
        --h5 /path/to/Kolmogorov2D_shu_stride4.h5 --train-frames 2560 \
        --cond-fields 0 --fields vorticity --n-sensors <1% of N> \
        --out-dir ../Save_TrainedModel/Kolmogorov2D/baseline_classical

Self test (CPU, no data files needed):
    python baseline_classical_2d.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from helpers import build_sparse_condition  # noqa: E402
from ensemble_eval import (  # noqa: E402
    ensemble_metrics,
    require_compute_node,
)


# ---------------------------------------------------------------------------
# np.percentile fast path (see --verify-percentile)
# ---------------------------------------------------------------------------
# Same exact-rewrite of numpy's 'linear' percentile as in the 3D script.  At
# 2D problem sizes it is not the bottleneck it was on the 1.95M-point cube,
# but it is kept (a) so the two files diff cleanly and (b) so a dense-grid 2D
# dataset never reintroduces the 125 s/snapshot pathology.

_np_percentile = np.percentile


def _percentile_axis0(a, q, axis=None, **kw):
    a_ = np.asarray(a)
    if axis != 0 or a_.ndim != 2 or kw or not np.issubdtype(a_.dtype, np.floating):
        return _np_percentile(a, q, axis=axis, **kw)
    x = np.sort(a_, axis=0)
    k = x.shape[0]
    qq = np.atleast_1d(np.asarray(q, dtype=np.float64))
    vi = qq / 100.0 * (k - 1)
    lo = np.floor(vi).astype(np.intp)
    hi = np.ceil(vi).astype(np.intp)
    fr = (vi - lo).astype(x.dtype)
    out = np.stack([x[l] + f * (x[h] - x[l]) for l, h, f in zip(lo, hi, fr)], axis=0)
    return out if np.ndim(q) else out[0]


def enable_percentile_fastpath():
    np.percentile = _percentile_axis0


def disable_percentile_fastpath():
    np.percentile = _np_percentile


DATA = ("/projects/ammoniacomb/generative_reconstruction/kolmogorov2d/"
        "Kolmogorov2D_shu_stride4.h5")
FIELD_NAMES = ("vorticity",)
COND_FIELDS = [0]
TRAIN_FRAMES_DEFAULT = 2560   # Kolmogorov: 32 train trajectories x 80 frames


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


class Timer:
    def __init__(self):
        self.wall = 0.0
        self.cpu = 0.0

    def __enter__(self):
        self._w = time.perf_counter()
        self._c = time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall += time.perf_counter() - self._w
        self.cpu += time.process_time() - self._c


# ---------------------------------------------------------------------------
# sensors -- same draw as ensemble_eval's convention
# ---------------------------------------------------------------------------

def draw_sensors(coords_t: torch.Tensor, fields_t: torch.Tensor, snap: int,
                 n_obs: int, seed: int, device: str,
                 cond_fields: Sequence[int] = COND_FIELDS):
    """Reproduce the ensemble_eval sensor-draw convention exactly.

    Per snapshot:
        torch.manual_seed(seed * 777 + snap)
        build_sparse_condition(coords[1,N,D], fields[1,N,C], cond_fields,
                               n_obs_min=n_obs, n_obs_max=n_obs)
    with the tensors on `device`.  The index draw is a torch.randperm over the
    N points per conditioned field, so any eval that calls the same function
    under the same seed on the same device gets identical index sets bit for
    bit (torch.randperm differs between CPU and CUDA, hence --sensor-device).
    """
    c = coords_t.unsqueeze(0).to(device)
    f = fields_t.unsqueeze(0).to(device)
    torch.manual_seed(seed * 777 + int(snap))
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=c, fields_full=f,
        cond_fields=list(cond_fields), n_obs_min=[n_obs] * len(cond_fields),
        n_obs_max=[n_obs] * len(cond_fields),
    )
    idx = oi[0].cpu().numpy()
    fid = ofid[0].cpu().numpy()
    val = ov[0, :, 0].cpu().numpy()
    out = {}
    for fld in cond_fields:
        m = fid == fld
        out[int(fld)] = (idx[m].astype(np.int64), val[m].astype(np.float32))
    del c, f, oc, ov, om, oi, ofid
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def kd_predict(coords_box: np.ndarray, sensors: Dict, n_ch: int,
               mode: str, k: int, boxsize):
    """NN (mode='nn') or inverse-distance-weighted (mode='idw') interpolation.

    Unobserved channels are left at 0 == the train-split mean in z-score units.
    """
    from scipy.spatial import cKDTree
    pred = np.zeros((coords_box.shape[0], n_ch), dtype=np.float32)
    for fld, (idx, val) in sensors.items():
        tree = cKDTree(coords_box[idx], boxsize=boxsize)
        if mode == "nn":
            _, nn = tree.query(coords_box, k=1, workers=-1)
            pred[:, fld] = val[nn]
        else:
            kk = min(k, len(idx))
            d, nn = tree.query(coords_box, k=kk, workers=-1)
            if kk == 1:
                d = d[:, None]
                nn = nn[:, None]
            exact = d[:, 0] < 1e-11
            d = np.maximum(d, 1e-12)
            w = 1.0 / d
            pred[:, fld] = (w * val[nn]).sum(1) / w.sum(1)
            pred[exact, fld] = val[nn[exact, 0]]
    return pred


class GappyPOD:
    """POD basis over training snapshots, coefficients from sparse sensors.

    The basis is never materialised.  With the snapshot (method-of-snapshots)
    decomposition  Xc = U S V^T  restricted to rank r, the modes are
    Phi = S^-1 U^T Xc, so any modal reconstruction  a @ Phi  equals  w @ Xc
    with  w = U S^-1 a.  Only Xc (n_train x N*C) is kept -- materialising Phi
    as well would double the memory for nothing.
    """

    def __init__(self, Xc: np.ndarray, mu: np.ndarray, rank: int):
        self.Xc = Xc                       # [n_train, P] centred, float32
        self.mu = mu                       # [P]
        G = (Xc @ Xc.T).astype(np.float64)  # [n, n] Gram
        lam, U = np.linalg.eigh(G)
        order = np.argsort(lam)[::-1]
        lam, U = lam[order], U[:, order]
        lam = np.clip(lam, 1e-12, None)
        self.lam = lam
        self.U = U
        self.energy = np.cumsum(lam) / lam.sum()
        self.set_rank(rank)

    def set_rank(self, r: int):
        self.r = int(min(r, self.Xc.shape[0] - 1))
        self.Ur = self.U[:, :self.r]
        self.inv_s = 1.0 / np.sqrt(self.lam[:self.r])

    def modes_at(self, cols: np.ndarray) -> np.ndarray:
        """Phi[:r, cols].T  ->  [n_obs, r]."""
        sub = self.Xc[:, cols].astype(np.float64)              # [n_train, n_obs]
        return (sub.T @ self.Ur) * self.inv_s[None, :]         # [n_obs, r]

    def reconstruct(self, cols: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        A = self.modes_at(cols)
        a, *_ = np.linalg.lstsq(A, y_obs - self.mu[cols].astype(np.float64),
                                rcond=None)
        w = (self.Ur * self.inv_s[None, :]) @ a                # [n_train]
        return self.mu + (w.astype(np.float32) @ self.Xc)


def obs_columns(sensors: Dict, n_ch: int):
    """Flatten (point, channel) sensor addresses into columns of the [N*C] vector."""
    cols, vals = [], []
    for fld, (idx, val) in sorted(sensors.items()):
        cols.append(idx * n_ch + fld)
        vals.append(val)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    o = np.argsort(cols)
    return cols[o], vals[o].astype(np.float64)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(pred: np.ndarray, true: np.ndarray, field_names: Sequence[str]) -> Dict:
    """Deterministic prediction -> ensemble_eval metrics.

    K=1 cannot be scored (the fair-CRPS pair term is 0/0 and the ddof=1 spread
    is undefined), so the single member is duplicated.  For two identical
    members the fair estimator's pair term is exactly 0 and CRPS reduces to the
    MAE, which is the correct CRPS of a deterministic forecast.
    """
    ens = np.repeat(pred[None], 2, axis=0)
    return ensemble_metrics(ens, true, list(field_names))


def summarize(per_snap: List[Dict]) -> Dict:
    keys = list(per_snap[0]["aggregate"].keys())
    fields = list(per_snap[0]["per_field"].keys())
    return {
        "aggregate": {k: float(np.mean([s["aggregate"][k] for s in per_snap]))
                      for k in keys},
        "per_field": {f: {k: float(np.mean([s["per_field"][f][k] for s in per_snap]))
                          for k in keys} for f in fields},
        # kept so the sweep can be re-averaged over any snapshot subset.
        "per_snapshot": [
            {"snapshot": i,
             "rel_l2_mean": float(s["aggregate"]["rel_l2_mean"]),
             "crps": float(s["aggregate"]["crps"]),
             "per_field": {f: {"rel_l2_mean": float(s["per_field"][f]["rel_l2_mean"]),
                               "crps": float(s["per_field"][f]["crps"])}
                           for f in fields}}
            for i, s in enumerate(per_snap)],
        "n_snapshots": len(per_snap),
    }


# ---------------------------------------------------------------------------
# data loading -- canonical 2D H5 layout
# ---------------------------------------------------------------------------

def read_layout(h5_path: str):
    """coords (N,2) float64 from column 0,1 of coordinates (N,1,1,3); (T, F)."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        craw3 = f["coordinates"][:, 0, 0, :].astype(np.float64)   # (N, 3)
        _, T, N, _, _, F = f["fields"].shape
    if craw3.shape[0] != N:
        raise SystemExit(f"coordinates N={craw3.shape[0]} != fields N={N}")
    z = craw3[:, 2]
    if z.size and (z.max() - z.min()) > 1e-9 * max(1.0, np.abs(z).max()):
        print(f"[warn] z column is not constant (range {z.min()}..{z.max()}); "
              "ignoring it as documented for the canonical 2D layout.", flush=True)
    return np.ascontiguousarray(craw3[:, :2]), int(T), int(F)


def compute_train_stats(h5_path: str, train_indices: np.ndarray, n_fields: int,
                        chunk: int = 64):
    """Per-field mean/std over train frames, float64 accumulation, population
    variance clamped at 1e-12 -- the same formula as
    helpers.TurbulentCombustionH5Dataset._load_or_compute_stats, so the
    z-scoring convention is identical to the model's."""
    import h5py
    total_sum = np.zeros(n_fields, dtype=np.float64)
    total_sq = np.zeros(n_fields, dtype=np.float64)
    total_count = 0
    with h5py.File(h5_path, "r") as f:
        for start in range(0, len(train_indices), chunk):
            idx = train_indices[start:start + chunk]
            arr = f["fields"][0, idx, :, 0, 0, :].astype(np.float32)  # [Tc, N, C]
            total_sum += arr.sum(axis=(0, 1), dtype=np.float64)
            total_sq += (arr.astype(np.float64) ** 2).sum(axis=(0, 1))
            total_count += arr.shape[0] * arr.shape[1]
    mean = (total_sum / total_count).astype(np.float32)
    var = np.clip(total_sq / total_count - mean.astype(np.float64) ** 2,
                  1e-12, None).astype(np.float32)
    return mean, np.sqrt(var)


def load_split(h5_path: str, indices, n_points: int, n_fields: int,
               mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """[n, N, C] z-scored with train stats."""
    import h5py
    out = np.empty((len(indices), n_points, n_fields), dtype=np.float32)
    with h5py.File(h5_path, "r") as f:
        for i, t in enumerate(indices):
            out[i] = (f["fields"][0, int(t), :, 0, 0, :].astype(np.float32) - mean) / std
    return out


def load_manifest(h5_path: str) -> Optional[Dict]:
    d = Path(h5_path).parent
    stem = Path(h5_path).stem
    for cand in (d / f"{stem}_manifest.json", d / "kolmogorov2d_manifest.json"):
        if cand.exists():
            try:
                m = json.load(open(cand))
                print(f"[manifest] loaded {cand}", flush=True)
                return m
            except Exception as e:  # unreadable sidecar must not kill the run
                print(f"[manifest] failed to parse {cand}: {e}", flush=True)
    return None


def manifest_train_frames(manifest: Optional[Dict]) -> Optional[int]:
    if not manifest:
        return None
    for key in ("train_frames", "n_train_frames", "train_end",
                "train_frame_end", "n_train"):
        if key in manifest:
            return int(manifest[key])
    return None


def build_coords_box(craw: np.ndarray, periodic: bool):
    """Rescale the grid to the unit box, exactly as the 3D script does.

    3D: side = round(N^(1/3)); dx = (max-lo)/(side-1); box = side*dx;
        coords_box = (craw-lo)/box; boxsize = 1.0 if periodic.
    Here side = round(sqrt(N)).  For a [0,2pi)^2 endpoint-free grid this gives
    box = 2pi per axis, so wrapping the unit box by 1 wraps dx,dy by 2pi.  A
    non-square point set (e.g. an unstructured cylinder mesh) cannot define a
    period, so --periodic is refused for it and the raw coordinates are used
    unscaled (NN/IDW are invariant to the uniform rescale anyway).
    """
    N = craw.shape[0]
    side = int(round(np.sqrt(N)))
    lo = craw.min(0)
    if side * side == N:
        dx = (craw.max(0) - lo) / (side - 1)
        box = side * dx
        if periodic and abs(box[0] - box[1]) > 1e-6 * max(box):
            raise SystemExit(f"[grid] periodic wrap needs equal per-axis periods, "
                             f"got box={box}")
        scale = float(box.max())
        coords_box = np.ascontiguousarray(((craw - lo) / scale).astype(np.float64))
        # guard against points landing exactly on 1.0 (cKDTree requires < boxsize)
        if periodic:
            coords_box = np.mod(coords_box, 1.0)
        print(f"[grid] side={side} dx={dx} box={box} periodic={periodic}", flush=True)
        return coords_box, (1.0 if periodic else None)
    if periodic:
        raise SystemExit(f"[grid] N={N} is not a square grid; cannot infer a "
                         "periodic box. Rerun with --no-periodic.")
    print(f"[grid] N={N} not a square grid; using raw coordinates, non-periodic",
          flush=True)
    return np.ascontiguousarray(craw.astype(np.float64)), None


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def self_test() -> int:
    """Synthetic in-memory check of every estimator path.  CPU, seconds."""
    print("[self-test] building 64x64 synthetic dataset "
          "(2 fields, 20 train + 4 test frames)", flush=True)
    rng = np.random.default_rng(0)
    side, n_ch, n_train, n_test = 64, 2, 20, 4
    N = side * side
    xs = np.linspace(0.0, 2.0 * np.pi, side, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    craw = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N, 2) on [0,2pi)^2

    def smooth_frame():
        f = np.zeros((N, n_ch), dtype=np.float64)
        for c in range(n_ch):
            for _ in range(4):
                kx, ky = rng.integers(1, 4, size=2)
                phx, phy = rng.uniform(0, 2 * np.pi, size=2)
                f[:, c] += rng.normal() * np.sin(kx * craw[:, 0] + phx) \
                                        * np.cos(ky * craw[:, 1] + phy)
        return f.astype(np.float32)

    frames = np.stack([smooth_frame() for _ in range(n_train + n_test)])
    mean = frames[:n_train].mean(axis=(0, 1))
    std = frames[:n_train].std(axis=(0, 1))
    Ztr = (frames[:n_train] - mean) / std
    Zte = (frames[n_train:] - mean) / std

    coords_box, _ = build_coords_box(craw, periodic=False)
    coords_box_p, boxsize_p = build_coords_box(craw, periodic=True)
    assert boxsize_p == 1.0

    # (a) sensors == ALL points: kdtree and idw must be exact on observed fields
    sensors_full = {c: (np.arange(N, dtype=np.int64), Zte[0][:, c].copy())
                    for c in range(n_ch)}
    for mode, name in (("nn", "kdtree"), ("idw", "idw")):
        pred = kd_predict(coords_box, sensors_full, n_ch, mode, 8, None)
        errs = [_rel_l2(pred[:, c], Zte[0][:, c]) for c in range(n_ch)]
        print(f"[self-test] (a) {name} all-points rel-L2 per field: "
              f"{['%.2e' % e for e in errs]}", flush=True)
        assert all(e < 1e-6 for e in errs), f"{name} not exact with full sensors"

    # (a') unobserved-channel convention: field 1 unobserved -> exactly 0
    sensors_f0 = {0: (np.arange(N, dtype=np.int64), Zte[0][:, 0].copy())}
    pred = kd_predict(coords_box, sensors_f0, n_ch, "nn", 8, None)
    assert np.all(pred[:, 1] == 0.0), "unobserved channel must be train mean (0)"
    print("[self-test] (a') unobserved channel predicted as 0 (train mean): ok",
          flush=True)

    # (b) gappy POD, rank >= n_train (capped to n_train-1): a train frame lies
    # in span(mu, modes), so a fully observed train frame reconstructs ~exactly
    Xtr = Ztr.reshape(n_train, -1)
    mu = Xtr.mean(axis=0)
    pod = GappyPOD((Xtr - mu).astype(np.float32), mu.astype(np.float32),
                   rank=n_train)
    assert pod.r == n_train - 1
    cols = np.arange(N * n_ch, dtype=np.int64)
    rec = pod.reconstruct(cols, Xtr[3].astype(np.float64))
    e_pod = _rel_l2(rec, Xtr[3])
    print(f"[self-test] (b) gappy POD rank={pod.r} train-frame rel-L2: "
          f"{e_pod:.2e}", flush=True)
    assert e_pod < 1e-3, "gappy POD failed near-exact train reconstruction"

    # (c) periodic vs non-periodic must change neighbour sets near the boundary
    from scipy.spatial import cKDTree
    def _nearest_grid(p):
        return int(np.argmin(((coords_box - p) ** 2).sum(1)))
    s_near_far_edge = _nearest_grid(np.array([0.98, 0.5]))
    s_interior = _nearest_grid(np.array([0.30, 0.5]))
    q = coords_box[_nearest_grid(np.array([0.0, 0.5]))][None, :]
    pts = coords_box[[s_near_far_edge, s_interior]]
    _, nn_np = cKDTree(pts, boxsize=None).query(q, k=1)
    _, nn_p = cKDTree(np.mod(pts, 1.0), boxsize=1.0).query(np.mod(q, 1.0), k=1)
    print(f"[self-test] (c) boundary query NN: non-periodic={int(nn_np[0])} "
          f"(interior), periodic={int(nn_p[0])} (across the wrap)", flush=True)
    assert int(nn_np[0]) == 1 and int(nn_p[0]) == 0, \
        "periodic flag did not change the neighbour set"

    # (d) end-to-end sparse path via the canonical draw (CPU, non-canonical
    # device -- fine for a self test): draw, interpolate, gappy, score
    coords_t = torch.from_numpy(coords_box.astype(np.float32))
    fields_t = torch.from_numpy(Zte[0])
    sens = draw_sensors(coords_t, fields_t, snap=0, n_obs=400, seed=0,
                        device="cpu", cond_fields=[0])
    idx, val = sens[0]
    assert idx.size == 400 and idx.min() >= 0 and idx.max() < N
    assert np.allclose(val, Zte[0][idx, 0])
    pred_nn = kd_predict(coords_box, sens, n_ch, "nn", 8, None)
    pred_idw = kd_predict(coords_box, sens, n_ch, "idw", 8, None)
    e_nn = _rel_l2(pred_nn[:, 0], Zte[0][:, 0])
    e_idw = _rel_l2(pred_idw[:, 0], Zte[0][:, 0])
    cols_s, vals_s = obs_columns(sens, n_ch)
    pred_pod = pod.reconstruct(cols_s, vals_s).reshape(N, n_ch)
    e_gp = _rel_l2(pred_pod[:, 0], Zte[0][:, 0])
    print(f"[self-test] (d) sparse 400/{N} sensors on field 0: "
          f"kdtree {e_nn:.3f}, idw {e_idw:.3f}, gappy_pod {e_gp:.3f}", flush=True)
    assert e_nn < 0.9 and e_idw < 0.9, "sparse interpolation worse than constant"
    assert np.all(pred_nn[:, 1] == 0.0) and np.all(pred_idw[:, 1] == 0.0)

    # (e) metrics schema: deterministic 2-member ensemble -> CRPS == MAE
    m = score(pred_nn, Zte[0], ["f0", "f1"])
    mae0 = float(np.abs(pred_nn[:, 0] - Zte[0][:, 0]).mean())
    d_crps = abs(m["per_field"]["f0"]["crps"] - mae0)
    print(f"[self-test] (e) score(): crps={m['per_field']['f0']['crps']:.6f} "
          f"mae={mae0:.6f} |diff|={d_crps:.2e}, spread="
          f"{m['per_field']['f0']['spread']:.1e}", flush=True)
    assert d_crps < 1e-6 * max(1.0, mae0), "deterministic CRPS != MAE"
    assert m["per_field"]["f0"]["spread"] == 0.0

    print("[self-test] ALL PASS", flush=True)
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default=DATA, help="canonical 2D H5 file")
    p.add_argument("--train-frames", type=int, default=None,
                   help="frames [0,N) are the train split, the rest test; "
                        "default: sidecar manifest, else "
                        f"{TRAIN_FRAMES_DEFAULT}")
    p.add_argument("--fields", nargs="+", default=list(FIELD_NAMES),
                   help="field names, in H5 channel order")
    p.add_argument("--cond-fields", type=int, nargs="+", default=COND_FIELDS,
                   help="indices of OBSERVED fields; sensors are drawn only "
                        "from these")
    p.add_argument("--methods", nargs="+",
                   default=["constant", "kdtree", "idw", "gappy_pod"])
    p.add_argument("--n-sensors", type=int, nargs="+", default=None,
                   help="Sensors per OBSERVED field. Several values = sweep. "
                        "Default: 1%% of N (rounded).")
    p.add_argument("--n-snapshots", type=int, default=None,
                   help="test frames scored, in split order; default: all")
    p.add_argument("--seed", type=int, default=0, help="ensemble_eval --seed")
    p.add_argument("--sensor-device", default="cuda:0",
                   help="Device for the sensor draw; must match the model "
                        "eval's --device for bit-identical index sets.")
    p.add_argument("--idw-k", type=int, default=8)
    p.add_argument("--pod-rank", type=int, default=80,
                   help="gappy POD rank (fixed; capped at n_train-1)")
    p.add_argument("--no-periodic", action="store_true",
                   help="kept for symmetry with the 3D CLI; non-periodic is "
                        "the DEFAULT (a past bug came from wrongly-periodic "
                        "floors; audit 2026-08-28 s26)")
    p.add_argument("--periodic", action="store_true",
                   help="wrap dx,dy by the domain period (Kolmogorov "
                        "[0,2pi)^2 is genuinely periodic)")
    p.add_argument("--verify-percentile", action="store_true",
                   help="Score one snapshot with and without the np.percentile "
                        "fast path and assert the metric dicts are identical.")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-dir",
                   default="../Save_TrainedModel/Kolmogorov2D/baseline_classical")
    p.add_argument("--self-test", action="store_true",
                   help="run the synthetic in-memory checks and exit")
    args = p.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if args.periodic and args.no_periodic:
        raise SystemExit("--periodic and --no-periodic are mutually exclusive")

    require_compute_node()
    dev = args.sensor_device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        msg = ("CUDA unavailable: sensor draws would fall back to CPU and will "
               "NOT be bit-identical to a model eval's CUDA draws.")
        if os.environ.get("ALLOW_LOGIN_EVAL") == "1":
            print("[warn] " + msg)
            dev = "cpu"
        else:
            raise SystemExit("[nodecheck] " + msg +
                             " Set ALLOW_LOGIN_EVAL=1 to override.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.h5).exists():
        raise SystemExit(f"[data] H5 not found: {args.h5} (still being written?)")

    craw, T, C = read_layout(args.h5)
    N = craw.shape[0]
    field_names = list(args.fields)
    if len(field_names) != C:
        raise SystemExit(f"--fields has {len(field_names)} names but the H5 has "
                         f"{C} field channels")
    cond_fields = [int(c) for c in args.cond_fields]
    if any(c < 0 or c >= C for c in cond_fields):
        raise SystemExit(f"--cond-fields {cond_fields} out of range for C={C}")

    manifest = load_manifest(args.h5)
    train_frames = args.train_frames
    mf_train = manifest_train_frames(manifest)
    if train_frames is None:
        train_frames = mf_train if mf_train is not None else TRAIN_FRAMES_DEFAULT
        print(f"[data] train-frames not given; using "
              f"{'manifest' if mf_train is not None else 'default'} value "
              f"{train_frames}", flush=True)
    elif mf_train is not None and mf_train != train_frames:
        print(f"[warn] --train-frames {train_frames} != manifest {mf_train}; "
              "using the CLI value.", flush=True)
    if not (0 < train_frames < T):
        raise SystemExit(f"train_frames={train_frames} out of range for T={T}")
    train_idx = np.arange(train_frames)
    test_idx = np.arange(train_frames, T)

    mean, std = compute_train_stats(args.h5, train_idx, C)
    print(f"[data] train {len(train_idx)} frames (t={train_idx[0]}..{train_idx[-1]}), "
          f"test {len(test_idx)} frames (t={test_idx[0]}..{test_idx[-1]}), "
          f"N={N} C={C}", flush=True)
    print(f"[data] train-split mean={mean} std={std}", flush=True)
    if manifest:
        for mk, sk, v in (("train_mean", "mean", mean), ("train_std", "std", std)):
            mv = manifest.get(mk, manifest.get(sk))
            if mv is not None:
                print(f"[manifest] {mk} (sidecar) = {mv}  vs computed {v.tolist()}",
                      flush=True)

    coords_box, boxsize = build_coords_box(craw, periodic=args.periodic)

    n_snap = len(test_idx) if args.n_snapshots is None \
        else min(args.n_snapshots, len(test_idx))
    snaps = list(range(n_snap))
    if n_snap < len(test_idx):
        print(f"[warn] n_snapshots={n_snap} < len(test)={len(test_idx)}: scoring "
              "only the first n test frames -- numbers are not comparable to "
              "full-split runs.", flush=True)

    n_sensors = args.n_sensors or [max(1, int(round(0.01 * N)))]
    if args.n_sensors is None:
        print(f"[sensors] --n-sensors not given; defaulting to 1% of N = "
              f"{n_sensors[0]} per observed field", flush=True)

    coords_t = torch.from_numpy(coords_box.astype(np.float32))

    # --- load the held-out frames once ------------------------------------
    t = Timer()
    with t:
        Yval = load_split(args.h5, test_idx[:n_snap], N, C, mean, std)
    print(f"[data] test frames loaded in {t.wall:.1f}s, "
          f"peak RSS {peak_rss_gb():.1f} GB", flush=True)

    percentile_check = None
    if args.verify_percentile:
        probe = np.zeros((N, C), dtype=np.float32)
        probe[:, 0] = np.linspace(-2, 2, N, dtype=np.float32)
        disable_percentile_fastpath()
        t0 = time.perf_counter()
        ref = score(probe, Yval[0], field_names)
        t_ref = time.perf_counter() - t0
        enable_percentile_fastpath()
        t0 = time.perf_counter()
        fast = score(probe, Yval[0], field_names)
        t_fast = time.perf_counter() - t0
        same = json.dumps(ref, sort_keys=True) == json.dumps(fast, sort_keys=True)
        percentile_check = {"identical": bool(same),
                            "numpy_percentile_s": t_ref, "fastpath_s": t_fast}
        print(f"[verify] np.percentile fast path identical={same} "
              f"({t_ref:.1f}s -> {t_fast:.1f}s per snapshot)", flush=True)
        assert same, "percentile fast path changed the metrics"
    enable_percentile_fastpath()

    timing: Dict[str, Dict] = {}
    results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # 1. constant predictors  (sensor-count independent)
    # ------------------------------------------------------------------
    if "constant" in args.methods:
        # train-split mean == 0 in z-score units, by construction of the stats.
        tf = Timer()
        with tf:
            const_train = np.zeros((N, C), dtype=np.float32)
        ti = Timer()
        per = []
        for i in snaps:
            with ti:
                pred = const_train
            per.append(score(pred, Yval[i], field_names))
        results["constant_train_mean"] = summarize(per)
        results["constant_train_mean"]["note"] = (
            "Predicts the TRAIN-split per-channel mean, which is exactly 0 in "
            "z-score units; relative L2 is therefore identically 1.0 by "
            "construction and CRPS is E|y| on the held-out frames. This is the "
            "no-information floor.")
        # contrast only: uses held-out information, NOT a legitimate baseline
        mu_val = Yval.mean(axis=(0, 1)).astype(np.float32)
        per = []
        for i in snaps:
            per.append(score(np.broadcast_to(mu_val, (N, C)).copy(), Yval[i],
                             field_names))
        results["constant_test_mean_CONTRAST"] = summarize(per)
        results["constant_test_mean_CONTRAST"]["note"] = (
            "NOT a legitimate baseline: predicts the TEST-split mean, "
            "which uses held-out information. Reported only to show how little "
            "it buys over the train-split mean (test mean in z-units = "
            f"{mu_val.tolist()}).")
        timing["constant"] = {"fit_wall_s": tf.wall,
                              "infer_wall_s_per_field": ti.wall / n_snap,
                              "infer_cpu_s_per_field": ti.cpu / n_snap,
                              "peak_rss_gb": peak_rss_gb(), "device": "cpu"}
        print(f"[constant] train-mean aggregate relL2="
              f"{results['constant_train_mean']['aggregate']['rel_l2_mean']:.4f} "
              f"CRPS={results['constant_train_mean']['aggregate']['crps']:.4f}",
              flush=True)
        print("[constant] per-field CRPS: " + " ".join(
            f"{f}:{results['constant_train_mean']['per_field'][f]['crps']:.4f}"
            for f in field_names), flush=True)

    # ------------------------------------------------------------------
    # 2. gappy POD (basis fitted once, then swept over sensor counts)
    # ------------------------------------------------------------------
    pod = None
    if "gappy_pod" in args.methods:
        tfit = Timer()
        with tfit:
            Xtr = load_split(args.h5, train_idx, N, C, mean, std)
            Xtr = Xtr.reshape(len(train_idx), -1)
            mu = Xtr.mean(axis=0)
            Xtr -= mu
        print(f"[pod] train matrix {Xtr.shape} loaded/centred in {tfit.wall:.1f}s, "
              f"peak RSS {peak_rss_gb():.1f} GB", flush=True)
        with tfit:
            pod = GappyPOD(Xtr, mu, rank=args.pod_rank)
        print(f"[pod] rank {pod.r} (cum-energy {pod.energy[pod.r - 1]:.4f}); "
              f"fit wall {tfit.wall:.1f}s cpu {tfit.cpu:.1f}s "
              f"peak RSS {peak_rss_gb():.1f} GB", flush=True)
        timing["gappy_pod_fit"] = {
            "fit_wall_s": tfit.wall, "fit_cpu_s": tfit.cpu,
            "peak_rss_gb": peak_rss_gb(), "device": "cpu",
            "selected_rank": pod.r,
            "cum_energy_at_rank": float(pod.energy[pod.r - 1]),
            "cum_energy_curve": [float(v) for v in pod.energy[:max(200, pod.r)]],
        }

    # ------------------------------------------------------------------
    # 3. sensor-dependent methods
    # ------------------------------------------------------------------
    for n_obs in n_sensors:
        frac = 100.0 * n_obs / N
        sens = {}
        tsens = Timer()
        with tsens:
            for i in snaps:
                sens[i] = draw_sensors(coords_t, torch.from_numpy(Yval[i]),
                                       snap=i, n_obs=n_obs, seed=args.seed,
                                       device=dev, cond_fields=cond_fields)
        print(f"\n===== n_sensors={n_obs} ({frac:.3f}% of {N}) : sensor draw "
              f"{tsens.wall:.1f}s on {dev} =====", flush=True)
        # fingerprint of the first snapshot's draw, so a model eval on the
        # same protocol can cross-check its own sensor sets against ours
        _n = sum(int(v[0].size) for v in sens[0].values())
        _s = sum(int(v[0].sum()) for v in sens[0].values())
        print(f"[seedcheck] snap=0 sensors={_n} idx_sum={_s} device={dev}",
              flush=True)

        for name, mode in (("kdtree", "nn"), ("idw", "idw")):
            if name not in args.methods:
                continue
            tt = Timer()
            per = []
            for i in snaps:
                with tt:
                    pred = kd_predict(coords_box, sens[i], C, mode, args.idw_k,
                                      boxsize)
                per.append(score(pred, Yval[i], field_names))
            key = f"{name}_n{n_obs}"
            results[key] = summarize(per)
            results[key]["n_obs_per_observed_field"] = n_obs
            results[key]["sensor_fraction_pct"] = frac
            results[key]["note"] = (
                "Unobserved channels are predicted as the train-split "
                "mean (0 in z-score units) -- no sensor constrains them.")
            timing[key] = {"fit_wall_s": 0.0,
                           "infer_wall_s_per_field": tt.wall / n_snap,
                           "infer_cpu_s_per_field": tt.cpu / n_snap,
                           "peak_rss_gb": peak_rss_gb(), "device": "cpu"}
            a = results[key]["aggregate"]
            pf = results[key]["per_field"]
            print(f"[{name} n={n_obs}] agg relL2={a['rel_l2_mean']:.4f} "
                  f"CRPS={a['crps']:.4f} | " +
                  " ".join(f"{f}:{pf[f]['rel_l2_mean']:.4f}/{pf[f]['crps']:.4f}"
                           for f in field_names) +
                  f" | {tt.wall / n_snap:.2f} s/field", flush=True)

        if pod is not None:
            tt = Timer()
            per = []
            for i in snaps:
                cols, vals = obs_columns(sens[i], C)
                with tt:
                    pred = pod.reconstruct(cols, vals).reshape(N, C)
                per.append(score(pred, Yval[i], field_names))
            key = f"gappy_pod_n{n_obs}"
            results[key] = summarize(per)
            results[key]["n_obs_per_observed_field"] = n_obs
            results[key]["sensor_fraction_pct"] = frac
            results[key]["rank"] = pod.r
            timing[key] = {"infer_wall_s_per_field": tt.wall / n_snap,
                           "infer_cpu_s_per_field": tt.cpu / n_snap,
                           "peak_rss_gb": peak_rss_gb(), "device": "cpu"}
            a = results[key]["aggregate"]
            pf = results[key]["per_field"]
            print(f"[gappy_pod n={n_obs} r={pod.r}] agg relL2={a['rel_l2_mean']:.4f} "
                  f"CRPS={a['crps']:.4f} | " +
                  " ".join(f"{f}:{pf[f]['rel_l2_mean']:.4f}/{pf[f]['crps']:.4f}"
                           for f in field_names) +
                  f" | {tt.wall / n_snap:.2f} s/field", flush=True)
        del sens

    payload = {
        "protocol": {
            "data": args.h5,
            "split": f"frames [0,{train_frames}) train, "
                     f"[{train_frames},{T}) test",
            "train_frames": int(train_frames),
            "field_names": field_names,
            "cond_fields": cond_fields, "n_points": N,
            "n_snapshots": n_snap, "seed": args.seed,
            "sensor_draw": ("helpers.build_sparse_condition under "
                            "torch.manual_seed(seed*777+snap) on " + dev +
                            " -- snap = 0-based position in the test split, "
                            "identical to the ensemble_eval convention"),
            "standardization": "per-field z-score with TRAIN-split stats "
                               f"(mean={mean.tolist()}, "
                               f"std={std.tolist()})",
            "crps": "deterministic estimator scored as a 2-member identical "
                    "ensemble; fair CRPS then equals the MAE exactly",
            "periodic_kdtree": boxsize is not None,
            "idw_k": args.idw_k,
            "pod_rank": args.pod_rank,
            "percentile_fastpath_check": percentile_check,
        },
        "timing_and_memory": timing,
        "results": results,
    }
    tag = args.tag or ("sweep" if len(n_sensors) > 1 else f"n{n_sensors[0]}")
    out = out_dir / f"classical_baselines_{tag}_2d.json"
    json.dump(payload, open(out, "w"), indent=1)
    print(f"\n[out] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
