"""Gappy-POD baseline for SHIFT-WING: surface observations -> volume field.

Classical lower bound (Everson & Sirovich 1995; Manohar et al. 2018). Unlike
isotropic turbulence, where POD has no usable low-rank structure, transonic
surface and volume fields over a *design family* genuinely are low-rank, so
this is a fair baseline here rather than a strawman.

Method. Stack each training case into one vector [surface ; volume] and take a
joint POD basis over the training cases. At test time only the surface rows are
observed, so the modal coefficients are recovered by least squares on the
observed surface taps alone, then the volume rows of the basis reconstruct the
interior:

    a* = argmin_a || Phi_s[obs] a - y_obs ||^2 ,   x_vol = Phi_v a

Because the estimator is linear and deterministic, its CRPS equals its MAE,
which is how it enters the probabilistic table alongside the generative models.

This script is intentionally standalone and CPU/numpy-only: it needs no GPU and
no training, so it can be run at any time as a reference point.

Usage:
    python baseline_gappy_pod_wing.py --processed-root <dir> --rank 64 \
        --n-taps 512 --out gappy_pod_wing.json
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def load_case(path: Path, sub_vol: np.ndarray, sub_surf: np.ndarray):
    with h5py.File(path, "r") as f:
        vol = f["volume/fields"][:][sub_vol]         # [n_v, 4]
        surf = f["surface/values"][:][sub_surf]      # [n_s, 4]
    return vol, surf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", required=True)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--n-taps", type=int, default=512,
                   help="Surface sensors visible at test time.")
    p.add_argument("--n-vol-sub", type=int, default=40000,
                   help="Volume points retained (subsampled to bound memory).")
    p.add_argument("--n-surf-sub", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    root = Path(args.processed_root)
    split = json.load(open(root / "stats.json"))["split"]
    rng = np.random.default_rng(args.seed)

    with h5py.File(root / split["train"][0], "r") as f:
        n_v_all = f["volume/fields"].shape[0]
        n_s_all = f["surface/values"].shape[0]
        n_f = f["volume/fields"].shape[1]
    sub_vol = np.sort(rng.choice(n_v_all, min(args.n_vol_sub, n_v_all), replace=False))
    sub_surf = np.sort(rng.choice(n_s_all, min(args.n_surf_sub, n_s_all), replace=False))

    def stack(names):
        V, S = [], []
        for n in names:
            v, s = load_case(root / n, sub_vol, sub_surf)
            V.append(v.ravel()); S.append(s.ravel())
        return np.stack(V), np.stack(S)          # [n_case, n_v*4], [n_case, n_s*4]

    print(f"loading {len(split['train'])} train / {len(split['val'])} val cases...")
    Vtr, Str = stack(split["train"])
    Vte, Ste = stack(split["val"])

    # Joint basis on train cases only; centre with the training mean.
    Xtr = np.concatenate([Str, Vtr], axis=1)
    mu = Xtr.mean(axis=0, keepdims=True)
    U, sing, Wt = np.linalg.svd(Xtr - mu, full_matrices=False)
    r = min(args.rank, Wt.shape[0])
    Phi = Wt[:r]                                  # [r, n_s*4 + n_v*4]
    n_s_cols = Str.shape[1]
    Phi_s, Phi_v = Phi[:, :n_s_cols], Phi[:, n_s_cols:]
    print(f"POD rank {r}; energy captured = "
          f"{(sing[:r] ** 2).sum() / (sing ** 2).sum():.4f}")

    # Observed taps: a fixed random subset of surface points, all 4 channels.
    taps = np.sort(rng.choice(len(sub_surf), min(args.n_taps, len(sub_surf)),
                              replace=False))
    obs_cols = np.concatenate([taps * n_f + c for c in range(n_f)])
    A = Phi_s[:, obs_cols].T                      # [n_obs, r]

    rel, mae = [], []
    for i in range(Vte.shape[0]):
        y = (Ste[i] - mu[0, :n_s_cols])[obs_cols]
        a, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = mu[0, n_s_cols:] + a @ Phi_v
        true = Vte[i]
        rel.append(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))
        mae.append(np.abs(pred - true).mean())

    res = {
        "method": "gappy_POD", "rank": r, "n_taps": int(len(taps)),
        "n_vol_points": int(len(sub_vol)), "n_surf_points": int(len(sub_surf)),
        "rel_l2_mean": float(np.mean(rel)),
        # A deterministic estimator's CRPS is exactly its MAE.
        "crps_equals_mae": float(np.mean(mae)),
        "n_val_cases": int(Vte.shape[0]),
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
