"""Training-free classical baselines for SHIFT-WING surface -> volume reconstruction.

Three estimators, all deterministic and CPU-only, scored with exactly the same
estimator (``ensemble_eval.ensemble_metrics``) and on exactly the same sensor
draw and the same validation cases as the trained DMF-Gen model:

  1. ``gappy_pod``  joint POD over the stacked [surface pool ; parameters ;
     volume] state across the 600 TRAIN cases; modal coefficients recovered by
     least squares from the observed surface taps / shear sensors and the two
     exact parameter tokens; the volume rows of the basis then reconstruct the
     interior.  Rank is swept.
  2. ``idw``        inverse-distance weighting (k = 8) from the surface sensors
     to every volume node, in the normalized coordinate box.  Two variants,
     both reported: ``idw_k8`` uses only the 512 Cp taps (the one observable
     that is physically the same field as a generated channel, converted
     between the surface and volume z-scoring conventions) and leaves the three
     velocity channels at the train mean; ``idw_k8_shearproxy`` additionally
     pushes the wall-shear components onto (Ux, Uy, Uz) in z-score units.  The
     sensors live on a 2-D manifold and the targets fill a 3-D volume, so this
     is expected to be poor; it is reported as found, untuned.
  3. ``train_mean``  the constant train-mean predictor, i.e. 0 in the dataset's
     volume z-score units.  This is the ~1.0 relative-L2 floor.

FAIRNESS.  The observation tuple is produced by calling
``evaluate_wing.build_surface_obs`` verbatim with the canonical arguments
(n_taps = 512, n_shear = 128, seed = seed*100 + case_index, seed = 0) on the
first ``--n-cases`` cases of the val split -- the same loop bound
``evaluate_wing.main`` uses.  The pool indices are re-derived by replaying the
same ``torch.Generator`` sequence and are asserted to reproduce the coordinates
and values returned by ``build_surface_obs`` bit for bit; their checksum is
printed per case as a fingerprint.

MESH CORRESPONDENCE.  Unlike the 2-D cases, SHIFT-WING has no node
correspondence across cases: the geometry varies (the span differs by >10%
between cases) and each case's stored 400k volume / 65k surface points are an
independent random subset of its own mesh.  A snapshot POD therefore needs a
common discretisation.  Resampling everything onto one shared reference cloud
costs a large, purely numerical error floor (measured: rel-L2 ~0.9 at 20k
reference points, because the truth itself has to be round-tripped through the
reference).  Instead the basis is rebuilt *in each val case's own
discretisation*: the 600 train cases are nearest-neighbour resampled onto that
val case's 400k volume nodes and 65k surface pool points, the POD is taken
there, and the observation columns are then the drawn pool indices exactly.
Only train fields are ever resampled -- the val truth is never touched and the
metrics are computed on exactly the nodes the model is scored on, so the
comparison carries no interpolation floor at all.  The price is that the basis
is rebuilt per case, which is charged to the reported cost.

As a diagnostic (not a baseline row) the script also reports the *oracle*
rank-r projection error: the best reconstruction of the val truth achievable in
the train basis if the whole volume field were observed.  It separates "the
train basis cannot represent this case" from "the 896 surface sensors cannot
identify the coefficients".

Usage (CPU only; see baseline_classical_wing.sh for the sbatch wrapper):
    python baseline_classical_wing.py \
        --processed-root /projects/.../shift_wing/processed_v3 \
        --out-dir ../Save_TrainedModel/wing/baseline_classical
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch

from dataset_shiftwing import ShiftWingDataset
from evaluate_wing import build_surface_obs          # canonical draw, verbatim
from ensemble_eval import ensemble_metrics
from baseline_classical_2d import GappyPOD, peak_rss_gb, Timer

FIELD_NAMES = ["Ux", "Uy", "Uz", "Cp"]
N_CH = 4
# Channel c of the surface pool carries observation field id SURF_IDS[c].
SURF_CP_CHANNEL = 0        # pool channel 0 is Cp == volume channel 3
VOL_CP_CHANNEL = 3


# ---------------------------------------------------------------------------
# canonical observations
# ---------------------------------------------------------------------------

def canonical_obs(item: Dict[str, torch.Tensor], n_taps: int, n_shear: int,
                  seed: int):
    """``build_surface_obs`` output plus the recovered pool indices.

    ``build_surface_obs`` does not return the indices it drew, so the same
    generator sequence is replayed here and the recovered indices are checked
    against the returned coordinates and values.  The check is an equality on
    the actual tensors, so a divergence in the draw cannot pass silently.
    """
    obs = build_surface_obs(item, n_taps, n_shear, "cpu", seed=seed)

    g = torch.Generator().manual_seed(seed)
    n_pool = item["obs_pool_coords"].shape[0]
    counts = [n_taps, n_shear, n_shear, n_shear]
    per_channel = []
    for c, m in enumerate(counts):
        if m <= 0:
            per_channel.append((c, torch.empty(0, dtype=torch.long)))
            continue
        per_channel.append((c, torch.randperm(n_pool, generator=g)[:m]))

    pc, pv = item["obs_pool_coords"], item["obs_pool_values"]
    coords = torch.cat([pc[i] for _, i in per_channel] + [item["param_coords"]], 0)
    values = torch.cat([pv[i, c:c + 1] for c, i in per_channel]
                       + [item["param_values"]], 0)
    assert torch.equal(coords, obs["coords"][0]), "sensor coord replay mismatch"
    assert torch.equal(values, obs["values"][0]), "sensor value replay mismatch"

    n_sensor = sum(counts)
    assert obs["coords"].shape[1] == n_sensor + item["param_coords"].shape[0]
    idx_all = torch.cat([i for _, i in per_channel])
    fp = {
        "n_tokens": int(obs["coords"].shape[1]),
        "n_sensors": int(n_sensor),
        "n_param_tokens": int(item["param_coords"].shape[0]),
        "pool_index_sum": int(idx_all.sum()),
        "pool_index_sum_per_channel": [int(i.sum()) for _, i in per_channel],
        "seed": int(seed),
    }
    return obs, per_channel, fp


# ---------------------------------------------------------------------------
# snapshot matrix: train cases resampled onto one target case's point cloud
# ---------------------------------------------------------------------------

_G: Dict = {}


def _init_worker(root, target_vol, target_surf, stats):
    _G["root"] = root
    _G["target_vol"], _G["target_surf"] = target_vol, target_surf
    _G["stats"] = stats


def _case_row(name: str) -> np.ndarray:
    """Resample one train case onto the target (val) case's own point cloud.

    Layout: [surface (N_SURF*4) | params (2) | volume (N_VOL*4)], z-scored with
    the dataset's own train statistics so the state is in the units the model is
    scored in.  Raw (unnormalized) coordinates are used for the neighbour search
    -- the global normalization is a per-axis affine map, which would change
    which node is nearest, and the raw metric is the physical one.
    """
    from scipy.spatial import cKDTree
    st = _G["stats"]
    with h5py.File(Path(_G["root"]) / name, "r") as f:
        vc = f["volume/coords"][:].astype(np.float64)
        vf = f["volume/fields"][:]
        sc = f["surface/coords"][:].astype(np.float64)
        sv = f["surface/values"][:]
        mach, alpha = float(f.attrs["mach"]), float(f.attrs["alpha"])

    vf = (vf - st["v_mean"]) / st["v_std"]
    sv = (sv - st["s_mean"]) / st["s_std"]
    _, iv = cKDTree(vc).query(_G["target_vol"], k=1, workers=1)
    _, isf = cKDTree(sc).query(_G["target_surf"], k=1, workers=1)
    params = np.array([(mach - st["mach_mean"]) / st["mach_std"],
                       (alpha - st["alpha_mean"]) / st["alpha_std"]],
                      dtype=np.float32)
    return np.concatenate([sv[isf].ravel(), params, vf[iv].ravel()]).astype(np.float32)


class BlockGramPOD(GappyPOD):
    """``GappyPOD`` with the snapshot Gram accumulated in float64 blocks.

    Identical maths to the parent; only the Gram assembly differs.  The state
    vector here is ~1.9M long, so a single float32 sgemm accumulates 1.9M terms
    per entry and the trailing eigenvalues -- the ones that matter at rank 160 --
    would be set by round-off rather than by the data.
    """

    def __init__(self, Xc: np.ndarray, mu: np.ndarray, rank: int,
                 block: int = 262_144):
        self.Xc, self.mu = Xc, mu
        n = Xc.shape[0]
        G = np.zeros((n, n), dtype=np.float64)
        for s in range(0, Xc.shape[1], block):
            B = np.ascontiguousarray(Xc[:, s:s + block])
            G += (B @ B.T).astype(np.float64)
        lam, U = np.linalg.eigh(G)
        order = np.argsort(lam)[::-1]
        self.lam, self.U = np.clip(lam[order], 1e-12, None), U[:, order]
        self.energy = np.cumsum(self.lam) / self.lam.sum()
        self.set_rank(rank)


def assemble_train_matrix(root: Path, train_names: Sequence[str], target_vol,
                          target_surf, st: Dict, workers: int) -> np.ndarray:
    """[n_train, N_SURF*4 + 2 + N_VOL*4] on the target case's discretisation."""
    from multiprocessing import Pool
    P = target_surf.shape[0] * N_CH + 2 + target_vol.shape[0] * N_CH
    X = np.empty((len(train_names), P), dtype=np.float32)
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init_worker,
              initargs=(str(root), target_vol, target_surf, st)) as pool:
        for j, row in enumerate(pool.imap(_case_row, train_names, chunksize=2)):
            X[j] = row
            if (j + 1) % 200 == 0:
                print(f"      resampled {j + 1}/{len(train_names)} train cases "
                      f"({time.perf_counter() - t0:.0f}s, RSS "
                      f"{peak_rss_gb():.1f} GB)", flush=True)
    return X


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

DETERMINISTIC_NULL = ("spread", "spread_error_ratio", "coverage_50", "coverage_90")


def score_deterministic(pred: np.ndarray, true: np.ndarray, check: Dict) -> Dict:
    """Deterministic prediction -> campaign-schema metrics.

    K = 1 cannot be scored (the fair-CRPS pair term is 0/0 and the ddof = 1
    spread is undefined), so the single member is tiled twice: for two
    identical members the fair pair term is exactly 0 and CRPS reduces to the
    MAE, which *is* the CRPS of a deterministic forecast.  The reduction is
    verified numerically here, per field.  Ensemble-derived entries
    (spread, spread/error, coverage, rank histogram) are meaningless for a
    deterministic estimator and are written as null rather than 0.
    """
    ens = np.repeat(pred[None].astype(np.float64), 2, axis=0)
    m = ensemble_metrics(ens, true.astype(np.float64), FIELD_NAMES)
    for j, f in enumerate(FIELD_NAMES):
        mae = float(np.abs(pred[:, j].astype(np.float64)
                           - true[:, j].astype(np.float64)).mean())
        d = abs(m["per_field"][f]["crps"] - mae)
        check.setdefault("max_abs_crps_minus_mae", 0.0)
        check["max_abs_crps_minus_mae"] = max(check["max_abs_crps_minus_mae"], d)
        assert d < 1e-12, f"CRPS != MAE for {f}: |diff| = {d:.3e}"
        for k in DETERMINISTIC_NULL:
            m["per_field"][f][k] = None
    for k in DETERMINISTIC_NULL:
        m["aggregate"][k] = None
    m["rank_hist"] = None
    return m


def summarize(cases: List[Dict]) -> Dict:
    keys = [k for k in cases[0]["aggregate"] if cases[0]["aggregate"][k] is not None]
    return {k: float(np.mean([c["aggregate"][k] for c in cases])) for k in keys}


def summarize_per_field(cases: List[Dict]) -> Dict:
    keys = [k for k in cases[0]["aggregate"] if cases[0]["aggregate"][k] is not None]
    return {f: {k: float(np.mean([c["per_field"][f][k] for c in cases]))
                for k in keys} for f in FIELD_NAMES}


def write_result(out_dir: Path, tag: str, method: str, config: Dict,
                 cases: List[Dict], cost: Dict, extra: Dict) -> Path:
    out = out_dir / f"wing_classical_{tag}.json"
    payload = {
        "method": method, "tag": tag, "config": config,
        "summary": summarize(cases),
        "summary_per_field": summarize_per_field(cases),
        "cases": cases, "cost": cost,
    }
    payload.update(extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    return out


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def idw_predict(sensor_xyz: np.ndarray, sensor_val: np.ndarray,
                query_xyz: np.ndarray, k: int) -> np.ndarray:
    """Inverse-distance weighted interpolation, k nearest sensors, Euclidean."""
    from scipy.spatial import cKDTree
    kk = min(k, sensor_xyz.shape[0])
    d, nn = cKDTree(sensor_xyz).query(query_xyz, k=kk, workers=-1)
    if kk == 1:
        d, nn = d[:, None], nn[:, None]
    exact = d[:, 0] < 1e-11
    w = 1.0 / np.maximum(d, 1e-12)
    out = (w * sensor_val[nn]).sum(1) / w.sum(1)
    out[exact] = sensor_val[nn[exact, 0]]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", default="/projects/ammoniacomb/"
                   "generative_reconstruction/shift_wing/processed_v3")
    p.add_argument("--out-dir", default="../Save_TrainedModel/wing/baseline_classical")
    p.add_argument("--n-cases", type=int, default=8)
    p.add_argument("--n-taps", type=int, default=512)
    p.add_argument("--n-shear", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ranks", type=int, nargs="+", default=[0, 10, 20, 40, 80, 160])
    p.add_argument("--idw-k", type=int, default=8)
    p.add_argument("--n-train", type=int, default=0, help="0 = all train cases")
    p.add_argument("--no-oracle", action="store_true",
                   help="Skip the oracle rank-r projection diagnostic.")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--methods", nargs="+",
                   default=["gappy_pod", "idw", "train_mean"])
    args = p.parse_args()

    root = Path(args.processed_root)
    out_dir = Path(args.out_dir)
    ds_val = ShiftWingDataset(root, split="val")
    stats = ds_val.stats
    train_names = list(stats["split"]["train"])
    if args.n_train:
        train_names = train_names[:args.n_train]
    st = {"v_mean": ds_val.v_mean, "v_std": ds_val.v_std,
          "s_mean": ds_val.s_mean, "s_std": ds_val.s_std,
          "mach_mean": stats["mach_mean"], "mach_std": stats["mach_std"],
          "alpha_mean": stats["alpha_mean"], "alpha_std": stats["alpha_std"]}
    n_cases = min(args.n_cases, len(ds_val))
    print(f"[wing-classical] {len(train_names)} train / {len(ds_val)} val cases; "
          f"scoring the first {n_cases} val cases", flush=True)

    # ---- canonical observations (identical draw to evaluate_wing) ----------
    val_items, val_obs, val_raw, fingerprints = [], [], [], []
    for i in range(n_cases):
        item = ds_val[i]
        obs, per_channel, fp = canonical_obs(item, args.n_taps, args.n_shear,
                                             seed=args.seed * 100 + i)
        fp["case"] = ds_val.files[i].name
        with h5py.File(ds_val.files[i], "r") as f:   # raw (physical) coordinates
            val_raw.append((f["volume/coords"][:].astype(np.float64),
                            f["surface/coords"][:].astype(np.float64)))
        val_items.append(item)
        val_obs.append((obs, per_channel))
        fingerprints.append(fp)
        print(f"  [sensors] {fp['case']}: seed={fp['seed']} tokens={fp['n_tokens']} "
              f"(sensors={fp['n_sensors']} + params={fp['n_param_tokens']}) "
              f"pool_index_sum={fp['pool_index_sum']} "
              f"per_channel={fp['pool_index_sum_per_channel']}", flush=True)

    checks: Dict = {}
    written: List[str] = []
    cfg_common = {"processed_root": str(root), "n_cases": n_cases,
                  "n_taps": args.n_taps, "n_shear": args.n_shear,
                  "seed": args.seed, "device": "cpu",
                  "n_train_cases": len(train_names)}

    # ---- 3. constant train-mean predictor ---------------------------------
    if "train_mean" in args.methods:
        cases, t_case = [], []
        for i in range(n_cases):
            true = val_items[i]["fields"].numpy()
            t0 = time.perf_counter()
            pred = np.zeros_like(true)          # train mean == 0 in z-score units
            t_case.append(time.perf_counter() - t0)
            m = score_deterministic(pred, true, checks)
            m["case"] = ds_val.files[i].name
            cases.append(m)
        cost = {"seconds_per_case": float(np.mean(t_case)),
                "fit_seconds": 0.0, "peak_rss_gb": peak_rss_gb(),
                "gpu_mem_gb": None, "device": "cpu"}
        pth = write_result(out_dir, "train_mean", "train_mean_constant",
                           dict(cfg_common), cases, cost,
                           {"sensor_fingerprints": fingerprints,
                            "note": "constant predictor: the dataset's train-split "
                                    "per-channel mean, i.e. 0 in z-score units. "
                                    "Uses no sensors.",
                            "crps_mae_check": dict(checks)})
        written.append(str(pth))
        print(f"[train_mean] relL2={summarize(cases)['rel_l2_mean']:.4f} "
              f"CRPS={summarize(cases)['crps']:.4f} -> {pth}", flush=True)

    # ---- 2. nearest-surface-sensor / IDW ----------------------------------
    if "idw" in args.methods:
        # Physical-unit bridge for the one shared observable: pool channel 0 is
        # Cp on the wall, volume channel 3 is Cp in the field.  They are the
        # same physical quantity but are z-scored with different statistics.
        s_mu, s_sd = float(st["s_mean"][SURF_CP_CHANNEL]), float(st["s_std"][SURF_CP_CHANNEL])
        v_mu, v_sd = float(st["v_mean"][VOL_CP_CHANNEL]), float(st["v_std"][VOL_CP_CHANNEL])
        for variant in ("idw_k8", "idw_k8_shearproxy"):
            cases, t_case = [], []
            for i in range(n_cases):
                item, (obs, per_channel) = val_items[i], val_obs[i]
                true = item["fields"].numpy()
                q, pc = val_raw[i]               # physical coordinates, metres
                pv = item["obs_pool_values"].numpy()
                t0 = time.perf_counter()
                pred = np.zeros_like(true)
                # Cp taps (pool channel 0) -> volume Cp channel.
                idx = per_channel[SURF_CP_CHANNEL][1].numpy()
                vals = (pv[idx, SURF_CP_CHANNEL] * s_sd + s_mu - v_mu) / v_sd
                pred[:, VOL_CP_CHANNEL] = idw_predict(
                    pc[idx], vals.astype(np.float64), q, args.idw_k)
                if variant.endswith("shearproxy"):
                    for c in (1, 2, 3):          # tau_x, tau_y, tau_z -> Ux, Uy, Uz
                        jdx = per_channel[c][1].numpy()
                        pred[:, c - 1] = idw_predict(
                            pc[jdx], pv[jdx, c].astype(np.float64), q, args.idw_k)
                t_case.append(time.perf_counter() - t0)
                m = score_deterministic(pred, true, checks)
                m["case"] = ds_val.files[i].name
                cases.append(m)
            cfg = dict(cfg_common); cfg["k"] = args.idw_k
            note = ("IDW k=%d from the drawn surface sensors to every volume node, "
                    "Euclidean distance in physical (metre) coordinates -- the global "
                    "[0,1] normalization is a strongly anisotropic per-axis rescale "
                    "(x:141, y:108, z:11 m), so distances there would be arbitrary. "
                    % args.idw_k)
            note += ("Only the 512 Cp taps are used (the sole observable that is the "
                     "same physical field as a generated channel); Ux,Uy,Uz are left "
                     "at the train mean." if variant == "idw_k8" else
                     "The wall-shear components are additionally pushed onto Ux,Uy,Uz "
                     "in z-score units as a (dimensionally arbitrary) proxy.")
            pth = write_result(out_dir, variant, variant, cfg, cases,
                               {"seconds_per_case": float(np.mean(t_case)),
                                "fit_seconds": 0.0, "peak_rss_gb": peak_rss_gb(),
                                "gpu_mem_gb": None, "device": "cpu"},
                               {"sensor_fingerprints": fingerprints, "note": note,
                                "crps_mae_check": dict(checks)})
            written.append(str(pth))
            s = summarize(cases)
            print(f"[{variant}] relL2={s['rel_l2_mean']:.4f} CRPS={s['crps']:.4f} "
                  f"-> {pth}", flush=True)

    # ---- 1. gappy POD -----------------------------------------------------
    if "gappy_pod" in args.methods:
        ranks = sorted(set(args.ranks))
        pod_cases: Dict[int, List[Dict]] = {r: [] for r in ranks}
        t_basis, t_solve, energies, oracle = [], {r: [] for r in ranks}, {}, []

        for i in range(n_cases):
            item, (obs, per_channel) = val_items[i], val_obs[i]
            true = item["fields"].numpy()
            tgt_vol, tgt_surf = val_raw[i]                 # raw (physical) coords
            n_vol, n_surf = tgt_vol.shape[0], tgt_surf.shape[0]
            n_s_cols, n_p_cols = n_surf * N_CH, 2

            with Timer() as tb:
                print(f"  [gappy_pod] case {i} ({ds_val.files[i].name}): resampling "
                      f"{len(train_names)} train cases onto its own "
                      f"{n_vol} volume / {n_surf} surface nodes", flush=True)
                X = assemble_train_matrix(root, train_names, tgt_vol, tgt_surf,
                                          st, args.workers)
                mu = X.mean(axis=0)
                X -= mu
                pod = BlockGramPOD(X, mu, max(max(ranks), 1))
            t_basis.append(tb.wall)
            energies[ds_val.files[i].name] = {
                str(r): float(pod.energy[min(r, len(pod.energy)) - 1])
                for r in ranks if r > 0}
            print(f"      basis in {tb.wall:.0f}s, RSS {peak_rss_gb():.1f} GB, "
                  f"energy r={max(ranks)}: "
                  f"{pod.energy[min(max(ranks), len(pod.energy)) - 1]:.4f}", flush=True)

            # Observation columns: the drawn pool indices, exactly.  No mapping
            # or interpolation is involved -- the basis lives on this case's own
            # pool points, so sensor s of pool channel c is column idx[s]*4 + c.
            pv = item["obs_pool_values"].numpy()
            cols = [idx.numpy() * N_CH + c for c, idx in per_channel]
            vals = [pv[idx.numpy(), c] for c, idx in per_channel]
            cols.append(n_s_cols + np.arange(n_p_cols))
            vals.append(item["param_values"].numpy()[:, 0])
            cols = np.concatenate(cols).astype(np.int64)
            vals = np.concatenate(vals).astype(np.float64)
            assert cols.shape[0] == obs["coords"].shape[1]
            # every observed value must be the one build_surface_obs handed over
            assert np.allclose(vals[:-n_p_cols],
                               obs["values"][0, :-n_p_cols, 0].numpy(), atol=0,
                               rtol=0), "observation value/column mismatch"

            if not args.no_oracle:
                Xv = np.ascontiguousarray(X[:, n_s_cols + n_p_cols:])
                Gv = (Xv @ Xv.T).astype(np.float64)
                yv = (true.ravel() - mu[n_s_cols + n_p_cols:]).astype(np.float32)
                cv = (Xv @ yv).astype(np.float64)
                del Xv

            for r in ranks:
                t0 = time.perf_counter()
                if r == 0:
                    full = pod.mu
                else:
                    pod.set_rank(r)
                    full = pod.reconstruct(cols, vals)
                pred = full[n_s_cols + n_p_cols:].reshape(n_vol, N_CH)
                t_solve[r].append(time.perf_counter() - t0)
                m = score_deterministic(np.ascontiguousarray(pred), true, checks)
                m["case"] = ds_val.files[i].name
                pod_cases[r].append(m)

                if not args.no_oracle:
                    # Best rank-r reconstruction if the *whole volume* were
                    # observed: a = argmin ||mu_v + a @ Phi_v - y||.  Scored with
                    # the same per-field-then-average rel-L2 as ensemble_metrics.
                    if r == 0:
                        po = mu[n_s_cols + n_p_cols:]
                    else:
                        W = (pod.Ur * pod.inv_s[None, :])            # [n, r]
                        a, *_ = np.linalg.lstsq(W.T @ Gv @ W, W.T @ cv, rcond=None)
                        po = (pod.mu + (W @ a).astype(np.float32) @ X
                              )[n_s_cols + n_p_cols:]
                    po = po.reshape(n_vol, N_CH)
                    pf = {f: float(np.linalg.norm(po[:, j] - true[:, j])
                                   / (np.linalg.norm(true[:, j]) + 1e-12))
                          for j, f in enumerate(FIELD_NAMES)}
                    oracle.append({"case": ds_val.files[i].name, "rank": r,
                                   "oracle_rel_l2": float(np.mean(list(pf.values()))),
                                   "oracle_rel_l2_per_field": pf})
            print("      rel-L2 " + "  ".join(
                f"r{r}={pod_cases[r][-1]['aggregate']['rel_l2_mean']:.4f}"
                for r in ranks), flush=True)
            del X, pod, mu
            if not args.no_oracle:
                del Gv, cv

        for r in ranks:
            cases = pod_cases[r]
            cfg = dict(cfg_common)
            cfg.update({"rank": r, "basis_discretisation": "per val case (own nodes)",
                        "energy_captured": energies})
            orc = [o["oracle_rel_l2"] for o in oracle if o["rank"] == r]
            pth = write_result(
                out_dir, f"gappy_pod_r{r}",
                "gappy_POD" if r > 0 else "gappy_POD_rank0_trainmeanfield",
                cfg, cases,
                {"seconds_per_case": float(np.mean(t_basis) + np.mean(t_solve[r])),
                 "basis_seconds_per_case": float(np.mean(t_basis)),
                 "solve_seconds_per_case": float(np.mean(t_solve[r])),
                 "fit_seconds": float(np.sum(t_basis)),
                 "peak_rss_gb": peak_rss_gb(), "gpu_mem_gb": None, "device": "cpu"},
                {"sensor_fingerprints": fingerprints,
                 "oracle_projection_rel_l2": float(np.mean(orc)) if orc else None,
                 "oracle_projection_per_case": [o for o in oracle if o["rank"] == r],
                 "note": "Joint POD over [surface pool ; params ; volume] across the "
                         "TRAIN cases, built in each val case's own discretisation "
                         "(the cases share no nodes), so the val truth is never "
                         "resampled.  Coefficients by least squares on the drawn "
                         "surface sensors plus the two exact parameter tokens; the "
                         "volume rows give the interior.  rank 0 = the train mean "
                         "field (no sensors used).  oracle_projection_rel_l2 is a "
                         "diagnostic, not a baseline: the best rank-r fit to the "
                         "truth if the whole volume were observed.",
                 "crps_mae_check": dict(checks)})
            written.append(str(pth))
            s = summarize(cases)
            print(f"[gappy_pod r={r:3d}] relL2={s['rel_l2_mean']:.4f} "
                  f"CRPS={s['crps']:.4f} oracle_relL2="
                  f"{np.mean(orc) if orc else float('nan'):.4f} -> {pth}", flush=True)

    print(f"\n[wing-classical] CRPS==MAE check: max |CRPS - MAE| over every "
          f"method/field/case = {checks.get('max_abs_crps_minus_mae', 0.0):.3e} "
          f"(tolerance 1e-12)", flush=True)
    print(f"[wing-classical] peak RSS {peak_rss_gb():.1f} GB")
    print("[wing-classical] wrote:")
    for w in written:
        print("   ", w)


if __name__ == "__main__":
    main()
