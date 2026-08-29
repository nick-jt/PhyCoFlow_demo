"""Posterior-ensemble evaluation for PointCloudFFM runs.

Draws K conditional samples per snapshot and computes probabilistic metrics:
ensemble-mean / single-sample relative L2, fair CRPS, spread-error ratio,
central-interval coverage, and rank histograms.

Sampling is chunked over query points. For the GL_rbf backbones the velocity
at a query point depends only on (t, state at that point, observations), so
integrating the ODE chunk-by-chunk from one jointly drawn source field is
exact, and full 125^3 snapshots fit in memory.

Usage:
    python ensemble_eval.py --run-dir ../Save_TrainedModel/<run> --ckpt best.pt \
        --K 8 --n-steps 16 --n-snapshots 8 --n-obs 19531 19531 --cond-fields 0 2 \
        --noise-sigma 0.0 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from evaluate_ffm import _build_model, _normalize_eval_config
from obs_consistency import scatter_observed_values


# ---------------------------------------------------------------------------
# Canonical-protocol guards (shared by every eval driver; audit 2026-08-29)
# ---------------------------------------------------------------------------

# Fingerprint of the canonical sensor draw (seed 0, cond_fields [0,2],
# n_obs 19531+19531) at snapshot 29, drawn on an H100 SXM compute node.
# See BASELINE_AUDIT_2026-08-28.md sections 2 and 33.
CANONICAL_FP = {"snap": 29, "sensors": 39062, "idx_sum": 37987162596}


def require_compute_node() -> None:
    """Abort when not inside a SLURM job (or on a PCIe/login-node GPU).

    torch.randperm on CUDA is NOT portable across GPU SKU: H100 PCIe (login)
    gives idx_sum=38044045759 at the canonical point, H100 SXM (compute)
    gives 37987162596, so an eval run off-SLURM silently scores a different
    inverse problem. Override with ALLOW_LOGIN_EVAL=1 (debugging only).
    """
    if os.environ.get("ALLOW_LOGIN_EVAL") == "1":
        print("[nodecheck] WARNING: ALLOW_LOGIN_EVAL=1 -- compute-node guard "
              "bypassed; sensor layouts may NOT match the canonical "
              "fingerprint and numbers are not comparable.", flush=True)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "[nodecheck] SLURM_JOB_ID is unset: refusing to run a canonical "
            "eval outside a SLURM compute job (login-node CUDA randperm draws "
            "a different sensor layout). Submit via sbatch/srun, or set "
            "ALLOW_LOGIN_EVAL=1 to override for debugging.")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        if "PCIe" in name:
            raise SystemExit(
                f"[nodecheck] GPU '{name}' is the PCIe (login-node) SKU; the "
                "canonical fingerprint requires H100 SXM compute nodes. Set "
                "ALLOW_LOGIN_EVAL=1 to override for debugging.")


def check_canonical_fingerprint(snap, sensors, idx_sum, seed, cond_fields, n_obs) -> None:
    """Abort on a wrong sensor set at the canonical operating point.

    Fires only when (snap, seed, cond_fields, n_obs) match the canonical
    protocol exactly, so non-canonical sweeps (other seeds, densities,
    channel sets) are unaffected. A mismatch means the wrong
    build_sparse_condition variant (helpers_baseline's CUDA-count copy) or
    the wrong GPU SKU -- either way a different inverse problem, so the run
    must not produce numbers.
    """
    if int(snap) != CANONICAL_FP["snap"] or int(seed) != 0:
        return
    if [int(c) for c in cond_fields] != [0, 2]:
        return
    if [int(x) for x in n_obs] != [19531, 19531]:
        return
    if int(sensors) != CANONICAL_FP["sensors"] or int(idx_sum) != CANONICAL_FP["idx_sum"]:
        raise SystemExit(
            f"[seedcheck] FINGERPRINT MISMATCH at snap 29: got sensors={int(sensors)} "
            f"idx_sum={int(idx_sum)}, expected sensors={CANONICAL_FP['sensors']} "
            f"idx_sum={CANONICAL_FP['idx_sum']}. Wrong build_sparse_condition "
            "variant or wrong GPU SKU -- numbers would not be comparable. Aborting.")


# ---------------------------------------------------------------------------
# Model / run loading
# ---------------------------------------------------------------------------

def load_run(run_dir: str, ckpt_name: str = "best.pt", device: str = "cuda:0"):
    run_dir = Path(run_dir)
    cfg = _normalize_eval_config(json.load(open(run_dir / "args.json")))

    script_dir = Path(__file__).resolve().parent
    demo_dir = script_dir.parent
    data_path = cfg["data"]
    if not os.path.isabs(data_path):
        data_path = str((script_dir / data_path).resolve())

    dataset = TurbulentCombustionH5Dataset(
        data_path,
        split="val",
        train_ratio=cfg.get("train_ratio", 0.9),
        field_names=cfg.get("field_names"),
        seed=cfg.get("seed", 42),
        time_stride=cfg.get("time_stride", 1),
        stats_path=str(run_dir / "dataset_stats.pt"),
    )

    model = _build_model(cfg, dataset)
    ckpt = torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    used_ema = False
    if ckpt.get("ema") is not None:
        state = model.state_dict()
        for k, v in ckpt["ema"]["shadow"].items():
            state[k].copy_(v.to(state[k].dtype))
        used_ema = True
    model.to(device).eval()
    print(f"[ensemble_eval] loaded {run_dir.name}/{ckpt_name} "
          f"(epoch {ckpt.get('epoch')}, ema={used_ema})")
    return model, dataset, cfg


# ---------------------------------------------------------------------------
# Chunked ensemble sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_ensemble(
    model,
    coords: torch.Tensor,          # [1, N, 3]
    obs: Dict[str, torch.Tensor],
    K: int,
    n_steps: int = 16,
    chunk: int = 262_144,
    clamp_hard: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Return K samples [K, N, C] for one snapshot's conditioning."""
    device = coords.device
    n = coords.shape[1]
    n_fields = model.model.n_fields
    out = torch.empty(K, n, n_fields, device="cpu")

    for k in range(K):
        torch.manual_seed(seed * 10_000 + k)
        x = model.sample_source(coords)  # joint GP draw over all points
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=coords.dtype)
        for i in range(n_steps):
            t0 = ts[i].expand(1)
            dt = ts[i + 1] - ts[i]
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                v = model.model(
                    t0, x[:, s:e], coords[:, s:e],
                    obs["coords"], obs["values"], obs["mask"], obs["field_ids"],
                )
                x[:, s:e] = x[:, s:e] + dt * v
            if clamp_hard:
                x = scatter_observed_values(
                    x=x, obs_values=obs["values"], obs_mask=obs["mask"],
                    obs_indices=obs["indices"], obs_field_ids=obs["field_ids"],
                    strength=1.0,
                )
        out[k] = x[0].cpu()
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rel_l2(pred: np.ndarray, true: np.ndarray, axis=None) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def fair_crps(ens: np.ndarray, y: np.ndarray, chunk: int = 500_000) -> float:
    """Fair empirical CRPS, averaged over points. ens [K, P], y [P]."""
    K = ens.shape[0]
    total = 0.0
    n = y.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x = np.sort(ens[:, s:e], axis=0)                    # [K, p]
        term1 = np.abs(x - y[None, s:e]).mean(axis=0)       # [p]
        w = (2.0 * np.arange(1, K + 1) - K - 1.0)           # [K]
        pair = (w[:, None] * x).sum(axis=0) / (K * (K - 1)) # fair estimator
        total += (term1 - pair).sum()
    return float(total / n)


def save_ensemble_figure(ens, true, coords_raw, field_names, out_path, tag=""):
    """Truth / ensemble mean / one sample / predictive spread, on a z-midplane slice.

    On by default. Every failure found in this project's baseline audit was
    visible here and invisible in the aggregate metrics: a sampler chunking
    raster indices produced sheared striping at relL2 ~1.0, an auto-decoder
    collapsed to a latent-independent mean field, and a latent baseline that
    leads on relative L2 has no phase coherence above k~7. Numbers alone did
    not catch any of them.

    Returns wall-clock seconds spent here, so callers can subtract it from
    runtime measurements (figure generation is diagnostic, not model cost).
    """
    import time as _time
    t0 = _time.perf_counter()
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    c = np.asarray(coords_raw)
    is_3d = c.shape[-1] >= 3 and bool(np.ptp(c[:, 2]) > 1e-6)
    if is_3d:
        lv = np.unique(c[:, 2])
        mask = c[:, 2] == lv[len(lv) // 2]
    else:
        mask = slice(None)
    xy = c[mask][:, :2]
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])

    mean = ens.mean(axis=0)
    std = ens.std(axis=0, ddof=1) if ens.shape[0] > 1 else np.zeros_like(mean)
    nC = len(field_names)
    fig, axs = plt.subplots(4, nC, figsize=(3.3 * nC, 12.4))
    axs = np.atleast_2d(axs)
    rows = [("truth", true), ("ens. mean", mean), ("sample 0", ens[0]), ("spread (std)", std)]
    for j in range(nC):
        tj = true[mask, j]
        lo, hi = (float(v) for v in np.percentile(tj, [1, 99]))
        for r, (lbl, arr) in enumerate(rows):
            a = arr[mask, j]
            if lbl.startswith("spread"):
                im = axs[r, j].tripcolor(tri, a, cmap="viridis", shading="gouraud")
            else:
                im = axs[r, j].tripcolor(tri, a, cmap="coolwarm", shading="gouraud",
                                         vmin=lo, vmax=hi)
            fig.colorbar(im, ax=axs[r, j], fraction=0.046)
            axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
            if r == 0:
                axs[r, j].set_title(str(field_names[j]), fontsize=11)
            if j == 0:
                axs[r, j].set_ylabel(lbl, fontsize=10)
    fig.suptitle(f"z-midplane slice  {tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return _time.perf_counter() - t0


def ensemble_metrics(ens: np.ndarray, true: np.ndarray,
                     field_names: Sequence[str]) -> Dict:
    """ens [K, N, C], true [N, C] (normalized units)."""
    K, n, c = ens.shape
    mean = ens.mean(axis=0)
    std = ens.std(axis=0, ddof=1)

    per_field = {}
    rank_hist = np.zeros((c, K + 1), dtype=np.int64)
    for j in range(c):
        e = ens[:, :, j]
        y = true[:, j]
        m = mean[:, j]
        err = m - y
        rmse = float(np.sqrt((err ** 2).mean()))
        spread = float(np.sqrt((std[:, j] ** 2).mean()))
        lo50, hi50 = np.percentile(e, [25, 75], axis=0)
        lo90, hi90 = np.percentile(e, [5, 95], axis=0)
        ranks = (e < y[None, :]).sum(axis=0)
        rank_hist[j] = np.bincount(ranks, minlength=K + 1)
        per_field[field_names[j]] = {
            "rel_l2_mean": _rel_l2(m, y),
            "rel_l2_single": float(np.mean([_rel_l2(e[k], y) for k in range(K)])),
            "crps": fair_crps(e, y),
            "rmse": rmse,
            "spread": spread,
            "spread_error_ratio": spread / (rmse + 1e-12),
            "coverage_50": float(((y >= lo50) & (y <= hi50)).mean()),
            "coverage_90": float(((y >= lo90) & (y <= hi90)).mean()),
        }

    agg = {
        key: float(np.mean([per_field[f][key] for f in per_field]))
        for key in next(iter(per_field.values()))
    }
    return {"per_field": per_field, "aggregate": agg,
            "rank_hist": rank_hist.tolist()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pt")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=16)
    p.add_argument("--n-snapshots", type=int, default=8)
    p.add_argument("--cond-fields", type=int, nargs="+", default=None)
    p.add_argument("--n-obs", type=int, nargs="+", default=None,
                   help="Exact sensors per conditioned field; defaults to run config max.")
    p.add_argument("--noise-sigma", type=float, default=0.0,
                   help="Gaussian sensor noise in z-score units at eval time.")
    p.add_argument("--occlude", type=str, default="",
                   help="'slab:0.25' or 'ball:0.25' - remove sensors in a region.")
    p.add_argument("--drop-fields", type=int, nargs="+", default=None,
                   help="Field ids whose sensors are removed at eval time.")
    p.add_argument("--no-figs", action="store_true",
                   help="Disable the per-snapshot diagnostic figure. Figures are ON by "
                        "default; their wall-clock cost is measured and reported "
                        "separately as figure_seconds so it can be subtracted from any "
                        "runtime comparison.")
    p.add_argument("--fig-every", type=int, default=1,
                   help="Write a figure every Nth evaluated snapshot.")
    p.add_argument("--op-seed", type=int, default=1000,
                   help="Base seed for operator realizations; per-snapshot seed = op_seed + snapshot index (match baselines via ENSEMBLE_OP_SEED).")
    p.add_argument("--no-clamp", action="store_true",
                   help="Disable hard observation clamping (use under noise).")
    p.add_argument("--query-subset", type=int, default=None,
                   help="Random query subset size (default: full snapshot).")
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    require_compute_node()
    model, dataset, cfg = load_run(args.run_dir, args.ckpt, args.device)
    cond_fields = args.cond_fields or cfg["cond_fields"]
    n_obs = args.n_obs or cfg["n_obs_max_list"]
    device = torch.device(args.device)

    clamp_hard = not (args.no_clamp or args.noise_sigma > 0.0 or args.occlude or args.drop_fields)
    if args.noise_sigma > 0.0 and not args.no_clamp:
        print("[ensemble_eval] noise > 0: hard clamping disabled automatically")

    results: List[Dict] = []
    fig_seconds = 0.0
    rng = np.random.default_rng(args.seed)
    snap_ids = rng.choice(len(dataset), size=min(args.n_snapshots, len(dataset)),
                          replace=False)

    for si, snap in enumerate(snap_ids):
        item = dataset[int(snap)]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)

        torch.manual_seed(args.seed * 777 + int(snap))
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=cond_fields, n_obs_min=n_obs, n_obs_max=n_obs,
        )
        _sel = oi[om > 0]
        print(f"[seedcheck] snap={int(snap)} sensors={int(_sel.numel())} "
              f"idx_sum={int(_sel.sum().item())}", flush=True)
        check_canonical_fingerprint(int(snap), int(_sel.numel()),
                                    int(_sel.sum().item()), args.seed,
                                    cond_fields, n_obs)
        _opg = torch.Generator(device=ov.device).manual_seed(args.op_seed + int(snap))
        if args.noise_sigma > 0.0:
            _noise = torch.randn(ov.shape, device=ov.device, dtype=ov.dtype,
                                 generator=_opg)
            ov = ov + args.noise_sigma * _noise * om.unsqueeze(-1)
        if args.occlude:
            from measurement_ops import apply_occlusion
            _kind, _frac = args.occlude.split(":")
            om = apply_occlusion(oc, om, prob=1.0, kind=_kind,
                                 frac_min=float(_frac), frac_max=float(_frac),
                                 generator=_opg)
        if args.drop_fields:
            _ids = torch.tensor(args.drop_fields, device=om.device)
            _rm = (ofid.unsqueeze(-1) == _ids.view(1, 1, -1)).any(-1)
            om = om * (~_rm).to(om.dtype)
        if args.occlude or args.drop_fields:
            print(f"[eval_ops] snap={snap} noise={args.noise_sigma} "
                  f"occlude={args.occlude!r} drop={args.drop_fields} "
                  f"valid={int(om.sum())}", flush=True)
        obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
               "field_ids": ofid}

        if args.query_subset is not None and args.query_subset < coords.shape[1]:
            sel = torch.from_numpy(
                rng.choice(coords.shape[1], size=args.query_subset, replace=False)
            ).sort().values.to(device)
            coords_q = coords[:, sel]
            fields_q = fields[:, sel]
            clamp = False  # obs indices no longer valid on the subset
        else:
            coords_q, fields_q, clamp = coords, fields, clamp_hard

        ens = sample_ensemble(model, coords_q, obs, K=args.K,
                              n_steps=args.n_steps, chunk=args.chunk,
                              clamp_hard=clamp, seed=args.seed * 131 + si)
        m = ensemble_metrics(ens.numpy(), fields_q[0].cpu().numpy(),
                             dataset.field_names)
        m["snapshot"] = int(snap)

        # Diagnostic figure (default ON). Timed separately so it never enters a
        # runtime comparison -- reported as figure_seconds in the output.
        if (not args.no_figs) and (si % max(1, args.fig_every) == 0):
            fig_dir = Path(args.out).parent if args.out else Path(args.run_dir) / "Evaluation"
            fig_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(args.out).stem if args.out else f"eval_K{args.K}_nfe{args.n_steps}"
            try:
                cr = item.get("coords_raw")
                cr = cr.cpu().numpy() if cr is not None else coords_q[0].cpu().numpy()
                if args.query_subset is not None and args.query_subset < coords.shape[1]:
                    cr = cr[sel.cpu().numpy()]
                fig_seconds += save_ensemble_figure(
                    ens.numpy(), fields_q[0].cpu().numpy(), cr,
                    dataset.field_names, fig_dir / f"{stem}_snap{int(snap)}.png",
                    tag=(f"{Path(args.run_dir).name}  snap {int(snap)}  "
                         f"K={args.K} NFE={args.n_steps}  "
                         f"relL2={m['aggregate']['rel_l2_mean']:.4f}"))
            except Exception as exc:                      # a plot must never kill an eval
                print(f"  [warn] figure for snap {int(snap)} failed: {exc}", flush=True)

        results.append(m)
        agg = m["aggregate"]
        print(f"  snap {snap}: relL2(mean)={agg['rel_l2_mean']:.4f} "
              f"relL2(single)={agg['rel_l2_single']:.4f} CRPS={agg['crps']:.4f} "
              f"spread/err={agg['spread_error_ratio']:.3f} "
              f"cov50={agg['coverage_50']:.3f} cov90={agg['coverage_90']:.3f}")

    keys = list(results[0]["aggregate"].keys())
    summary = {k: float(np.mean([r["aggregate"][k] for r in results])) for k in keys}
    print(f"\n[ensemble_eval] diagnostic figures cost {fig_seconds:.1f}s "
          f"(excluded from runtime comparisons)")
    print("\n[ensemble_eval] summary over snapshots:")
    for k, v in summary.items():
        print(f"  {k:22s} {v:.5f}")

    if args.out:
        payload = {
            "run_dir": str(args.run_dir), "ckpt": args.ckpt,
            "K": args.K, "n_steps": args.n_steps,
            "cond_fields": cond_fields, "n_obs": n_obs,
            "noise_sigma": args.noise_sigma,
            "summary": summary, "snapshots": results,
            "figure_seconds": round(fig_seconds, 2),
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[ensemble_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
