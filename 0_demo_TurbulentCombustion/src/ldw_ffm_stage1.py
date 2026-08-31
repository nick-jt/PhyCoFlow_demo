"""LDW-FFM Stage-1 blend statistics (PLAN_IMPROVE_2026-08-30 section 5).

Tests the LDW-FFM idea with ZERO model changes: per-channel convex blend
    u_blend = w * IDW + (1 - w) * FFM_member
on held-out cube-3 validation data, at three sensor densities.

For one density this script, per canonical snapshot:
  (a) draws the canonical seed-0 sensor set (torch.manual_seed(seed*777+snap)
      + helpers.build_sparse_condition on CUDA -- bit-identical to
      ensemble_eval.main, fingerprint-gated at snap 29 / n=19531);
  (b) samples the K=8, NFE-4, hard-clamped FFM ensemble with the exact
      canonical member seeding (sample_ensemble's torch.manual_seed
      (seed*10_000+k) with seed = args.seed*131 + si, si = position in the
      rng.choice(50) permutation -- reproduces canonical_all50_nfe4_K8);
  (c) computes the non-periodic IDW-k8 prediction with
      baseline_classical_jhu.kd_predict (unobserved channels Uy,p = train
      mean = 0 in z-units);
  (d) saves per-channel SUFFICIENT STATISTICS for blending instead of fields:
      with d = u - FFM_mean and g = u - IDW,
        a = <d,d>,  b = <g,g>,  c = <d,g>,  u2 = <u,u>
      so ||u - (w*IDW + (1-w)*FFM_mean)||^2 = a - 2w(a-c) + w^2(a+b-2c) is
      closed-form in w; plus ensemble metrics of the BLENDED MEMBERS on a
      fine w-grid. The blend is per-point affine with slope (1-w) >= 0, so:
        - the fair-CRPS pair term scales exactly by (1-w),
        - member percentiles blend affinely: q_w = (1-w) q_0 + w IDW,
        - the ensemble spread scales exactly by (1-w).
      Only the first CRPS term needs an explicit per-w pass (cheap on GPU).

No tuning happens here: TUNE/TEST-split fitting of w* lives in
ldw_ffm_stage1_analyze.py, which consumes these JSONs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = ("/home/ntricard/generative_reconstruction/temp/"
       "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src")
sys.path.insert(0, SRC)

os.environ.setdefault("JHU_SPLIT_MODE", "block")
os.environ.setdefault("JHU_SPLIT_GAP", "0")

from helpers import build_sparse_condition                    # noqa: E402
from ensemble_eval import (                                   # noqa: E402
    load_run,
    sample_ensemble,
    require_compute_node,
    check_canonical_fingerprint,
)
from baseline_classical_jhu import kd_predict                 # noqa: E402

FIELD_NAMES = ("Ux", "Uy", "Uz", "p")
COND_FIELDS = [0, 2]
RUN_DIR = ("/home/ntricard/generative_reconstruction/temp/"
           "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/"
           "Save_TrainedModel/JHU/pointcloud_ffm/"
           "iclr_jhu_xcube_spec02_DemoN29_20260822_140100")


@torch.no_grad()
def blend_stats(ens: torch.Tensor, y: torch.Tensor, idw: torch.Tensor,
                w_grid: np.ndarray) -> dict:
    """ens [K,N,C], y [N,C], idw [N,C]; all float64 on the same device.

    Returns per-channel sufficient statistics + blended-ensemble metric grids.
    """
    K, N, C = ens.shape
    Fm = ens.mean(dim=0)                       # [N,C]
    d = y - Fm
    g = y - idw
    a = (d * d).sum(0)                         # [C]
    b = (g * g).sum(0)
    c = (d * g).sum(0)
    u2 = (y * y).sum(0)
    ynorm = u2.sqrt()

    xs, _ = ens.sort(dim=0)
    wk = (2.0 * torch.arange(1, K + 1, device=ens.device, dtype=ens.dtype)
          - K - 1.0).view(K, 1, 1)
    pair0 = (wk * xs).sum(0) / (K * (K - 1))   # [N,C] fair-CRPS pair term at w=0
    pair0_mean = pair0.mean(0)                 # [C]
    spread0 = ens.var(dim=0, unbiased=True).mean(0).sqrt()   # [C]

    def pct(q: float) -> torch.Tensor:
        # numpy default 'linear' percentile along the member axis
        vi = q / 100.0 * (K - 1)
        lo, hi = int(np.floor(vi)), int(np.ceil(vi))
        fr = vi - lo
        return xs[lo] + fr * (xs[hi] - xs[lo])

    lo90_0, hi90_0 = pct(5.0), pct(95.0)
    lo50_0, hi50_0 = pct(25.0), pct(75.0)

    nW = len(w_grid)
    crps_g = np.empty((nW, C))
    rmse_g = np.empty((nW, C))
    rel_g = np.empty((nW, C))
    cov90_g = np.empty((nW, C))
    cov50_g = np.empty((nW, C))
    for iw, w in enumerate(w_grid):
        w = float(w)
        e = (1.0 - w) * ens + w * idw.unsqueeze(0)              # [K,N,C]
        term1 = (e - y.unsqueeze(0)).abs().mean(dim=(0, 1))     # [C]
        crps_g[iw] = (term1 - (1.0 - w) * pair0_mean).cpu().numpy()
        err = (1.0 - w) * Fm + w * idw - y
        rmse_g[iw] = err.pow(2).mean(0).sqrt().cpu().numpy()
        rel_g[iw] = (err.norm(dim=0) / (ynorm + 1e-12)).cpu().numpy()
        l9 = (1.0 - w) * lo90_0 + w * idw
        h9 = (1.0 - w) * hi90_0 + w * idw
        cov90_g[iw] = ((y >= l9) & (y <= h9)).double().mean(0).cpu().numpy()
        l5 = (1.0 - w) * lo50_0 + w * idw
        h5 = (1.0 - w) * hi50_0 + w * idw
        cov50_g[iw] = ((y >= l5) & (y <= h5)).double().mean(0).cpu().numpy()

    out = {"per_field": {}}
    for j, name in enumerate(FIELD_NAMES):
        out["per_field"][name] = {
            "a": float(a[j]), "b": float(b[j]), "c": float(c[j]),
            "u2": float(u2[j]),
            "spread0": float(spread0[j]),
            "pair0_mean": float(pair0_mean[j]),
            "crps_grid": crps_g[:, j].tolist(),
            "rmse_grid": rmse_g[:, j].tolist(),
            "rel_l2_grid": rel_g[:, j].tolist(),
            "cov90_grid": cov90_g[:, j].tolist(),
            "cov50_grid": cov50_g[:, j].tolist(),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-obs", type=int, required=True,
                   help="sensors per OBSERVED channel (1953 / 19531 / 195312)")
    p.add_argument("--run-dir", default=RUN_DIR)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--idw-k", type=int, default=8)
    p.add_argument("--w-points", type=int, default=101,
                   help="uniform w grid on [0,1] for blended-ensemble metrics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    require_compute_node()
    device = torch.device("cuda:0")
    model, dataset, cfg = load_run(args.run_dir, args.ckpt, "cuda:0")
    if len(dataset) != 50:
        raise SystemExit(f"expected 50 cube-3 val snapshots, got {len(dataset)} "
                         "(JHU_SPLIT_MODE/GAP wrong?)")
    N, C = dataset.num_points, dataset.num_fields

    # coords_box: identical transform to baseline_classical_jhu.main
    craw = dataset.coords_raw.numpy().astype(np.float64)
    side = int(round(N ** (1.0 / 3.0)))
    lo = craw.min(0)
    dx = (craw.max(0) - lo) / (side - 1)
    box = side * dx
    coords_box = np.ascontiguousarray(((craw - lo) / box).astype(np.float64))

    w_grid = np.linspace(0.0, 1.0, args.w_points)
    n_obs = [args.n_obs] * len(COND_FIELDS)

    # canonical snapshot ORDER (member seeds depend on position si)
    rng = np.random.default_rng(args.seed)
    snap_ids = rng.choice(len(dataset), size=min(args.n_snapshots, len(dataset)),
                          replace=False)

    payload = {
        "protocol": {
            "run_dir": args.run_dir, "ckpt": args.ckpt,
            "K": args.K, "n_steps": args.n_steps, "seed": args.seed,
            "cond_fields": COND_FIELDS, "n_obs": n_obs,
            "clamp_hard": True,
            "idw": f"kd_predict mode=idw k={args.idw_k} boxsize=None "
                   "(non-periodic); unobserved channels = train mean (0 in "
                   "z-units)",
            "member_seeding": "sample_ensemble(seed=seed*131+si) -> "
                              "torch.manual_seed(seed*10000+k); si = position "
                              "in rng.choice permutation (canonical)",
            "sensor_draw": "torch.manual_seed(seed*777+snap) + "
                           "build_sparse_condition on cuda:0",
            "field_names": list(FIELD_NAMES),
            "stats": "a=|u-Fmean|^2, b=|u-IDW|^2, c=<u-Fmean,u-IDW>, u2=|u|^2 "
                     "(float64 sums over all points); metric grids are for "
                     "PER-MEMBER blending w*IDW+(1-w)*member",
        },
        "w_grid": w_grid.tolist(),
        "snapshots": [],
    }

    # Exact resume: sensor seed depends only on snap, member seed only on si,
    # so re-running the remaining (si, snap) pairs reproduces the same fields.
    done = {}
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            if prev.get("w_grid") == payload["w_grid"] and \
               prev.get("protocol", {}).get("n_obs") == n_obs:
                done = {s["si"]: s for s in prev.get("snapshots", [])}
                print(f"[resume] found {len(done)} completed snapshots in "
                      f"{args.out}", flush=True)
        except Exception as exc:
            print(f"[resume] could not reuse {args.out}: {exc}", flush=True)
    t_job = time.perf_counter()
    for si, snap in enumerate(snap_ids):
        if si in done:
            payload["snapshots"].append(done[si])
            continue
        t0 = time.perf_counter()
        item = dataset[int(snap)]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)

        torch.manual_seed(args.seed * 777 + int(snap))
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=COND_FIELDS, n_obs_min=n_obs, n_obs_max=n_obs,
        )
        _sel = oi[om > 0]
        print(f"[seedcheck] snap={int(snap)} sensors={int(_sel.numel())} "
              f"idx_sum={int(_sel.sum().item())}", flush=True)
        check_canonical_fingerprint(int(snap), int(_sel.numel()),
                                    int(_sel.sum().item()), args.seed,
                                    COND_FIELDS, n_obs)
        obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
               "field_ids": ofid}

        ens = sample_ensemble(model, coords, obs, K=args.K,
                              n_steps=args.n_steps, chunk=args.chunk,
                              clamp_hard=True, seed=args.seed * 131 + si)
        t_ffm = time.perf_counter() - t0

        # IDW-k8 (CPU, exact reuse of the classical-baseline estimator)
        t1 = time.perf_counter()
        idx_np = oi[0].cpu().numpy()
        fid_np = ofid[0].cpu().numpy()
        val_np = ov[0, :, 0].cpu().numpy()
        sensors = {f: (idx_np[fid_np == f].astype(np.int64),
                       val_np[fid_np == f].astype(np.float32))
                   for f in COND_FIELDS}
        idw = kd_predict(coords_box, sensors, C, "idw", args.idw_k, None)
        t_idw = time.perf_counter() - t1

        t2 = time.perf_counter()
        stats = blend_stats(ens.to(device).double(),
                            fields[0].double(),
                            torch.from_numpy(idw).to(device).double(),
                            w_grid)
        t_stat = time.perf_counter() - t2
        stats["snapshot"] = int(snap)
        stats["si"] = int(si)
        stats["timing_s"] = {"ffm": round(t_ffm, 2), "idw": round(t_idw, 2),
                             "stats": round(t_stat, 2)}
        payload["snapshots"].append(stats)

        pf = stats["per_field"]
        msg = " ".join(
            f"{f}:F{pf[f]['rel_l2_grid'][0]:.3f}/I{pf[f]['rel_l2_grid'][-1]:.3f}"
            for f in FIELD_NAMES)
        print(f"  snap {int(snap)} (si={si}): {msg} "
              f"[ffm {t_ffm:.0f}s idw {t_idw:.0f}s stats {t_stat:.0f}s]",
              flush=True)

        # checkpoint the JSON every snapshot (cheap; long job safety)
        with open(args.out, "w") as f:
            json.dump(payload, f)

        del ens, obs, oc, ov, om, oi, ofid, coords, fields
        torch.cuda.empty_cache()

    payload["total_wall_s"] = round(time.perf_counter() - t_job, 1)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[ldw_ffm_stage1] wrote {args.out} "
          f"({payload['total_wall_s']:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
