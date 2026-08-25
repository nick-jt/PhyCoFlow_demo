"""Evaluate a SHIFT-WING run: surface-only observations -> volumetric posterior.

For each validation case, samples a K-member posterior ensemble of the
volumetric fields from a fixed number of surface pressure taps (and
optionally wall-shear sensors) plus the Mach/alpha parameter tokens, and
reports per-field ensemble metrics. Optionally saves midspan slice plots
(prediction vs truth vs ensemble std) as scatter maps on the point cloud.

Usage:
    python evaluate_wing.py --run-dir ../Save_TrainedModel/<wing run> \
        --ckpt best.pt --K 4 --n-steps 16 --n-taps 512 --n-shear 128 \
        --n-cases 8 --plots --out wing_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from dataset_shiftwing import ShiftWingDataset, SURF_VALUE_FIELD_IDS, PARAM_FIELD_IDS
from evaluate_ffm import _build_model, _normalize_eval_config
from ensemble_eval import sample_ensemble, ensemble_metrics


def build_surface_obs(item: Dict[str, torch.Tensor], n_taps: int, n_shear: int,
                      device, seed: int, noise_sigma: float = 0.0):
    """Fixed-count surface observation tuple + parameter tokens."""
    g = torch.Generator().manual_seed(seed)
    pc = item["obs_pool_coords"]
    pv = item["obs_pool_values"]
    n_pool = pc.shape[0]

    counts = [n_taps, n_shear, n_shear, n_shear]
    coords_l, values_l, ids_l = [], [], []
    for c, m in enumerate(counts):
        if m <= 0:
            continue
        idx = torch.randperm(n_pool, generator=g)[:m]
        coords_l.append(pc[idx])
        values_l.append(pv[idx, c:c + 1])
        ids_l.append(torch.full((m,), SURF_VALUE_FIELD_IDS[c], dtype=torch.long))

    # Parameter tokens (exact, appended last).
    coords_l.append(item["param_coords"])
    values_l.append(item["param_values"])
    ids_l.append(item["param_field_ids"])

    oc = torch.cat(coords_l, 0)[None].to(device)
    ov = torch.cat(values_l, 0)[None].to(device)
    ofid = torch.cat(ids_l, 0)[None].to(device)
    om = torch.ones(1, oc.shape[1], device=device)
    if noise_sigma > 0:
        n_sensor = oc.shape[1] - item["param_coords"].shape[0]
        noise = torch.zeros_like(ov)
        noise[:, :n_sensor] = noise_sigma * torch.randn(
            1, n_sensor, 1, generator=g).to(device)
        ov = ov + noise
    oi = torch.zeros(1, oc.shape[1], dtype=torch.long, device=device)
    return {"coords": oc, "values": ov, "mask": om, "indices": oi,
            "field_ids": ofid}


def midspan_slice_plot(coords, true, pred, std, field_names, out_png,
                       axis=1, width=0.01):
    """Scatter maps on a thin midspan slab of the point cloud."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = coords
    mid = 0.5 * (c[:, axis].min() + c[:, axis].max())
    sl = np.abs(c[:, axis] - mid) < width
    x, z = c[sl, 0], c[sl, 2]
    nf = true.shape[1]
    fig, axes = plt.subplots(3, nf, figsize=(3.1 * nf, 7.5), dpi=140)
    for j in range(nf):
        vmin, vmax = np.percentile(true[sl, j], [1, 99])
        for row, (data, label) in enumerate(
                [(true[sl, j], "truth"), (pred[sl, j], "posterior mean"),
                 (std[sl, j], "posterior std")]):
            ax = axes[row, j]
            kw = dict(s=1.0, rasterized=True)
            if row < 2:
                sc = ax.scatter(x, z, c=data, vmin=vmin, vmax=vmax,
                                cmap="RdBu_r", **kw)
            else:
                sc = ax.scatter(x, z, c=data, cmap="magma", **kw)
            plt.colorbar(sc, ax=ax, fraction=0.04)
            ax.set_title(f"{field_names[j]} ({label})", fontsize=8)
            ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pt")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=16)
    p.add_argument("--n-taps", type=int, default=512)
    p.add_argument("--n-shear", type=int, default=128)
    p.add_argument("--noise-sigma", type=float, default=0.0)
    p.add_argument("--n-cases", type=int, default=8)
    p.add_argument("--chunk", type=int, default=131_072)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = _normalize_eval_config(json.load(open(run_dir / "args.json")))
    dataset = ShiftWingDataset(cfg["processed_root"], split="val")
    cfg.setdefault("n_obs_field_types", dataset.n_obs_field_types)

    model = _build_model(cfg, dataset)
    ckpt = torch.load(run_dir / args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema") is not None:
        state = model.state_dict()
        for k, v in ckpt["ema"]["shadow"].items():
            state[k].copy_(v.to(state[k].dtype))
    device = torch.device(args.device)
    model.to(device).eval()
    print(f"[wing_eval] {run_dir.name}/{args.ckpt} epoch={ckpt.get('epoch')} "
          f"taps={args.n_taps} shear={args.n_shear} K={args.K}")

    results: List[Dict] = []
    n_cases = min(args.n_cases, len(dataset))
    for i in range(n_cases):
        item = dataset[i]
        coords = item["coords"][None].to(device)
        true = item["fields"].numpy()
        obs = build_surface_obs(item, args.n_taps, args.n_shear, device,
                                seed=args.seed * 100 + i,
                                noise_sigma=args.noise_sigma)
        ens = sample_ensemble(model, coords, obs, K=args.K,
                              n_steps=args.n_steps, chunk=args.chunk,
                              clamp_hard=False, seed=args.seed * 31 + i)
        m = ensemble_metrics(ens.numpy(), true, dataset.field_names)
        m["case"] = dataset.files[i].name
        results.append(m)
        agg = m["aggregate"]
        print(f"  {m['case']}: relL2(mean)={agg['rel_l2_mean']:.4f} "
              f"CRPS={agg['crps']:.4f} cov90={agg['coverage_90']:.3f}")

        if args.plots:
            # Keep figures with the other evaluation artifacts and tag them
            # with the solver-step count, so sweeps over NFE do not overwrite
            # each other.
            _fig_dir = run_dir / "Evaluation"
            _fig_dir.mkdir(parents=True, exist_ok=True)
            out_png = _fig_dir / f"wing_eval_slice_nfe{args.n_steps}_{i:02d}.png"
            e = ens.numpy()
            midspan_slice_plot(item["coords"].numpy(), true, e.mean(0),
                               e.std(0, ddof=1), dataset.field_names, out_png)
            print(f"    slice plot -> {out_png}")

    keys = list(results[0]["aggregate"].keys())
    summary = {k: float(np.mean([r["aggregate"][k] for r in results])) for k in keys}
    print("\n[wing_eval] summary:")
    for k, v in summary.items():
        print(f"  {k:22s} {v:.5f}")
    if args.out:
        json.dump({"config": vars(args), "summary": summary,
                   "cases": results}, open(args.out, "w"), indent=2)
        print(f"[wing_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
