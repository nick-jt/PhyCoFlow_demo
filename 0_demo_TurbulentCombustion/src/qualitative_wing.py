"""Qualitative reconstruction + uncertainty figures for the wing (surface-only).

Reconstructs a held-out geometry from surface pressure taps and shear gauges
and shows a midspan slice: truth / posterior mean / single sample / posterior
std, per field. The uncertainty profile that matters here is distance from
the wing surface -- the observations live on the skin, so a well-behaved
predictive distribution should be tight near the surface and widen into the
boundary layer and wake.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("../Paper/iclr2027/figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None)
    p.add_argument("--case", type=int, default=0)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--n-taps", type=int, default=512)
    p.add_argument("--n-shear", type=int, default=128)
    p.add_argument("--tag", default="wing")
    args = p.parse_args()

    import glob
    run_dir = Path(args.run_dir or sorted(glob.glob(
        "../Save_TrainedModel/wing/pointcloud_ffm/iclr_wing_v4_sym_DemoN19_*"))[-1])
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)

    from evaluate_wing import (build_surface_obs, _normalize_eval_config,
                               _build_model)
    from dataset_shiftwing import ShiftWingDataset
    from ensemble_eval import sample_ensemble

    cfg = _normalize_eval_config(json.load(open(run_dir / "args.json")))
    dataset = ShiftWingDataset(cfg["processed_root"], split="val")
    cfg.setdefault("n_obs_field_types", dataset.n_obs_field_types)
    model = _build_model(cfg, dataset)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema") is not None:
        state = model.state_dict()
        for k, v in ckpt["ema"]["shadow"].items():
            state[k].copy_(v.to(state[k].dtype))
    model.to(device).eval()

    item = dataset[args.case]
    coords = item["coords"][None].to(device)
    truth = item["fields"].numpy()
    names = [str(n) for n in dataset.field_names]
    obs = build_surface_obs(item, args.n_taps, args.n_shear, device,
                            seed=args.case, noise_sigma=0.0)
    ens = sample_ensemble(model, coords, obs, K=args.K, n_steps=args.n_steps,
                          chunk=131072, clamp_hard=False, seed=31 + args.case).numpy()
    mean = ens.mean(0); std = ens.std(0); sample0 = ens[0]

    xyz = item["coords"].numpy()
    skin = item["obs_pool_coords"].numpy()
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(skin).query(xyz, k=1)

    # midspan slab
    y = xyz[:, 1]
    y0 = np.median(y[np.abs(y - np.quantile(y, 0.55)) < 1e-6]) if False else np.quantile(y, 0.55)
    band = np.abs(y - y0) < (y.max() - y.min()) * 0.01
    np.savez_compressed(OUT / f"qual_{args.tag}.npz", truth=truth, mean=mean,
                        std=std, sample0=sample0, dist=dist, coords=xyz,
                        band=band, names=np.array(names))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    xs, zs = xyz[band, 0], xyz[band, 2]
    tri = mtri.Triangulation(xs, zs)
    cols = ["truth", "posterior mean", "single sample", "posterior std"]
    show = [3, 0, 2]     # Cp, Ux, Uz
    fig, axes = plt.subplots(len(show), 4, figsize=(12.5, 2.35 * len(show)), dpi=200)
    for r, j in enumerate(show):
        vals = [truth[band, j], mean[band, j], sample0[band, j], std[band, j]]
        vmin, vmax = np.percentile(vals[0], [2, 98])
        for c, arr in enumerate(vals):
            ax = axes[r, c]
            cmap = "viridis" if c == 3 else "RdBu_r"
            vr = (0, np.percentile(arr, 99)) if c == 3 else (vmin, vmax)
            im = ax.tripcolor(tri, arr, cmap=cmap, vmin=vr[0], vmax=vr[1],
                              shading="gouraud")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(names[j], fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=6)
    fig.suptitle("Wing, held-out geometry: volumetric reconstruction from "
                 f"{args.n_taps} surface pressure taps + {3*args.n_shear} shear gauges "
                 "(midspan slice)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / f"qual_{args.tag}.pdf"); fig.savefig(OUT / f"qual_{args.tag}.png")
    print("wrote", OUT / f"qual_{args.tag}.pdf", flush=True)

    fig2, ax2 = plt.subplots(1, 2, figsize=(8.6, 3.1), dpi=200)
    dq = np.quantile(dist, np.linspace(0, 0.99, 14))
    for j, nm in enumerate(names):
        xsd, ysd, yer = [], [], []
        for a, b in zip(dq[:-1], dq[1:]):
            m_ = (dist >= a) & (dist < b)
            if m_.sum() > 100:
                xsd.append(dist[m_].mean())
                ysd.append(std[m_, j].mean())
                yer.append(np.sqrt(((mean[m_, j] - truth[m_, j]) ** 2).mean()))
        ax2[0].plot(xsd, ysd, "o-", ms=3, lw=1.2, color=f"C{j}", label=nm)
        ax2[0].plot(xsd, yer, "--", lw=1.0, color=f"C{j}", alpha=0.6)
    ax2[0].set_xlabel("distance from wing surface (normalized)")
    ax2[0].set_ylabel("std (solid) / RMS error (dashed)")
    ax2[0].set_title("Uncertainty away from the skin", fontsize=10)
    ax2[0].legend(fontsize=7, frameon=False)

    for j, nm in enumerate(names):
        sd = std[:, j]; ee = np.abs(mean[:, j] - truth[:, j])
        qs = np.quantile(sd, np.linspace(0, 1, 13))
        xs2, ys2 = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            m_ = (sd >= a) & (sd < b)
            if m_.sum() > 100:
                xs2.append(sd[m_].mean()); ys2.append(np.sqrt((ee[m_] ** 2).mean()))
        ax2[1].plot(xs2, ys2, "o-", ms=3, lw=1.2, color=f"C{j}", label=nm)
    lim = max(ax2[1].get_xlim()[1], ax2[1].get_ylim()[1])
    ax2[1].plot([0, lim], [0, lim], "k--", lw=1)
    ax2[1].set_xlabel("predicted std"); ax2[1].set_ylabel("RMS error")
    ax2[1].set_title("Spread reliability", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(OUT / f"uncertainty_{args.tag}.pdf")
    fig2.savefig(OUT / f"uncertainty_{args.tag}.png")
    print("wrote", OUT / f"uncertainty_{args.tag}.pdf", flush=True)

    summ = {"case": int(args.case), "K": args.K,
            "corr_std_err": float(np.corrcoef(std.ravel(),
                                              np.abs(mean - truth).ravel())[0, 1])}
    near = dist <= np.quantile(dist, 0.1); far = dist >= np.quantile(dist, 0.9)
    for j, nm in enumerate(names):
        summ[nm] = {"rmse": float(np.sqrt(((mean[:, j]-truth[:, j])**2).mean())),
                    "std_near_surface": float(std[near, j].mean()),
                    "std_far_field": float(std[far, j].mean())}
    json.dump(summ, open(OUT / f"uncertainty_{args.tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
