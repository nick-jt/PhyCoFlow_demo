"""Qualitative reconstruction + uncertainty figures on held-out FireBench.

Wind sensors only (u, v, w at ~1% of points); the thermochemical channels
theta and rho_f are never observed, so their panels show cross-variable
inference and the uncertainty that goes with it. Slices are taken through
the fire front (the plane of maximum truth temperature variance) rather
than the domain midplane, which would cut mostly quiescent air.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

NX, NY, NZ = 152, 126, 192
OUT = Path("../Paper/iclr2027/figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None)
    p.add_argument("--snapshot", type=int, default=3)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--n-obs", type=int, default=36772)
    p.add_argument("--drop-fields", type=int, nargs="*", default=None,
                   help="observed-channel ids to remove (e.g. 1 2 = u-only)")
    p.add_argument("--tag", default="firebench")
    args = p.parse_args()

    import glob
    run_dir = args.run_dir or sorted(glob.glob(
        "../Save_TrainedModel/firebench/pointcloud_ffm/iclr_firebench_v4_DemoN18_*"))[-1]
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)

    from ensemble_eval import load_run, sample_ensemble
    from helpers import build_sparse_condition

    model, dataset, _ = load_run(run_dir, "best.pt", str(device))
    item = dataset[args.snapshot]
    coords = item["coords"][None].to(device)
    fields = item["fields"][None].to(device)
    truth = item["fields"].numpy()
    names = [str(n) for n in dataset.field_names]

    cond_fields = [0, 1, 2]
    torch.manual_seed(1000 + args.snapshot)
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords, fields_full=fields, cond_fields=cond_fields,
        n_obs_min=[args.n_obs], n_obs_max=[args.n_obs])
    if args.drop_fields:
        keep = ~torch.isin(ofid, torch.tensor(args.drop_fields, device=ofid.device))
        om = om * keep.to(om.dtype)
    obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
           "field_ids": ofid}
    ens = sample_ensemble(model, coords, obs, K=args.K, n_steps=args.n_steps,
                          chunk=262144, clamp_hard=False,
                          seed=7 + args.snapshot).numpy()
    mean = ens.mean(0); std = ens.std(0); sample0 = ens[0]
    err = np.abs(mean - truth)

    valid = om[0].bool()
    s_idx = np.unique(oi[0, valid].long().cpu().numpy())
    xyz = item["coords"].numpy()
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(xyz[s_idx]).query(xyz, k=1)

    # slice plane: the y-index whose truth temperature has maximum variance
    th = truth[:, 3].reshape(NX, NY, NZ)
    yc = int(np.argmax(th.var(axis=(0, 2))))
    np.savez_compressed(OUT / f"qual_{args.tag}.npz", truth=truth, mean=mean,
                        std=std, sample0=sample0, dist=dist, yc=yc,
                        names=np.array(names))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sl(a, j):
        return a[:, j].reshape(NX, NY, NZ)[:, yc, :]
    cols = ["truth", "posterior mean", "single sample", "posterior std", "|error|"]
    show = [0, 3, 4]     # u (observed), theta (unobserved), rho_f (unobserved)
    fig, axes = plt.subplots(len(show), 5, figsize=(13.6, 2.2 * len(show)), dpi=200)
    for r, j in enumerate(show):
        t = sl(truth, j); m = sl(mean, j); s1 = sl(sample0, j)
        sd = sl(std, j); e = sl(err, j)
        vmin, vmax = np.percentile(t, [1, 99])
        for c, (arr, cmap, vr) in enumerate([
                (t, "inferno", (vmin, vmax)), (m, "inferno", (vmin, vmax)),
                (s1, "inferno", (vmin, vmax)),
                (sd, "viridis", (0, np.percentile(sd, 99))),
                (e, "magma", (0, np.percentile(e, 99)))]):
            ax = axes[r, c]
            im = ax.imshow(arr.T, origin="lower", cmap=cmap, aspect="auto",
                           vmin=vr[0], vmax=vr[1], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                lbl = " (observed)" if j in cond_fields else " (unobserved)"
                ax.set_ylabel(names[j] + lbl, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    ttl = "wind sensors only" if not args.drop_fields else "single wind component"
    fig.suptitle(f"FireBench held-out snapshot, vertical slice through the fire front "
                 f"({ttl}, {args.K}-member ensemble)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / f"qual_{args.tag}.pdf"); fig.savefig(OUT / f"qual_{args.tag}.png")
    print("wrote", OUT / f"qual_{args.tag}.pdf", flush=True)

    # uncertainty profiles
    fig2, ax2 = plt.subplots(1, 3, figsize=(11.5, 3.1), dpi=200)
    colors = [f"C{i}" for i in range(len(names))]
    for j, nm in enumerate(names):
        sd = std[:, j]; ee = np.abs(mean[:, j] - truth[:, j])
        qs = np.quantile(sd, np.linspace(0, 1, 13))
        xs, ys = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            m_ = (sd >= a) & (sd < b)
            if m_.sum() > 100:
                xs.append(sd[m_].mean()); ys.append(np.sqrt((ee[m_] ** 2).mean()))
        ax2[0].plot(xs, ys, "o-", ms=3, color=colors[j], lw=1.2, label=nm)
    lim = max(ax2[0].get_xlim()[1], ax2[0].get_ylim()[1])
    ax2[0].plot([0, lim], [0, lim], "k--", lw=1)
    ax2[0].set_xlabel("predicted std"); ax2[0].set_ylabel("RMS error")
    ax2[0].set_title("Spread reliability", fontsize=10)
    ax2[0].legend(fontsize=7, frameon=False)

    dq = np.quantile(dist, np.linspace(0, 1, 13))
    for j, nm in enumerate(names):
        xs, ys = [], []
        for a, b in zip(dq[:-1], dq[1:]):
            m_ = (dist >= a) & (dist < b)
            if m_.sum() > 100:
                xs.append(dist[m_].mean()); ys.append(std[m_, j].mean())
        ax2[1].plot(xs, ys, "o-", ms=3, color=colors[j], lw=1.2, label=nm)
    ax2[1].set_xlabel("distance to nearest sensor (normalized)")
    ax2[1].set_ylabel("posterior std")
    ax2[1].set_title("Uncertainty vs sensor distance", fontsize=10)

    # std conditioned on the fire front: std vs local truth temperature
    thv = truth[:, 3]
    tq = np.quantile(thv, np.linspace(0, 1, 13))
    for j, nm in enumerate(names):
        xs, ys = [], []
        for a, b in zip(tq[:-1], tq[1:]):
            m_ = (thv >= a) & (thv < b)
            if m_.sum() > 100:
                xs.append(thv[m_].mean()); ys.append(std[m_, j].mean())
        ax2[2].plot(xs, ys, "o-", ms=3, color=colors[j], lw=1.2, label=nm)
    ax2[2].set_xlabel("local potential temperature (standardized)")
    ax2[2].set_ylabel("posterior std")
    ax2[2].set_title("Uncertainty across the fire front", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(OUT / f"uncertainty_{args.tag}.pdf")
    fig2.savefig(OUT / f"uncertainty_{args.tag}.png")
    print("wrote", OUT / f"uncertainty_{args.tag}.pdf", flush=True)

    summ = {"snapshot": args.snapshot, "K": args.K, "slice_y": yc,
            "corr_std_err": float(np.corrcoef(std.ravel(), err.ravel())[0, 1])}
    for j, nm in enumerate(names):
        summ[nm] = {"rmse": float(np.sqrt(((mean[:, j]-truth[:, j])**2).mean())),
                    "std_mean": float(std[:, j].mean()),
                    "std_hot": float(std[thv >= np.quantile(thv, 0.95), j].mean()),
                    "std_cold": float(std[thv <= np.quantile(thv, 0.5), j].mean())}
    json.dump(summ, open(OUT / f"uncertainty_{args.tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
