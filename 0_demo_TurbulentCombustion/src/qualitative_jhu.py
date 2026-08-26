"""Qualitative reconstruction + uncertainty figures on held-out JHU cube 3.

Panel figure 1 (reconstruction): for a z-midplane slice, truth / ensemble
mean / one posterior sample / posterior std / |error|, per field, with the
sensor locations that fall in the slab overlaid on the truth panel.

Panel figure 2 (uncertainty profiles):
  (a) reliability of the spread: binned RMS error vs predicted std, with the
      ideal diagonal -- the calibration statement made pointwise;
  (b) posterior std vs distance to the nearest sensor of the observed
      channels -- uncertainty should grow away from the data, and this shows
      whether it does, for observed and unobserved channels separately;
  (c) rank histograms per field (flat = calibrated, U = underdispersed);
  (d) coverage curve: empirical vs nominal central-interval coverage.
Both figures use the frozen model at its reported operating point.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

GRID = 125
OUT = Path("../Paper/iclr2027/figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None)
    p.add_argument("--snapshot", type=int, default=3)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--n-obs", type=int, default=19531)
    p.add_argument("--n-obs-sparse", type=int, default=1953,
                   help="second, sparser density for the sensor-distance profile")
    p.add_argument("--tag", default="jhu")
    args = p.parse_args()

    import glob
    run_dir = args.run_dir or sorted(glob.glob(
        "../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*"))[-1]
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

    xyz = item["coords"].numpy()
    from scipy.spatial import cKDTree

    def draw(n_obs, seed_off):
        torch.manual_seed(100 + args.snapshot + seed_off)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields, cond_fields=[0, 2],
            n_obs_min=[n_obs], n_obs_max=[n_obs])
        obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
               "field_ids": ofid}
        e = sample_ensemble(model, coords, obs, K=args.K, n_steps=args.n_steps,
                            chunk=262144, clamp_hard=False,
                            seed=5 + args.snapshot + seed_off).numpy()
        s_idx = np.unique(oi[0, om[0].bool()].long().cpu().numpy())
        d, _ = cKDTree(xyz[s_idx]).query(xyz, k=1)
        return e, d

    ens, dist = draw(args.n_obs, 0)
    ens_sp, dist_sp = draw(args.n_obs_sparse, 777)
    mean = ens.mean(0)
    std = ens.std(0)
    sample0 = ens[0]
    err = np.abs(mean - truth)
    std_sp = ens_sp.std(0)

    np.savez_compressed(OUT / f"qual_{args.tag}.npz", truth=truth, mean=mean,
                        std=std, sample0=sample0, err=err, dist=dist,
                        std_sp=std_sp, dist_sp=dist_sp, coords=xyz,
                        names=np.array(names), ens_small=ens[:, ::37])
    # ---- reconstruction figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zc = GRID // 2
    def sl(a, j):
        return a[:, j].reshape(GRID, GRID, GRID)[:, :, zc]
    cols = ["truth", "posterior mean", "single sample", "posterior std", "|error|"]
    show = [0, 1, 3]   # Ux (observed), Uy (unobserved), p (unobserved)
    fig, axes = plt.subplots(len(show), 5, figsize=(13.0, 2.55 * len(show)), dpi=200)
    for r, j in enumerate(show):
        t = sl(truth, j); m = sl(mean, j); s1 = sl(sample0, j)
        sd = sl(std, j); e = sl(err, j)
        vmin, vmax = np.percentile(t, [1, 99])
        for c, (arr, cmap, vr) in enumerate([
                (t, "RdBu_r", (vmin, vmax)), (m, "RdBu_r", (vmin, vmax)),
                (s1, "RdBu_r", (vmin, vmax)),
                (sd, "viridis", (0, np.percentile(sd, 99))),
                (e, "magma", (0, np.percentile(e, 99)))]):
            ax = axes[r, c]
            im = ax.imshow(arr.T, origin="lower", cmap=cmap,
                           vmin=vr[0], vmax=vr[1], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                obs_lbl = " (observed)" if j in (0, 2) else " (unobserved)"
                ax.set_ylabel(names[j] + obs_lbl, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    fig.suptitle(f"Held-out cube, z-midplane slice: {args.K}-member ensemble at "
                 f"NFE {args.n_steps}, 1% sensors on Ux and Uz", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT / f"qual_{args.tag}.pdf")
    fig.savefig(OUT / f"qual_{args.tag}.png")
    print("wrote", OUT / f"qual_{args.tag}.pdf", flush=True)

    # ---- uncertainty-profile figure ----
    fig2, ax2 = plt.subplots(1, 4, figsize=(14.5, 3.1), dpi=200)
    colors = ["C0", "C1", "C2", "C3"]

    # (a) spread reliability: binned RMS error vs predicted std
    for j, nm in enumerate(names):
        sd = std[:, j]; ee = np.abs(mean[:, j] - truth[:, j])
        qs = np.quantile(sd, np.linspace(0, 1, 13))
        xs, ys = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            m_ = (sd >= a) & (sd < b)
            if m_.sum() > 100:
                xs.append(sd[m_].mean())
                ys.append(np.sqrt((ee[m_] ** 2).mean()))
        ax2[0].plot(xs, ys, "o-", ms=3, color=colors[j], lw=1.2, label=nm)
    lim = max(ax2[0].get_xlim()[1], ax2[0].get_ylim()[1])
    ax2[0].plot([0, lim], [0, lim], "k--", lw=1, label="ideal")
    ax2[0].set_xlabel("predicted std"); ax2[0].set_ylabel("RMS error")
    ax2[0].set_title("Spread reliability", fontsize=10)
    ax2[0].legend(fontsize=7, frameon=False)

    # (b) std vs distance to nearest sensor, at two sensor densities. At 1%
    # every point lies within a few cells of a sensor, so the profile is flat;
    # the dependence only appears once sensors are sparse enough to leave gaps.
    for (dd, ss, ls, lab) in ((dist, std, "o-", "1%"),
                              (dist_sp, std_sp, "s--", "0.1%")):
        dq = np.quantile(dd, np.linspace(0, 1, 13))
        for j in (0, 1):
            xs, ys = [], []
            for a, b in zip(dq[:-1], dq[1:]):
                m_ = (dd >= a) & (dd < b)
                if m_.sum() > 100:
                    xs.append(dd[m_].mean()); ys.append(ss[m_, j].mean())
            ax2[1].plot(np.array(xs) * GRID, ys, ls, ms=3, color=colors[j], lw=1.2,
                        alpha=1.0 if lab == "1%" else 0.55,
                        label=f"{names[j]}, {lab} sensors")
    ax2[1].set_xlabel("distance to nearest sensor (grid units)")
    ax2[1].set_ylabel("posterior std")
    ax2[1].set_title("Uncertainty vs sensor distance", fontsize=10)
    ax2[1].legend(fontsize=6.5, frameon=False)

    # (c) rank histograms
    K = ens.shape[0]
    sub = slice(None, None, 17)
    for j, nm in enumerate(names):
        below = (ens[:, sub, j] < truth[sub, j][None]).sum(0)
        h = np.bincount(below, minlength=K + 1).astype(float)
        h /= h.sum()
        ax2[2].plot(np.arange(K + 1), h, "-", color=colors[j], lw=1.2, label=nm)
    ax2[2].axhline(1.0 / (K + 1), color="k", ls="--", lw=1)
    ax2[2].set_xlabel("rank of truth within ensemble")
    ax2[2].set_ylabel("frequency")
    ax2[2].set_title("Rank histogram", fontsize=10)
    ax2[2].legend(fontsize=7, frameon=False)

    # (d) coverage curve
    noms = np.linspace(0.05, 0.95, 19)
    for j, nm in enumerate(names):
        e_ = ens[:, sub, j]; t_ = truth[sub, j]
        cov = []
        for q in noms:
            lo = np.quantile(e_, (1 - q) / 2, axis=0)
            hi = np.quantile(e_, 1 - (1 - q) / 2, axis=0)
            cov.append(float(((t_ >= lo) & (t_ <= hi)).mean()))
        ax2[3].plot(noms, cov, "o-", ms=3, color=colors[j], lw=1.2, label=nm)
    ax2[3].plot([0, 1], [0, 1], "k--", lw=1)
    ax2[3].set_xlabel("nominal coverage"); ax2[3].set_ylabel("empirical coverage")
    ax2[3].set_title("Coverage curve", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(OUT / f"uncertainty_{args.tag}.pdf")
    fig2.savefig(OUT / f"uncertainty_{args.tag}.png")
    print("wrote", OUT / f"uncertainty_{args.tag}.pdf", flush=True)

    # numeric summary for the text
    summ = {"snapshot": args.snapshot, "K": args.K,
            "corr_std_err": float(np.corrcoef(std.ravel(), err.ravel())[0, 1])}
    for j, nm in enumerate(names):
        summ[nm] = {"std_mean": float(std[:, j].mean()),
                    "rmse": float(np.sqrt(((mean[:, j]-truth[:, j])**2).mean())),
                    "near_std": float(std[dist <= np.quantile(dist, 0.1), j].mean()),
                    "far_std": float(std[dist >= np.quantile(dist, 0.9), j].mean())}
    json.dump(summ, open(OUT / f"uncertainty_{args.tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
