"""Paper figures for JHU: per-model midplane reconstructions and uncertainty maps.

K=25 samples per model per snapshot, so the predictive standard deviation is a
usable spatial field rather than an 8-member estimate.

Figure 1  rows = {truth, each model}, cols = {Ux, Uy, Uz, p}, midplane slice,
          each model panel annotated with its per-field relative L2 and CRPS.
Figure 2  rows = models, cols = fields, predictive std at the same slice, with
          sensor locations overlaid on the observed channels.

Only models whose numbers currently stand are included. SiT (invalid sampler,
retraining), Senseiver (Fourier-encoder and residual-MLP defects) and CoNFiLD
(pipeline incomplete) are deliberately absent -- see MODELS below.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

GRID = 125
OUT = Path("../Paper/iclr2027/figures")
FIELDS = ["Ux", "Uy", "Uz", "p"]
OBSERVED = (0, 2)          # sensors on Ux and Uz only


def midplane(a):
    """[N,C] -> [n,n,C] at the central z level."""
    return a.reshape(GRID, GRID, GRID, -1)[:, :, GRID // 2, :]


def rel_l2(p, t):
    return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-12))


def fair_crps_1d(ens, y, chunk=400_000):
    """Fair (unbiased) CRPS for an ensemble of size K over N points."""
    K, N = ens.shape
    out = 0.0
    for s in range(0, N, chunk):
        e = ens[:, s:s + chunk].astype(np.float64)
        yy = y[s:s + chunk].astype(np.float64)
        t1 = np.abs(e - yy[None]).mean(axis=0)
        es = np.sort(e, axis=0)
        w = np.arange(1, K + 1)
        t2 = (2.0 / (K * (K - 1))) * ((2 * w - K - 1)[:, None] * es).sum(axis=0)
        out += (t1 - 0.5 * t2).sum()
    return float(out / N)


def sample_ours(run_glob, snap, K, nfe, device, n_obs):
    import glob
    from ensemble_eval import load_run, sample_ensemble
    from helpers import build_sparse_condition
    rd = sorted(glob.glob(run_glob))[-1]
    model, dataset, _ = load_run(rd, "best.pt", str(device))
    item = dataset[snap]
    coords = item["coords"][None].to(device)
    fields = item["fields"][None].to(device)
    torch.manual_seed(100 + snap)
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords, fields_full=fields, cond_fields=[0, 2],
        n_obs_min=[n_obs], n_obs_max=[n_obs])
    obs = {"coords": oc, "values": ov, "mask": om, "indices": oi, "field_ids": ofid}
    # no clamping: the observation projection must not flatter our panels
    ens = sample_ensemble(model, coords, obs, K=K, n_steps=nfe, chunk=262144,
                          clamp_hard=False, seed=5 + snap).numpy()
    truth = fields[0].cpu().numpy()
    sensors = coords[0, oi[0][om[0].bool()]].cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return ens, truth, sensors, Path(rd).name


def load_baseline_npz(path):
    """Load an ensemble dumped by a baseline's own visualizer (ENSEMBLE_NPZ).

    Reusing the baseline's sampler rather than reimplementing it: the first
    version of this script reimplemented latent-FM's grid conditioning, got the
    build_obs_grid_mask3d signature wrong, and silently dropped the model from
    the figure.
    """
    d = np.load(path, allow_pickle=True)
    return d["ens"].astype(np.float64), d["truth"].astype(np.float64)


def panel_figure(truth, models, sensors, out_path, title):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = 1 + len(models)
    fig, axs = plt.subplots(rows, 4, figsize=(4.0 * 4, 3.7 * rows), squeeze=False)
    ts = midplane(truth)
    for j in range(4):
        lo, hi = (float(v) for v in np.percentile(ts[:, :, j], [1, 99]))
        axs[0, j].imshow(ts[:, :, j].T, origin="lower", cmap="coolwarm",
                         vmin=lo, vmax=hi)
        obs_tag = "observed" if j in OBSERVED else "UNOBSERVED"
        axs[0, j].set_title(f"{FIELDS[j]}  [{obs_tag}]", fontsize=12)
        if j == 0:
            axs[0, j].set_ylabel("ground truth", fontsize=11)
        for r, (name, ens, met) in enumerate(models, start=1):
            m = midplane(ens.mean(0))
            axs[r, j].imshow(m[:, :, j].T, origin="lower", cmap="coolwarm",
                             vmin=lo, vmax=hi)
            axs[r, j].set_title(
                f"relL2 {met[FIELDS[j]]['rel_l2']:.3f}   "
                f"CRPS {met[FIELDS[j]]['crps']:.3f}", fontsize=10)
            if j == 0:
                axs[r, j].set_ylabel(name, fontsize=11)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def uncertainty_figure(models, sensors, out_path, title, K):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = len(models)
    fig, axs = plt.subplots(rows, 4, figsize=(4.0 * 4, 3.7 * rows), squeeze=False)
    for r, (name, ens, met) in enumerate(models):
        sd = midplane(ens.std(0, ddof=1))
        for j in range(4):
            im = axs[r, j].imshow(sd[:, :, j].T, origin="lower", cmap="viridis")
            fig.colorbar(im, ax=axs[r, j], fraction=0.046)
            obs_tag = "observed" if j in OBSERVED else "UNOBSERVED"
            axs[r, j].set_title(f"{FIELDS[j]} [{obs_tag}]  "
                                f"mean sd {sd[:, :, j].mean():.3f}", fontsize=10)
            axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
            if j == 0:
                axs[r, j].set_ylabel(name, fontsize=11)
    fig.suptitle(title + f"   (predictive std over K={K} samples)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=int, default=0)
    p.add_argument("--K", type=int, default=25)
    p.add_argument("--nfe", type=int, default=4)
    p.add_argument("--n-obs", type=int, default=19531)
    args = p.parse_args()
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)

    models, summary = [], {}
    truth = sensors = None

    ens, truth, sensors, tag = sample_ours(
        "../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*",
        args.snapshot, args.K, args.nfe, device, args.n_obs)
    met = {FIELDS[j]: {"rel_l2": rel_l2(ens.mean(0)[:, j], truth[:, j]),
                       "crps": fair_crps_1d(ens[:, :, j], truth[:, j])}
           for j in range(4)}
    models.append((f"LOCUS (ours, NFE {args.nfe})", ens, met)); summary["ours"] = met
    print("[ours]", {k: round(v["rel_l2"], 4) for k, v in met.items()}, flush=True)

    try:
        npz = Path(f"../Paper/iclr2027/figures/ens_latentfm_s{args.snapshot}.npz")
        if not npz.exists():
            raise FileNotFoundError(f"{npz} - run evaluate_Gen_Baseline with "
                                    f"ENSEMBLE_K and ENSEMBLE_NPZ set")
        ens2, t2 = load_baseline_npz(npz)
        met2 = {FIELDS[j]: {"rel_l2": rel_l2(ens2.mean(0)[:, j], t2[:, j]),
                            "crps": fair_crps_1d(ens2[:, :, j], t2[:, j])}
                for j in range(4)}
        models.append(("Latent flow matching", ens2, met2)); summary["latent_fm"] = met2
        print("[latent_fm]", {k: round(v["rel_l2"], 4) for k, v in met2.items()}, flush=True)
    except Exception as exc:
        print(f"[warn] latent_fm panel skipped: {exc}", flush=True)

    tail = f"snapshot {args.snapshot}, {args.n_obs} sensors/field on Ux and Uz, z-midplane"
    panel_figure(truth, models, sensors,
                 OUT / f"paper_jhu_panels_s{args.snapshot}.png",
                 "JHU cross-cube reconstruction  --  " + tail)
    uncertainty_figure(models, sensors,
                       OUT / f"paper_jhu_uncertainty_s{args.snapshot}.png",
                       "JHU predictive uncertainty  --  " + tail, args.K)
    json.dump(summary, open(OUT / f"paper_jhu_panels_s{args.snapshot}.json", "w"), indent=2)
    print("wrote panels + uncertainty + json", flush=True)


if __name__ == "__main__":
    main()
