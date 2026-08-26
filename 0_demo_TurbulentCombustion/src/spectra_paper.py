"""Paper spectral/turbulence-statistics figure on the honest cross-cube split.

Answers the reviewer point that our anti-latent motivation (small scales are
lost by compression) needs turbulence statistics, not just a band ratio:
  (a) shell-averaged energy spectra, single samples, held-out cube 3;
  (b) longitudinal second-order structure functions S2(r);
  (c) PDFs of the longitudinal velocity gradient (small-scale intermittency).
Models: ours frozen (N29), ours no-spectral-loss (N15), latent FM (DemoN23),
all under the identical sensor protocol and the same snapshots. Ensemble
means are NOT used -- averaging destroys small scales by construction, so
every curve is a single posterior sample (the honest object to compare).
"""
import json
from pathlib import Path

import numpy as np
import torch

GRID = 125
KMAX = 62
SNAPS = (0, 3, 12, 23)
NFE = 4
OUT = Path("../Paper/iclr2027/figures")


def shell_spectrum(g):
    f = np.abs(np.fft.fftn(g)) ** 2
    k = np.fft.fftfreq(GRID) * GRID
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kb = np.round(np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)).astype(int)
    return np.bincount(kb.ravel(), weights=f.ravel())


def struct_fn(u, rs):
    """Longitudinal S2(r) along x for the x-velocity component."""
    g = u.reshape(GRID, GRID, GRID)
    return np.array([float(((np.roll(g, -r, axis=0) - g) ** 2).mean()) for r in rs])


def grad_pdf(u, bins):
    g = u.reshape(GRID, GRID, GRID)
    d = (np.roll(g, -1, axis=0) - g).ravel()
    d = d / (d.std() + 1e-12)
    h, edges = np.histogram(d, bins=bins, range=(-8, 8), density=True)
    return h, 0.5 * (edges[1:] + edges[:-1])


def ours_samples(run_dir, device):
    from ensemble_eval import load_run, sample_ensemble
    from helpers import build_sparse_condition
    model, dataset, _ = load_run(run_dir, "best.pt", str(device))
    out = {}
    for si in SNAPS:
        item = dataset[si]
        coords = item["coords"][None].to(device)
        fields = item["fields"][None].to(device)
        torch.manual_seed(100 + si)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields, cond_fields=[0, 2],
            n_obs_min=[19531], n_obs_max=[19531])
        obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
               "field_ids": ofid}
        ens = sample_ensemble(model, coords, obs, K=1, n_steps=NFE,
                              chunk=262144, clamp_hard=False, seed=5 + si)
        out[si] = (ens[0].numpy(), item["fields"].numpy())
    del model
    torch.cuda.empty_cache()
    return out


def lfm_samples(run_dir, device):
    import yaml
    from model_baseline import (get_baseline_adapter, build_dataset,
                                build_obs_grid_mask3d, grid3d_to_pointcloud,
                                safe_torch_load)
    from helpers import build_sparse_condition
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / "run_config.yaml"))
    ds = build_dataset(cfg, split="val", stats_path=run_dir / "dataset_stats.pt")
    adapter = get_baseline_adapter("latent_fm")
    bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=run_dir,
                                        train_set=ds, val_set=ds)
    adapter.load_checkpoint(bundle, safe_torch_load(run_dir / "best.pt",
                                                    map_location="cpu"))
    out = {}
    for si in SNAPS:
        item = ds[si]
        coords = item["coords"][None].to(device)
        truth = item["fields"][None].to(device)
        torch.manual_seed(100 + si)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=truth, cond_fields=[0, 2],
            n_obs_min=[19531], n_obs_max=[19531])
        gv, gm = build_obs_grid_mask3d(ov, om, ofid, oi, ds.num_fields,
                                       ds.num_points, GRID, GRID, GRID,
                                       GRID, GRID, GRID)
        ix = (oi % GRID).float() / (GRID - 1)
        iy = (oi.div(GRID, rounding_mode="floor") % GRID).float() / (GRID - 1)
        iz = (oi.div(GRID * GRID, rounding_mode="floor")).float() / (GRID - 1)
        cond = {"obs_value_grid": gv, "obs_mask_grid": gm,
                "obs_coords_2d": torch.stack([ix, iy, iz], dim=-1),
                "obs_values": ov, "obs_mask": om, "obs_field_ids": ofid}
        with adapter.evaluation_weights(bundle):
            bundle.model.eval()
            with torch.no_grad():
                rg = bundle.model.sample(cond, n_steps=16, ode_solver="euler")
        out[si] = (grid3d_to_pointcloud(rg, GRID, GRID, GRID)[0].cpu().numpy(),
                   item["fields"].numpy())
    del bundle
    torch.cuda.empty_cache()
    return out


def main():
    import glob
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {
        "ours": sorted(glob.glob("../Save_TrainedModel/JHU/pointcloud_ffm/"
                                 "iclr_jhu_xcube_spec02_DemoN29_*"))[-1],
        "ours_nospec": "../Save_TrainedModel/JHU/pointcloud_ffm/"
                       "iclr_jhu_xcube_aug_DemoN15_20260818_083446",
    }
    samples = {k: ours_samples(v, device) for k, v in runs.items()}
    samples["latent_fm"] = lfm_samples(
        "../Save_TrainedModel/JHU/baseline_latent_fm/"
        "Baseline_latent_fm_Stage2_DemoN23_20260818_153527", device)

    rs = np.arange(1, 33)
    nbins = 80
    res = {"k": list(range(1, KMAX + 1)), "r": rs.tolist()}
    truth_ref = None
    for key in ("truth", "ours", "ours_nospec", "latent_fm"):
        sp, sf, pdfs = [], [], []
        for si in SNAPS:
            src = samples["ours"][si][1] if key == "truth" else samples[key][si][0]
            if key == "truth":
                truth_ref = truth_ref or True
            comp = [src[:, j] for j in (0, 1, 2)]
            sp.append(np.mean([shell_spectrum(c.reshape(GRID, GRID, GRID))[1:KMAX + 1]
                               for c in comp], axis=0))
            sf.append(struct_fn(src[:, 0], rs))
            h, ctr = grad_pdf(src[:, 0], nbins)
            pdfs.append(h)
        res[key] = {"spectrum": np.mean(sp, axis=0).tolist(),
                    "S2": np.mean(sf, axis=0).tolist(),
                    "grad_pdf": np.mean(pdfs, axis=0).tolist()}
    res["grad_bins"] = ctr.tolist()

    # band ratios for the text
    for key in ("ours", "ours_nospec", "latent_fm"):
        s = np.array(res[key]["spectrum"]); t = np.array(res["truth"]["spectrum"])
        print(f"{key}: inertial(k8-31)={s[7:31].sum()/t[7:31].sum():.3f} "
              f"dissip(k32-62)={s[31:62].sum()/t[31:62].sum():.3f}", flush=True)
    json.dump(res, open(OUT / "spectra_stats.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = [("truth", "k-", "DNS truth"),
              ("ours", "C2-", r"\model (ours)".replace("\\model", "ours")),
              ("ours_nospec", "C0-.", "ours, no spectral loss"),
              ("latent_fm", "C1--", "latent FM")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=200)
    ks = np.arange(1, KMAX + 1)
    for key, st, lab in styles:
        axes[0].loglog(ks, res[key]["spectrum"], st, lw=1.5, label=lab)
        axes[1].loglog(rs, res[key]["S2"], st, lw=1.5)
        axes[2].semilogy(res["grad_bins"], res[key]["grad_pdf"], st, lw=1.5)
    axes[0].axvspan(32, KMAX, color="gray", alpha=0.12)
    axes[0].set_xlabel("wavenumber $k$"); axes[0].set_ylabel("$E(k)$")
    axes[0].set_title("Energy spectrum (single samples)", fontsize=10)
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].set_xlabel("separation $r$ (grid units)")
    axes[1].set_ylabel("$S_2(r)$")
    axes[1].set_title("Longitudinal structure function", fontsize=10)
    axes[2].set_xlabel(r"$\partial_x u / \sigma$")
    axes[2].set_ylabel("PDF"); axes[2].set_ylim(1e-5, 1)
    axes[2].set_title("Velocity-gradient PDF", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "spectra_stats.pdf")
    fig.savefig(OUT / "spectra_stats.png")
    print("wrote", OUT / "spectra_stats.pdf", flush=True)


if __name__ == "__main__":
    main()
