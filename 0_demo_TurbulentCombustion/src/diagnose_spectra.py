"""Diagnose the small-scale spectral excess in conditioned channels.

Runs sampling-side tests on one JHU val snapshot (125^3):
  A. reference spectra: truth and a raw RFF prior draw
  B. NFE sweep (euler 4/16/64, heun 16), no obs clamping
  C. sensor-count sweep (1953 vs 19531 per field)
  D. spatial fingerprint: high-pass energy of the sample binned by
     distance to the nearest sensor (imprinting -> energy at sensors)
Prints per-test small-band energy ratios for observed (Ux) and
unobserved (Uy) channels.
"""

import numpy as np
import torch
import json

from ensemble_eval import load_run, sample_ensemble
from helpers import build_sparse_condition

GRID = 125


def band_ratio(field_grid: np.ndarray) -> dict:
    """Shell-averaged spectrum -> energy ratios in 3 bands vs a reference."""
    f = np.fft.fftn(field_grid)
    p = np.abs(f) ** 2
    k = np.fft.fftfreq(GRID) * GRID
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kr = np.sqrt(KX**2 + KY**2 + KZ**2)
    kmax = GRID // 2
    bands = {"large": (0, kmax / 8), "medium": (kmax / 8, kmax / 2),
             "small": (kmax / 2, kmax)}
    return {n: float(p[(kr >= lo) & (kr < hi)].sum())
            for n, (lo, hi) in bands.items()}


def ratios(pred: np.ndarray, true: np.ndarray) -> dict:
    bp, bt = band_ratio(pred), band_ratio(true)
    return {n: bp[n] / bt[n] for n in bp}


def to_grid(x_flat: np.ndarray) -> np.ndarray:
    return x_flat.reshape(GRID, GRID, GRID)


def main():
    device = torch.device("cuda:0")
    runs = {
        "clean(DemoN3,best)": "../Save_TrainedModel/_legacy/ffm_tc_pointcloud_match_lfm_DemoN3_20260713_081738",
        "robust(DemoN5,last)": "../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_robust_DemoN5_20260813_122006",
    }
    model, dataset, cfg = load_run(runs["clean(DemoN3,best)"], "best.pt")
    item = dataset[0]
    coords = item["coords"].unsqueeze(0).to(device)
    fields = item["fields"].unsqueeze(0).to(device)
    true = item["fields"].numpy()

    # A. prior draw spectrum
    torch.manual_seed(0)
    x0 = model.sample_source(coords)[0].cpu().numpy()
    print("A. RFF prior draw vs truth band ratios (ch0):",
          {k: round(v, 3) for k, v in ratios(to_grid(x0[:, 0]), to_grid(true[:, 0])).items()})

    def obs_for(n_obs, seed=1):
        torch.manual_seed(seed)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=[0, 2], n_obs_min=[n_obs], n_obs_max=[n_obs])
        return {"coords": oc, "values": ov, "mask": om, "indices": oi,
                "field_ids": ofid}

    def spectra_report(tag, ens):
        e = ens[0].numpy()
        r_obs = ratios(to_grid(e[:, 0]), to_grid(true[:, 0]))
        r_un = ratios(to_grid(e[:, 1]), to_grid(true[:, 1]))
        print(f"{tag}: Ux(obs) small={r_obs['small']:.2f} med={r_obs['medium']:.2f} | "
              f"Uy(unobs) small={r_un['small']:.2f} med={r_un['medium']:.2f}")
        return e

    # B. NFE sweep at 19531 sensors, no clamping
    obs = obs_for(19531)
    e16 = None
    for nfe in [4, 16, 64]:
        ens = sample_ensemble(model, coords, obs, K=1, n_steps=nfe,
                              chunk=262144, clamp_hard=False, seed=5)
        e = spectra_report(f"B. euler nfe={nfe:3d}, 19531 sensors", ens)
        if nfe == 16:
            e16 = e

    # C. sensor-count sweep at nfe 16
    for n_obs in [1953, 6000]:
        obs_c = obs_for(n_obs)
        ens = sample_ensemble(model, coords, obs_c, K=1, n_steps=16,
                              chunk=262144, clamp_hard=False, seed=5)
        spectra_report(f"C. euler nfe=16, {n_obs:5d} sensors", ens)

    # D. spatial fingerprint: high-pass |energy| vs distance to nearest sensor
    g = to_grid(e16[:, 0])
    f = np.fft.fftn(g)
    k = np.fft.fftfreq(GRID) * GRID
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kr = np.sqrt(KX**2 + KY**2 + KZ**2)
    f[kr < GRID // 4] = 0.0
    hp = np.real(np.fft.ifftn(f)).ravel() ** 2

    oc = obs["coords"][0].cpu()
    om = obs["mask"][0].cpu() > 0
    ofid = obs["field_ids"][0].cpu()
    sens = oc[om & (ofid == 0)]  # Ux sensors only
    cq = item["coords"]
    d = torch.cdist(cq[None].to(device),
                    sens[None].to(device)).amin(dim=-1)[0].cpu().numpy()
    edges = np.quantile(d, np.linspace(0, 1, 8))
    print("D. high-pass energy of Ux sample vs distance-to-nearest-Ux-sensor:")
    for i in range(7):
        m = (d >= edges[i]) & (d < edges[i + 1])
        print(f"   d in [{edges[i]:.4f},{edges[i+1]:.4f}): mean hp energy "
              f"{hp[m].mean():.4e}  (n={m.sum()})")

    # E. robust EMA checkpoint (different conditioning: fields 0-3)
    model2, dataset2, cfg2 = load_run(runs["robust(DemoN5,last)"], "last.pt")
    item2 = dataset2[0]
    coords2 = item2["coords"].unsqueeze(0).to(device)
    fields2 = item2["fields"].unsqueeze(0).to(device)
    true2 = item2["fields"].numpy()
    torch.manual_seed(1)
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords2, fields_full=fields2,
        cond_fields=[0, 1, 2, 3], n_obs_min=[9765], n_obs_max=[9765])
    obs2 = {"coords": oc, "values": ov, "mask": om, "indices": oi,
            "field_ids": ofid}
    ens = sample_ensemble(model2, coords2, obs2, K=1, n_steps=16,
                          chunk=262144, clamp_hard=False, seed=5)
    e = ens[0].numpy()
    r0 = ratios(to_grid(e[:, 0]), to_grid(true2[:, 0]))
    print(f"E. robust EMA ckpt (all-ch obs, noise-trained): Ux small={r0['small']:.2f} "
          f"med={r0['medium']:.2f}")


if __name__ == "__main__":
    main()
