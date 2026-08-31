"""Per-channel band ratios for the spectral-fix arms: N29 vs N34 vs N35.

N34 gives the Gaussian-process source a k^-5/3 radial density; N35 windows the
spectral term in the training objective. Both were built to move one number:
the fraction of DNS inertial-band energy a sample retains on a channel no
sensor constrains (U_y). Standard metrics do not measure that, so this does.

Same estimator as everywhere else -- Hann-windowed shell spectra, because the
cutouts are sub-blocks of a larger DNS and are not periodic. Ratios computed on
a raw FFT are ratios of two leakage floors and tend to unity for every method.
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch

from spectral_utils import shell_spectrum

GRID = 125
KMAX = 62
SNAPS = (0, 3, 12, 23)
NFE = 4
OUT = Path("../Paper/iclr2027/figures")

ARMS = [
    ("N29 baseline",      "iclr_jhu_xcube_spec02_DemoN29_*"),
    ("N34 k^-5/3 prior",  "iclr_jhu_xcube_kprior_DemoN34_*"),
    ("N35 windowed loss", "iclr_jhu_xcube_specwin_DemoN35_*"),
]


def shell(g):
    return shell_spectrum(g, kmax=KMAX, window=True)


def bands(s, t):
    """Inertial k=8-31, dissipation k=32-45 (upper cut keeps the band clear
    of the windowed noise floor)."""
    return (float(s[7:31].sum() / t[7:31].sum()),
            float(s[31:45].sum() / t[31:45].sum()))


def main():
    device = torch.device("cuda:0")
    from ensemble_eval import load_run, sample_ensemble
    from helpers import build_sparse_condition

    res = {}
    truth_sp = None
    names = None

    for tag, pat in ARMS:
        runs = sorted(glob.glob(f"../Save_TrainedModel/JHU/pointcloud_ffm/{pat}"))
        if not runs:
            print(f"[skip] {tag}: no run dir", flush=True)
            continue
        model, dataset, _ = load_run(runs[-1], "best.pt", str(device))
        if names is None:
            names = [str(n) for n in dataset.field_names]
            acc_t = {nm: [] for nm in names}
            for si in SNAPS:
                tr = dataset[si]["fields"].numpy()
                for j, nm in enumerate(names):
                    acc_t[nm].append(shell(tr[:, j].reshape(GRID, GRID, GRID)))
            truth_sp = {nm: np.mean(v, axis=0) for nm, v in acc_t.items()}
            res["truth"] = {nm: v.tolist() for nm, v in truth_sp.items()}

        # source prior: what an unconstrained channel relaxes toward
        item0 = dataset[SNAPS[0]]
        with torch.no_grad():
            x0 = model.sample_source(item0["coords"][None].to(device))[0].cpu().numpy()
        res[f"{tag} :: prior"] = {
            nm: shell(x0[:, j].reshape(GRID, GRID, GRID)).tolist()
            for j, nm in enumerate(names)}

        acc = {nm: [] for nm in names}
        for si in SNAPS:
            item = dataset[si]
            coords = item["coords"][None].to(device)
            fields = item["fields"][None].to(device)
            torch.manual_seed(100 + si)
            oc, ov, om, oi, ofid = build_sparse_condition(
                coords_full=coords, fields_full=fields, cond_fields=[0, 2],
                n_obs_min=[19531], n_obs_max=[19531])
            obs = {"coords": oc, "values": ov, "mask": om,
                   "indices": oi, "field_ids": ofid}
            e = sample_ensemble(model, coords, obs, K=1, n_steps=NFE,
                                chunk=262144, clamp_hard=False,
                                seed=5 + si)[0].numpy()
            for j, nm in enumerate(names):
                acc[nm].append(shell(e[:, j].reshape(GRID, GRID, GRID)))
        res[tag] = {nm: np.mean(v, axis=0).tolist() for nm, v in acc.items()}

        print(f"\n[{tag}]  NFE {NFE}", flush=True)
        for j, nm in enumerate(names):
            i_, d_ = bands(np.array(res[tag][nm]), truth_sp[nm])
            pi, pd = bands(np.array(res[f"{tag} :: prior"][nm]), truth_sp[nm])
            kind = "obs  " if j in (0, 2) else "UNOBS"
            print(f"   {kind} {nm:3s}  inertial {i_:.4f}  dissip {d_:.4f}"
                  f"   (prior: {pi:.5f} / {pd:.5f})", flush=True)
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "spectra_arms.json", "w"))
    print(f"\nwritten -> {OUT / 'spectra_arms.json'}", flush=True)


if __name__ == "__main__":
    main()
