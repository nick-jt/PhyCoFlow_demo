"""Per-channel single-sample spectra: are unobserved channels spectrally starved?

Separates observed (Ux, Uz) from unobserved (Uy, p) channels, sweeps the
solver step count, and includes the source prior's own spectrum and the
latent-FM baseline. The question is whether a posterior sample for an
unconstrained channel carries turbulent small scales or is essentially a
prior draw.
"""
import json
from pathlib import Path

import numpy as np
import torch

GRID = 125
KMAX = 62
SNAPS = (0, 3, 12, 23)
OUT = Path("../Paper/iclr2027/figures")


from spectral_utils import shell_spectrum


def shell(g):
    # Hann-windowed: the cutout is not periodic, and a raw FFT floor would
    # otherwise dominate every band above k ~ 32 (see spectral_utils).
    return shell_spectrum(g, kmax=KMAX, window=True)


def bands(s, t):
    # inertial k=8-31 and dissipation k=32-45; the upper cut keeps the
    # dissipation band clear of the windowed noise floor.
    return (float(s[7:31].sum() / t[7:31].sum()), float(s[31:45].sum() / t[31:45].sum()))


def main():
    import glob
    device = torch.device("cuda:0")
    from ensemble_eval import load_run, sample_ensemble
    from helpers import build_sparse_condition

    run = sorted(glob.glob("../Save_TrainedModel/JHU/pointcloud_ffm/"
                           "iclr_jhu_xcube_spec02_DemoN29_*"))[-1]
    model, dataset, _ = load_run(run, "best.pt", str(device))
    names = [str(n) for n in dataset.field_names]
    res = {"names": names, "k": list(range(1, KMAX + 1))}

    # --- prior spectrum (what the sampler starts from) ---
    item0 = dataset[SNAPS[0]]
    coords0 = item0["coords"][None].to(device)
    with torch.no_grad():
        x0 = model.sample_source(coords0)[0].cpu().numpy()
    res["prior"] = {nm: shell(x0[:, j].reshape(GRID, GRID, GRID)).tolist()
                    for j, nm in enumerate(names)}

    truth_sp = {nm: [] for nm in names}
    for si in SNAPS:
        tr = dataset[si]["fields"].numpy()
        for j, nm in enumerate(names):
            truth_sp[nm].append(shell(tr[:, j].reshape(GRID, GRID, GRID)))
    res["truth"] = {nm: np.mean(v, axis=0).tolist() for nm, v in truth_sp.items()}

    # --- ours at several step counts ---
    for nfe in (2, 4, 16, 64):
        acc = {nm: [] for nm in names}
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
            e = sample_ensemble(model, coords, obs, K=1, n_steps=nfe,
                                chunk=262144, clamp_hard=False, seed=5 + si)[0].numpy()
            for j, nm in enumerate(names):
                acc[nm].append(shell(e[:, j].reshape(GRID, GRID, GRID)))
        res[f"ours_nfe{nfe}"] = {nm: np.mean(v, axis=0).tolist()
                                 for nm, v in acc.items()}
        print(f"[ours NFE {nfe}]", flush=True)
        for j, nm in enumerate(names):
            i_, d_ = bands(np.array(res[f"ours_nfe{nfe}"][nm]),
                           np.array(res["truth"][nm]))
            tag = "obs " if j in (0, 2) else "UNOBS"
            print(f"   {tag} {nm}: inertial {i_:.3f}  dissip {d_:.3f}", flush=True)
    del model
    torch.cuda.empty_cache()

    # --- latent-FM baseline, per channel ---
    import yaml
    from model_baseline import (get_baseline_adapter, build_dataset,
                                build_obs_grid_mask3d, grid3d_to_pointcloud,
                                safe_torch_load)
    rd = Path("../Save_TrainedModel/JHU/baseline_latent_fm/"
              "Baseline_latent_fm_Stage2_DemoN23_20260818_153527")
    cfg = yaml.safe_load(open(rd / "run_config.yaml"))
    ds = build_dataset(cfg, split="val", stats_path=rd / "dataset_stats.pt")
    ad = get_baseline_adapter("latent_fm")
    bundle = ad.build_for_training(cfg=cfg, device=device, run_dir=rd,
                                   train_set=ds, val_set=ds)
    ad.load_checkpoint(bundle, safe_torch_load(rd / "best.pt", map_location="cpu"))
    acc = {nm: [] for nm in names}
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
        with ad.evaluation_weights(bundle):
            bundle.model.eval()
            with torch.no_grad():
                rg = bundle.model.sample(cond, n_steps=16, ode_solver="euler")
        lf = grid3d_to_pointcloud(rg, GRID, GRID, GRID)[0].cpu().numpy()
        for j, nm in enumerate(names):
            acc[nm].append(shell(lf[:, j].reshape(GRID, GRID, GRID)))
    res["latent_fm"] = {nm: np.mean(v, axis=0).tolist() for nm, v in acc.items()}
    print("[latent-FM]", flush=True)
    for j, nm in enumerate(names):
        i_, d_ = bands(np.array(res["latent_fm"][nm]), np.array(res["truth"][nm]))
        tag = "obs " if j in (0, 2) else "UNOBS"
        print(f"   {tag} {nm}: inertial {i_:.3f}  dissip {d_:.3f}", flush=True)

    # prior band content, for reference
    print("[prior]", flush=True)
    for j, nm in enumerate(names[:1]):
        i_, d_ = bands(np.array(res["prior"][nm]), np.array(res["truth"][nm]))
        print(f"   prior vs truth: inertial {i_:.5f}  dissip {d_:.5f}", flush=True)

    json.dump(res, open(OUT / "sample_spectra_perchannel.json", "w"))
    print("wrote", OUT / "sample_spectra_perchannel.json", flush=True)


if __name__ == "__main__":
    main()
