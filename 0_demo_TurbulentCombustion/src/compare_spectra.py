"""Shell-spectrum comparison on one JHU val snapshot: truth vs LOCUS vs latent-FM.

Same snapshot (val index 0), same sensor protocol (fields [0,2], 19531 each,
torch seed 1), 16 sampling steps for both generative models. Saves fields and
the comparison figure. Bands are capped at the grid Nyquist (k=62 on 125^3):
super-Nyquist corner shells carry ~zero truth energy and produce spurious
ratios (the earlier "150x excess").
"""

import json
from pathlib import Path

import numpy as np
import torch

GRID = 125
KMAX = GRID // 2  # 62
OUT = Path("../Save_reconstruction_files/spectra_comparison")


def shell_spectrum(g: np.ndarray) -> np.ndarray:
    f = np.abs(np.fft.fftn(g)) ** 2
    k = np.fft.fftfreq(GRID) * GRID
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kb = np.round(np.sqrt(KX**2 + KY**2 + KZ**2)).astype(int)
    return np.bincount(kb.ravel(), weights=f.ravel(), minlength=KMAX + 1)


def main():
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- ours ----------------
    from ensemble_eval import load_run, sample_ensemble
    from helpers import build_sparse_condition

    model, dataset, _ = load_run(
        "../Save_TrainedModel/_legacy/ffm_tc_pointcloud_match_lfm_DemoN3_20260713_081738",
        "best.pt")
    item = dataset[0]
    coords = item["coords"][None].to(device)
    fields = item["fields"][None].to(device)
    truth = item["fields"].numpy()

    torch.manual_seed(1)
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0, 2], n_obs_min=[19531], n_obs_max=[19531])
    obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
           "field_ids": ofid}
    ens = sample_ensemble(model, coords, obs, K=1, n_steps=16,
                          chunk=262144, clamp_hard=False, seed=5)
    ours = ens[0].numpy()
    del model
    torch.cuda.empty_cache()

    # ---------------- latent-FM ----------------
    from model_baseline import (get_baseline_adapter, build_dataset,
                                build_obs_grid_mask3d, grid3d_to_pointcloud,
                                safe_torch_load)
    import yaml

    run_dir = Path("../Save_TrainedModel/JHU/baseline_latent_fm/"
                   "Baseline_latent_fm_Stage2_DemoN0_20260607_034747")
    cfg = yaml.safe_load(open(run_dir / "run_config.yaml"))
    ds2 = build_dataset(cfg, split="val", stats_path=run_dir / "dataset_stats.pt")
    adapter = get_baseline_adapter("latent_fm")
    bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=run_dir,
                                        train_set=ds2, val_set=ds2)
    ckpt = safe_torch_load(run_dir / "best.pt", map_location="cpu")
    adapter.load_checkpoint(bundle, ckpt)

    item2 = ds2[0]
    coords2 = item2["coords"][None].to(device)
    truth2 = item2["fields"][None].to(device)
    assert np.allclose(item2["fields"].numpy(), truth, atol=1e-5), \
        "val snapshot mismatch between pipelines"

    n_fields = ds2.num_fields
    n_pts = ds2.num_points
    torch.manual_seed(1)
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords2, fields_full=truth2,
        cond_fields=[0, 2], n_obs_min=[19531], n_obs_max=[19531])
    grid_v, grid_m = build_obs_grid_mask3d(
        ov, om, ofid, oi, n_fields, n_pts,
        GRID, GRID, GRID, GRID, GRID, GRID)
    ix = (oi % GRID).float() / (GRID - 1)
    iy = (oi.div(GRID, rounding_mode="floor") % GRID).float() / (GRID - 1)
    iz = (oi.div(GRID * GRID, rounding_mode="floor")).float() / (GRID - 1)
    cond_inputs = {
        "obs_value_grid": grid_v, "obs_mask_grid": grid_m,
        "obs_coords_2d": torch.stack([ix, iy, iz], dim=-1),
        "obs_values": ov, "obs_mask": om, "obs_field_ids": ofid,
    }
    with adapter.evaluation_weights(bundle):
        bundle.model.eval()
        with torch.no_grad():
            recon_grid = bundle.model.sample(cond_inputs, n_steps=16,
                                             ode_solver="euler")
    lfm = grid3d_to_pointcloud(recon_grid, GRID, GRID, GRID)[0].cpu().numpy()

    # ---------------- spectra ----------------
    names = dataset.field_names
    results = {}
    for j, nm in enumerate(names):
        st = shell_spectrum(truth[:, j].reshape(GRID, GRID, GRID))
        so = shell_spectrum(ours[:, j].reshape(GRID, GRID, GRID))
        sl = shell_spectrum(lfm[:, j].reshape(GRID, GRID, GRID))
        results[nm] = {"truth": st.tolist(), "ours": so.tolist(),
                       "latent_fm": sl.tolist()}
        def band(s, lo, hi):
            return s[int(lo):int(hi) + 1].sum()
        print(f"{nm}: band ratio ours/truth  "
              f"inertial(k8-31)={band(so,8,31)/band(st,8,31):.3f} "
              f"dissip(k32-62)={band(so,32,62)/band(st,32,62):.3f} | "
              f"latentFM/truth inertial={band(sl,8,31)/band(st,8,31):.3f} "
              f"dissip={band(sl,32,62)/band(st,32,62):.3f}")

    np.savez(OUT / "fields_snapshot0.npz", truth=truth, ours=ours,
             latent_fm=lfm)
    json.dump(results, open(OUT / "shell_spectra.json", "w"))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = np.arange(1, KMAX + 1)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), dpi=150)
    for j, (nm, ax) in enumerate(zip(names, axes)):
        r = results[nm]
        for key, style, lab in [("truth", "k-", "DNS truth"),
                                ("ours", "C2-", "ours (ambient)"),
                                ("latent_fm", "C1--", "latent FM")]:
            ax.loglog(ks, np.array(r[key])[1:KMAX + 1], style, lw=1.4,
                      label=lab)
        ax.axvspan(32, KMAX, color="gray", alpha=0.12)
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel("k")
        if j == 0:
            ax.set_ylabel("E(k)")
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Shell-averaged energy spectra, JHU val snapshot 0 "
                 "(1% sensors on Ux,Uz; 16 steps; shaded = dissipation band)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "spectra_comparison.png", bbox_inches="tight")
    print("wrote", OUT / "spectra_comparison.png")


if __name__ == "__main__":
    main()
