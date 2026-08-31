"""Is the predictive spread information-dependent in space?

For each sensor density, bins the pointwise ensemble spread and |error| by the
distance (in grid cells) to the nearest sensor OF THE SAME CHANNEL, and also
reports the spread exactly AT sensor locations (clamping off). An
information-using posterior must have near-zero spread at a sensor and grow
with distance; a width that is flat in distance is an information-independent
learned conditional.
"""
import json, time
import numpy as np
import torch
from scipy import ndimage

from ensemble_eval import load_run, sample_ensemble
from helpers import build_sparse_condition

RD = ("/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/"
      "0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/"
      "iclr_jhu_xcube_spec02_DemoN29_20260822_140100")
DEV, SEED = "cuda:0", 0
model, dataset, cfg = load_run(RD, "best.pt", DEV)
rng = np.random.default_rng(SEED)
snap_ids = rng.choice(len(dataset), size=50, replace=False)
snaps = [int(s) for s in snap_ids[:3]]

c = dataset.coords_raw.numpy()
order = np.lexsort((c[:, 2], c[:, 1], c[:, 0]))     # point idx -> grid raster pos
side = int(round(dataset.num_points ** (1 / 3)))
inv = np.empty_like(order); inv[order] = np.arange(len(order))   # point idx -> raster
BINS = np.array([0, 0.001, 1.0, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0, 15.0, 1e9])
out = {"snaps": snaps, "bins": BINS.tolist()}

for nobs, nfe in [(1953, 4), (19531, 4), (195312, 4), (19531, 16)]:
    key = f"n{nobs}_nfe{nfe}"
    out[key] = []
    for si, snap in enumerate(snaps):
        it = dataset[snap]
        co = it["coords"].unsqueeze(0).to(DEV)
        fl = it["fields"].unsqueeze(0).to(DEV)
        torch.manual_seed(SEED * 777 + snap)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=co, fields_full=fl, cond_fields=[0, 2],
            n_obs_min=[nobs, nobs], n_obs_max=[nobs, nobs])
        obs = {"coords": oc, "values": ov, "mask": om, "indices": oi, "field_ids": ofid}
        t0 = time.time()
        ens = sample_ensemble(model, co, obs, K=8, n_steps=nfe, chunk=262_144,
                              clamp_hard=False, seed=SEED * 131 + si).numpy()
        dt = time.time() - t0
        tru = fl[0].cpu().numpy()
        std = ens.std(axis=0, ddof=1)
        err = np.abs(ens.mean(axis=0) - tru)
        idx_np = oi[0].cpu().numpy(); fid_np = ofid[0].cpu().numpy()
        m_np = om[0].cpu().numpy() > 0
        rec = {"snap": snap, "seconds": round(dt, 1), "channels": {}}
        for j, nm in enumerate(dataset.field_names):
            if j in (0, 2):
                sel = idx_np[m_np & (fid_np == j)]
                vol = np.ones((side, side, side), dtype=bool)
                vol.flat[inv[sel]] = False
                d = ndimage.distance_transform_edt(vol).ravel()[inv]  # cells
                at_sensor = {"spread": float(std[sel, j].mean()),
                             "abs_err": float(err[sel, j].mean()),
                             "n": int(sel.size)}
            else:
                d = np.zeros(dataset.num_points) - 1.0     # unobserved channel
                at_sensor = None
            prof = []
            for b in range(len(BINS) - 1):
                mm = (d >= BINS[b]) & (d < BINS[b + 1])
                if mm.sum() == 0:
                    prof.append(None); continue
                prof.append({"lo": float(BINS[b]), "n": int(mm.sum()),
                             "spread": float(std[mm, j].mean()),
                             "abs_err": float(err[mm, j].mean())})
            rec["channels"][nm] = {"at_sensor": at_sensor, "profile": prof,
                                   "spread_all": float(np.sqrt((std[:, j] ** 2).mean())),
                                   "rmse_all": float(np.sqrt(((ens.mean(axis=0)[:, j] - tru[:, j]) ** 2).mean()))}
        for nm in ("Ux", "Uz"):
            ch = rec["channels"][nm]
            p = [x for x in ch["profile"] if x]
            print(f"[p2] {key} snap={snap} {nm} sens={ch['at_sensor']['spread']:.4f}/"
                  f"{ch['at_sensor']['abs_err']:.4f} " +
                  " ".join(f"d>={x['lo']:g}:{x['spread']:.3f}/{x['abs_err']:.3f}" for x in p),
                  flush=True)
        out[key].append(rec)
    with open(f"{RD}/Evaluation/calib_probe2.json", "w") as f:
        json.dump(out, f, indent=2)
print("[p2] wrote calib_probe2.json", flush=True)
