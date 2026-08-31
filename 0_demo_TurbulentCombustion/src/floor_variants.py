"""Reconcile the competing 'trivial interpolation' floors on ONE canonical draw.

Three numbers are in circulation for Ux relL2 at 1% sensor density:
  0.210  nearest-sensor fill (this repo, helpers_baseline.nearest_sensor_fill_nodes)
  0.290  cKDTree nearest-neighbour, periodic_kdtree=True
  0.235  inverse-distance weighted, k=8

All variants below run on the SAME canonical sensor draw (helpers.py,
seed*777+snap) and the same query points, so any residual difference is the
method, not the sampling.

Geometry note: each cube spans 0.7609 of the 2*pi JHU domain (12.1%), i.e. it
is a 125^3 SUB-CUBE of the 1024^3 DNS and is NOT periodic.  The periodic
variant is included to quantify the cost of assuming otherwise.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

import model_baseline as MB
from helpers import build_sparse_condition

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--snaps", type=int, default=10)
ap.add_argument("--n-obs", type=int, default=19531)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-query", type=int, default=250_000,
                help="random query subsample; relL2 is a ratio of norms so this is unbiased")
ap.add_argument("--idw-k", type=int, default=8)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = MB.validate_and_normalize_config(MB.load_yaml(MB.ensure_absolute(a.config)))
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ds = MB.build_dataset(cfg, split="val",
                      stats_path=MB.ensure_absolute(cfg["shared"]["paths"]["save_root"]) / "floor_stats.pt")
cond = list(cfg["shared"]["conditioning"]["cond_fields"])
names = list(ds.field_names)[: ds.num_fields]

rng = np.random.default_rng(a.seed)
snap_ids = rng.choice(len(ds), size=min(a.snaps, len(ds)), replace=False)

# Period for the (physically wrong) periodic variant: 125 cells of spacing h,
# in normalized [0,1] coords that is 125/124.
PERIOD = 125.0 / 124.0

def nn_fill(q, oc, ov, periodic, k=1):
    """q [Q,3], oc [M,3], ov [M] -> [Q] (k=1 nearest) or IDW over k neighbours."""
    out = torch.empty(q.shape[0], device=q.device, dtype=q.dtype)
    for s in range(0, q.shape[0], a.chunk):
        qq = q[s:s + a.chunk]
        if periodic:
            d = qq[:, None, :] - oc[None, :, :]
            d = d - PERIOD * torch.round(d / PERIOD)      # minimum image
            dist = d.pow(2).sum(-1).sqrt()
        else:
            dist = torch.cdist(qq, oc)
        if k == 1:
            out[s:s + a.chunk] = ov[dist.argmin(dim=1)]
        else:
            dk, ik = torch.topk(dist, k, dim=1, largest=False)
            w = 1.0 / dk.clamp_min(1e-12)
            out[s:s + a.chunk] = (w * ov[ik]).sum(1) / w.sum(1)
    return out

variants = {
    "nn_nonperiodic":  dict(periodic=False, k=1),
    "nn_periodic":     dict(periodic=True,  k=1),
    "idw8_nonperiodic":dict(periodic=False, k=a.idw_k),
    "idw8_periodic":   dict(periodic=True,  k=a.idw_k),
}
acc = {v: {f: [] for f in names} for v in variants}

for snap in snap_ids:
    item = ds[int(snap)]
    coords = item["coords"].unsqueeze(0).to(dev)
    fields = item["fields"].unsqueeze(0).to(dev)
    torch.manual_seed(a.seed * 777 + int(snap))
    oc_all, ov_all, om, oi, ofid = build_sparse_condition(
        coords_full=coords, fields_full=fields, cond_fields=cond,
        n_obs_min=[a.n_obs] * len(cond), n_obs_max=[a.n_obs] * len(cond))
    g = torch.Generator(device="cpu").manual_seed(12345 + int(snap))
    qidx = torch.randperm(coords.shape[1], generator=g)[: a.n_query].to(dev)
    q = coords[0, qidx]
    valid = om[0].bool()
    for fi, fname in enumerate(names):
        truth = fields[0, qidx, fi]
        sel = valid & (ofid[0] == fi)
        if not sel.any():
            continue                      # unobserved channel: floor is 0 -> relL2 = 1
        oc = oc_all[0, sel]; ov = ov_all[0, sel, 0]
        for vname, kw in variants.items():
            pred = nn_fill(q, oc, ov, **kw)
            r = float(torch.linalg.vector_norm(pred - truth) /
                      (torch.linalg.vector_norm(truth) + 1e-12))
            acc[vname][fname].append(r)
    print(f"[floor] snap {int(snap)} done "
          f"nn={acc['nn_nonperiodic']['Ux'][-1]:.4f} "
          f"nnP={acc['nn_periodic']['Ux'][-1]:.4f} "
          f"idw={acc['idw8_nonperiodic']['Ux'][-1]:.4f}", flush=True)

res = {"snaps": [int(s) for s in snap_ids], "n_obs": a.n_obs, "n_query": a.n_query,
       "seed": a.seed, "period_normalized": PERIOD, "cube_span_frac_of_2pi": 0.1211,
       "variants": {}}
print()
print(f"{'variant':>18} " + " ".join(f"{n:>9}" for n in names))
for v in variants:
    row = {f: (float(np.mean(acc[v][f])) if acc[v][f] else 1.0) for f in names}
    res["variants"][v] = row
    print(f"{v:>18} " + " ".join(f"{row[n]:9.4f}" for n in names))
json.dump(res, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
