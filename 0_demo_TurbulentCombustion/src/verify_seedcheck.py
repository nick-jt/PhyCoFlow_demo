"""Standalone sensor-draw fingerprint check (no model needed).

Compares the canonical helpers.py draw against the helpers_baseline.py copy to
show they are NOT interchangeable, and prints the fingerprint the coordinator
asked for.
"""
import argparse, torch, numpy as np
import model_baseline as MB
from helpers import build_sparse_condition as canonical
from helpers_baseline import build_sparse_condition as baseline_variant

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--snaps", type=int, nargs="*", default=[29])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-obs", type=int, default=19531)
a = ap.parse_args()

cfg = MB.validate_and_normalize_config(MB.load_yaml(MB.ensure_absolute(a.config)))
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[verify] device={dev}", flush=True)
ds = MB.build_dataset(cfg, split="val",
                      stats_path=MB.ensure_absolute(cfg["shared"]["paths"]["save_root"])
                      / "seedcheck_stats.pt")
cond = list(cfg["shared"]["conditioning"]["cond_fields"])
n = [a.n_obs] * len(cond)

for snap in a.snaps:
    item = ds[int(snap)]
    coords = item["coords"].unsqueeze(0).to(dev)
    fields = item["fields"].unsqueeze(0).to(dev)
    out = {}
    for tag, fn in (("canonical(helpers)", canonical), ("baseline(helpers_baseline)", baseline_variant)):
        torch.manual_seed(a.seed * 777 + int(snap))
        _, _, om, oi, _ = fn(coords_full=coords, fields_full=fields, cond_fields=cond,
                             n_obs_min=n, n_obs_max=n)
        sel = oi[om > 0]
        out[tag] = sel
        if tag.startswith("canonical"):
            print(f"[seedcheck] snap={int(snap)} sensors={int(sel.numel())} "
                  f"idx_sum={int(sel.sum().item())}", flush=True)
        print(f"[verify]   {tag}: sensors={int(sel.numel())} idx_sum={int(sel.sum().item())}",
              flush=True)
    A = set(out["canonical(helpers)"].tolist())
    B = set(out["baseline(helpers_baseline)"].tolist())
    ov = len(A & B)
    print(f"[verify]   index overlap canonical vs baseline = {ov}/{len(A)} "
          f"({100.0*ov/max(len(A),1):.2f}%)", flush=True)
