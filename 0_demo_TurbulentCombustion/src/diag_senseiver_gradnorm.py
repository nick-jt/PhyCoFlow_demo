"""Does clip_grad_norm_(1.0) actually bind for Senseiver at our scale?

model_baseline.run_epoch_senseiver uses F.mse_loss(..., reduction='mean') and
then clips the global grad norm at 1.0.  Upstream (network_light.py:68,78) uses
reduction='sum' with plain Adam and NO clipping.

Adam is invariant to a constant rescaling of the loss, so 'mean' vs 'sum' is by
itself a no-op.  Clipping is NOT scale invariant, so the two choices interact:
under 'sum' the norm is numel x larger and the clip would saturate on every
step, turning Adam into normalised-gradient Adam -- a materially different
optimiser from upstream's.  Under 'mean' the clip is a no-op IFF the observed
norms sit below 1.0.

This script replicates run_epoch_senseiver exactly but records the PRE-clip
global grad norm, so we can say which combination is observationally identical
to upstream.  It touches no shared code and trains nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import model_baseline as MB
from helpers_baseline import build_sparse_condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = MB.validate_and_normalize_config(MB.load_yaml(MB.ensure_absolute(args.config)))
    stage = MB.resolve_stage_config(cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    MB.set_seed(int(cfg["shared"]["seed"]))

    run_dir = Path(args.out).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    train_set = MB.build_dataset(cfg, split="train", stats_path=run_dir / "diag_stats.pt")
    loader = MB.build_dataloader(train_set, batch_size=int(stage["training"]["batch_size"]),
                                 num_workers=int(cfg["shared"]["data"]["num_workers"]),
                                 shuffle=True)
    adapter = MB.get_baseline_adapter(cfg["baseline_model"])
    bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=run_dir,
                                        train_set=train_set, val_set=train_set)
    model, opt = bundle.model, bundle.optimizer
    cond = cfg["shared"]["conditioning"]
    n_query = int(stage["training"].get("n_query_points", 4096))

    norms_mean, step = [], 0
    model.train(True)
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            coords = batch["coords"].to(device)
            fields = batch["fields"].to(device)
            n_pts = coords.shape[1]
            oc, ov, om, _, ofid = build_sparse_condition(
                coords_full=coords, fields_full=fields,
                cond_fields=cond["cond_fields"],
                n_obs_min=cond["n_obs_min_list"], n_obs_max=cond["n_obs_max_list"])
            idx = torch.randperm(n_pts, device=device)[:n_query]
            pred = model(coords[:, idx], oc, ov, om, ofid)
            loss = F.mse_loss(pred, fields[:, idx])          # mean reduction
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Pre-clip global norm, exactly what clip_grad_norm_ would compare
            # against.  max_norm=inf makes the call a pure measurement.
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")))
            norms_mean.append(gn)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            step += 1
            if step % 10 == 0:
                print(f"[diag] step={step} loss={float(loss):.4e} pre_clip_grad_norm={gn:.4f}",
                      flush=True)

    a = np.asarray(norms_mean)
    numel = int(stage["training"]["batch_size"]) * n_query * train_set.num_fields
    res = {
        "steps": int(a.size),
        "reduction": "mean",
        "loss_numel_per_step": numel,
        "pre_clip_grad_norm": {
            "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "min": float(a.min()),
        },
        "frac_steps_clip_binds_at_1.0_mean_reduction": float((a > 1.0).mean()),
        # Under reduction='sum' every gradient is numel x larger.
        "implied_sum_reduction_norm_median": float(np.median(a) * numel),
        "frac_steps_clip_binds_at_1.0_sum_reduction": float(((a * numel) > 1.0).mean()),
    }
    print("[diag] " + json.dumps(res), flush=True)
    with open(args.out, "w", encoding="utf-8") as h:
        json.dump(res, h, indent=2)


if __name__ == "__main__":
    main()
