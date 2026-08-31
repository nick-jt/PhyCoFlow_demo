"""Honest pure-compute benchmark of the PRODUCTION training step for N29.

Runs the trainer's own run_epoch() over an in-memory loader (data already
resident on the GPU, no I/O, no loader) so the number is pure compute, and
includes every loss term the real step computes -- flow matching AND the
binned spectral loss on the 32^3 block at lambda_sp = 0.02. Reports eager vs
compiled and with vs without the spectral term.
"""
import json, os, sys, time
import numpy as np, torch

import train_pointcloud_ffm as T
from evaluate_ffm import _build_model, _normalize_eval_config
from helpers import TurbulentCombustionH5Dataset

RD = ("/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/"
      "0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/"
      "iclr_jhu_xcube_spec02_DemoN29_20260822_140100")
cfg = json.load(open(f"{RD}/args.json"))
ncfg = _normalize_eval_config(dict(cfg))
dev = torch.device("cuda:0")

ds = TurbulentCombustionH5Dataset(
    cfg["data"], split="train", train_ratio=cfg["train_ratio"],
    field_names=cfg["field_names"], seed=cfg["seed"], time_stride=1,
    stats_path=f"{RD}/dataset_stats.pt")
print(f"[bench] train snapshots={len(ds)} points={ds.num_points} "
      f"batch={cfg['batch_size']} steps/epoch={int(np.ceil(len(ds)/cfg['batch_size']))}", flush=True)

B = cfg["batch_size"]
items = [ds[i] for i in range(B)]
batch_gpu = {"coords": torch.stack([it["coords"] for it in items]).to(dev),
             "fields": torch.stack([it["fields"] for it in items]).to(dev)}
batch_cpu = {"coords": torch.stack([it["coords"] for it in items]).pin_memory(),
             "fields": torch.stack([it["fields"] for it in items]).pin_memory()}
side = int(round(ds.num_points ** (1 / 3)))
GRID = (side, side, side)

def make_model(compile_it):
    torch.manual_seed(cfg["seed"])
    m = _build_model(ncfg, ds).to(dev)
    m.t_sampling = cfg.get("t_sampling", "uniform")
    if compile_it:
        m.model.forward = torch.compile(m.model.forward, mode="max-autotune-no-cudagraphs")
    return m

def bench(model, opt, ema, batch, spectral_w, n_epochs, steps):
    loader = [batch] * steps
    times = []
    for ep in range(n_epochs):
        _, dt, _ = T.run_epoch(
            model=model, loader=loader, optimizer=opt, device=dev,
            cond_fields=cfg["cond_fields"],
            n_obs_min_list=cfg["n_obs_min_list"], n_obs_max_list=cfg["n_obs_max_list"],
            n_query_points=cfg["n_query_points"], query_sampling=cfg["query_sampling"],
            query_sample_near_ratio=cfg["query_sample_near_ratio"],
            query_sample_far_ratio=cfg["query_sample_far_ratio"],
            query_sample_sigma_ratio=cfg["query_sample_sigma_ratio"],
            epoch=ep, use_amp=cfg["use_amp"], ema=ema,
            spectral_weight=spectral_w, spectral_grid_shape=GRID,
            spectral_block=cfg["spectral_block"], spectral_bins=cfg["spectral_bins"],
            spectral_window=cfg.get("spectral_window", False))
        times.append(dt / steps)
        print(f"    epoch {ep}: {dt:.2f}s -> {dt/steps:.4f} s/step", flush=True)
    return times

STEPS = 8
out = {}
for compile_it in (False, True):
    for sw, tag in ((cfg["spectral_weight"], "spectral_on"), (0.0, "spectral_off")):
        for resident, rtag in ((True, "gpu_resident"), (False, "pinned_h2d")):
            if not resident and (sw == 0.0 or not compile_it):
                continue      # H2D variant only for the headline config
            key = f"{'compiled' if compile_it else 'eager'}_{tag}_{rtag}"
            print(f"[bench] === {key}", flush=True)
            m = make_model(compile_it)
            opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            ema = T.EMAWeights(m, cfg["ema_decay"]) if cfg.get("ema_decay", 0) > 0 else None
            n_ep = 8 if compile_it else 5
            ts = bench(m, opt, ema, batch_gpu if resident else batch_cpu, sw, n_ep, STEPS)
            steady = ts[3:] if compile_it else ts[2:]
            out[key] = {"per_step_s_median": float(np.median(steady)),
                        "per_step_s_min": float(np.min(steady)),
                        "all": ts}
            print(f"[bench] {key}: median {np.median(steady):.4f} s/step "
                  f"(min {np.min(steady):.4f})", flush=True)
            del m, opt, ema
            torch.cuda.empty_cache()

n_params = None
print("\n[bench] SUMMARY (48,000 optimizer steps = 6000 epochs x 8 steps)")
for k, v in out.items():
    h = v["per_step_s_median"] * 48000 / 3600
    print(f"  {k:44s} {v['per_step_s_median']:.4f} s/step -> {h:6.2f} compute-h")
json.dump(out, open(f"{RD}/Evaluation/calib_step_bench.json", "w"), indent=2)
print("[bench] wrote calib_step_bench.json", flush=True)
