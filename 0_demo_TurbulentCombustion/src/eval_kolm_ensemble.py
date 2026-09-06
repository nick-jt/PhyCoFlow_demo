#!/usr/bin/env python3
"""Matched-protocol posterior-ensemble evaluation for the 2D Kolmogorov fleet.

One driver for all three trained models (DMF-Gen point-cloud FFM, latent-FM
stage 2, SiT patch-tokenizer), following the post-audit canonical pattern of
eval_latentfm_ensemble.py / eval_sit_ensemble.py rather than the env-gated
ENSEMBLE_K hook in model_baseline.py, because for the 2D fleet the hook path
is unusable for a matched protocol:

  * the hook draws sensors through helpers_baseline.build_sparse_condition,
    which is NOT RNG-equivalent to the canonical helpers.py draw (the baseline
    variant burns a CUDA randint before the randperm; see
    eval_sit_ensemble.py:25-32) -- layouts would differ from ours' eval;
  * nothing seeds the draw per snapshot in the visualize path, so layouts
    would not even be reproducible;
  * for the SiT *patch* tokenizer the hook never fires at all
    (model_baseline.py gates it on `_draw_point_sample is not None`, which is
    only set for the pointnet tokenizer).

This driver instead draws sensors ONCE per (snapshot, n_obs) via the
canonical path -- torch.manual_seed(seed * 777 + snap) immediately before
helpers.build_sparse_condition -- and feeds the identical sensor set to
whichever model it is scoring, so sensor layouts are bit-identical across all
three models (same job, same GPU SKU). Ensemble noise follows the canonical
convention: per-snapshot base = seed * 131 + si (si = position in the frame
list), torch.manual_seed(base * 10_000 + k) before sample k. Metrics are
ensemble_eval.ensemble_metrics, verbatim.

Frames are FIXED and evenly spaced over the val split (block mode, gap 0):
    snap_i = (i * len(val)) // n_frames,  i = 0..n_frames-1
(with len(val) = 640 this is i*12.8 floored: 0, 12, 25, 38, 51, ...).
This replaces ensemble_eval.main()'s rng.choice snapshot selection -- the
random-selection convention is deliberately NOT used here so every model and
every sensor density scores the same frames; everything downstream of the
frame choice keeps the canonical seeding contract.

2D adaptations (vs the 3D/JHU drivers):
  * no 3D grid helpers -- build_obs_grid_mask / grid_to_pointcloud (2D);
  * the JHU canonical fingerprint gate does not apply (different protocol
    tuple); we print + store [seedcheck] sensors/idx_sum per snapshot as the
    2D fingerprint instead;
  * no spectra / clamped z-slab extras; the diagnostic spread figure
    (save_ensemble_figure) already handles 2D and is kept, every --fig-every.

Cost fields: k==0 sample wall-clock (CUDA-synchronized) and peak GPU memory
are recorded per snapshot and summarized in every JSON. For DMF-Gen the K
samples are drawn in one sample_ensemble call, so we record total/K as
seconds per field and note the difference.

Deterministic models (--model senseiver): one chunked forward per snapshot,
scored the way the JHU canonical eval scores deterministic rows
(eval_senseiver_iclr.det_ensemble): the prediction is tiled into TWO identical
ensemble members so ensemble_eval's fair CRPS estimator and ddof=1 spread are
well defined -- the pair term vanishes exactly, so crps == MAE and
spread == 0 -- then the dispersion fields (spread, spread_error_ratio,
coverage_50/90, rank_hist) are set to null in the output because they carry
no information for a point predictor. K is reported as 1.

Touches no shared module.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Matched 2D Kolmogorov fleet ensemble eval")
    p.add_argument("--model", required=True,
                   choices=["dmfgen", "latent_fm", "sit", "senseiver"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--nfe", type=int, default=None,
                   help="ODE steps. Default: 4 for dmfgen/latent_fm; the run "
                        "config's sampling_N for sit (its configured "
                        "benchmark step count).")
    p.add_argument("--n-obs-list", type=int, nargs="+", default=[655],
                   help="Sensor counts to evaluate (per conditioned field). "
                        "More than one entry = a sweep (dmfgen only).")
    p.add_argument("--n-frames", type=int, default=50)
    p.add_argument("--expect-val-len", type=int, default=640,
                   help="Abort if the val split length differs (protocol guard).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op-seed", type=int, default=1000,
                   help="Protocol symmetry only; a no-op here (noise=0, no ops).")
    p.add_argument("--fig-every", type=int, default=10)
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--dry-run", action="store_true",
                   help="CPU-safe: resolve config + dataset + frame list, "
                        "print the plan, touch no GPU, build no model.")
    return p.parse_args()


def require_block_split() -> None:
    if os.environ.get("JHU_SPLIT_MODE") != "block" or \
       os.environ.get("JHU_SPLIT_GAP") != "0":
        raise SystemExit(
            "[guard] JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 must be exported "
            "(protocol: trajectory-holdout val block). Refusing to run.")


def frame_list(n_total: int, n_frames: int) -> list[int]:
    return [(i * n_total) // n_frames for i in range(n_frames)]


DISPERSION_KEYS = ("spread", "spread_error_ratio", "coverage_50", "coverage_90")


def mean_std(per_snap: list[dict], path: list[str]):
    vals = []
    for m in per_snap:
        d = m
        for k in path:
            d = d[k]
        if d is None:            # nulled dispersion field (deterministic model)
            return None, None
        vals.append(float(d))
    return float(np.mean(vals)), float(np.std(vals))


def null_dispersion(m: dict) -> dict:
    """Deterministic row: dispersion metrics carry no information -- null them
    (mirrors how the JHU canonical deterministic evals report only error keys)."""
    for d in [m["aggregate"], *m["per_field"].values()]:
        for k in DISPERSION_KEYS:
            d[k] = None
    m["rank_hist"] = None
    m["deterministic"] = True
    return m


def summarize(per_snap: list[dict], field_names: list[str]) -> dict:
    agg_keys = list(per_snap[0]["aggregate"].keys())
    fld_keys = list(per_snap[0]["per_field"][field_names[0]].keys())
    return {
        "aggregate": {k: mean_std(per_snap, ["aggregate", k])[0] for k in agg_keys},
        "aggregate_std": {k: mean_std(per_snap, ["aggregate", k])[1] for k in agg_keys},
        "per_field": {f: {k: mean_std(per_snap, ["per_field", f, k])[0]
                          for k in fld_keys} for f in field_names},
    }


# ---------------------------------------------------------------------------
# Model loading (per family)
# ---------------------------------------------------------------------------

def load_dmfgen(run_dir: Path, ckpt: str, device: str):
    from ensemble_eval import load_run
    ckpt_name = ckpt if ckpt.endswith(".pt") else f"{ckpt}.pt"
    model, dataset, cfg = load_run(str(run_dir), ckpt_name, device)
    return model, dataset, cfg


def load_baseline(run_dir: Path, ckpt: str, model_name: str, split: str):
    import model_baseline as MB
    cfg = MB.validate_and_normalize_config(MB.load_yaml(run_dir / "run_config.yaml"))
    cfg["baseline_model"] = model_name
    cfg["training_stage"] = 2 if model_name == "latent_fm" else 1
    checkpoint = MB.safe_torch_load(run_dir / f"{ckpt}.pt", map_location="cpu")
    if model_name == "latent_fm" and checkpoint.get("ae_checkpoint"):
        cfg["latent_fm_params"]["stage2"]["stage1_checkpoint"] = checkpoint["ae_checkpoint"]
        print(f"[eval] stage1 ckpt {checkpoint['ae_checkpoint']}", flush=True)
    device = MB.infer_device(None, cfg["shared"]["device_ids"])
    dataset = MB.build_dataset(cfg, split=split, stats_path=run_dir / "dataset_stats.pt")
    adapter = MB.get_baseline_adapter(model_name)
    bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=run_dir,
                                        train_set=dataset, val_set=dataset)
    adapter.load_checkpoint(bundle, checkpoint)
    print(f"[eval] loaded {run_dir.name}/{ckpt}.pt "
          f"(epoch {checkpoint.get('epoch')})", flush=True)
    return MB, adapter, bundle, dataset, cfg, device


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    require_block_split()
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "Evaluation"

    if args.model != "dmfgen" and len(args.n_obs_list) != 1:
        raise SystemExit("[guard] the sensor-count sweep is dmfgen-only.")

    # ---------------- dry run: CPU-safe plan check -------------------------
    if args.dry_run:
        if args.model == "dmfgen":
            from evaluate_ffm import _normalize_eval_config
            from helpers import TurbulentCombustionH5Dataset
            cfg = _normalize_eval_config(json.load(open(run_dir / "args.json")))
            dataset = TurbulentCombustionH5Dataset(
                cfg["data"], split=args.split,
                train_ratio=cfg.get("train_ratio", 0.9),
                field_names=cfg.get("field_names"),
                seed=cfg.get("seed", 42),
                time_stride=cfg.get("time_stride", 1),
                stats_path=str(run_dir / "dataset_stats.pt"))
            ckpt_file = run_dir / (args.ckpt if args.ckpt.endswith(".pt")
                                   else f"{args.ckpt}.pt")
            nfe = args.nfe if args.nfe is not None else 4
        else:
            import model_baseline as MB
            cfg = MB.validate_and_normalize_config(
                MB.load_yaml(run_dir / "run_config.yaml"))
            cfg["training_stage"] = 2 if args.model == "latent_fm" else 1
            dataset = MB.build_dataset(cfg, split=args.split,
                                       stats_path=run_dir / "dataset_stats.pt")
            ckpt_file = run_dir / f"{args.ckpt}.pt"
            if args.model == "senseiver":
                nfe = 1  # deterministic: single forward, no ODE
            elif args.nfe is not None:
                nfe = args.nfe
            elif args.model == "sit":
                nfe = int(MB.resolve_stage_config(cfg)["sampling"]["sampling_N"])
            else:
                nfe = 4
        n = len(dataset)
        frames = frame_list(n, args.n_frames)
        print(f"[dry-run] model={args.model} run={run_dir}")
        print(f"[dry-run] ckpt={ckpt_file} exists={ckpt_file.is_file()}")
        print(f"[dry-run] split={args.split} len={n} "
              f"(expected {args.expect_val_len}) "
              f"fields={list(dataset.field_names)} points={dataset.num_points}")
        print(f"[dry-run] K={args.K} nfe={nfe} n_obs_list={args.n_obs_list} "
              f"seed={args.seed}")
        print(f"[dry-run] frames ({len(frames)}): {frames}")
        print(f"[dry-run] out_dir={out_dir}")
        if n != args.expect_val_len:
            raise SystemExit(f"[dry-run] VAL LENGTH MISMATCH: {n} != "
                             f"{args.expect_val_len}")
        print("[dry-run] OK")
        return

    # ---------------- real run --------------------------------------------
    from ensemble_eval import (ensemble_metrics, require_compute_node,
                               save_ensemble_figure)
    from helpers import build_sparse_condition  # canonical draw -- NOT helpers_baseline
    require_compute_node()
    gpu_name = torch.cuda.get_device_name(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "dmfgen":
        device = "cuda:0"
        model, dataset, cfg = load_dmfgen(run_dir, args.ckpt, device)
        nfe = args.nfe if args.nfe is not None else 4
        solver = "euler"
        draw_ctx = None
    else:
        MB, adapter, bundle, dataset, cfg, device = load_baseline(
            run_dir, args.ckpt, args.model, args.split)
        import helpers_baseline as HB
        if args.model == "senseiver":
            solver = "none"
            nfe = 1  # deterministic: single forward, no ODE
        else:
            sampling_cfg = MB.resolve_stage_config(cfg)["sampling"]
            solver = str(sampling_cfg["ode_solver"])
            if args.nfe is not None:
                nfe = args.nfe
            elif args.model == "sit":
                nfe = int(sampling_cfg["sampling_N"])
            else:
                nfe = 4
        num_x = int(cfg["shared"]["data"]["num_x"])
        num_y = int(cfg["shared"]["data"]["num_y"])
        n_fields = dataset.num_fields
        n_pts = dataset.num_points

    n_total = len(dataset)
    if n_total != args.expect_val_len:
        raise SystemExit(f"[guard] {args.split} split has {n_total} frames, "
                         f"expected {args.expect_val_len}. Wrong split env?")
    frames = frame_list(n_total, args.n_frames)
    field_names = list(dataset.field_names)
    print(f"[protocol] model={args.model} split={args.split} n={n_total} "
          f"frames={frames}", flush=True)
    print(f"[protocol] K={args.K} nfe={nfe} solver={solver} seed={args.seed} "
          f"op_seed={args.op_seed} n_obs_list={args.n_obs_list} gpu={gpu_name}",
          flush=True)

    seeding_note = (
        "sensors: helpers.build_sparse_condition (canonical CPU-randint variant, "
        "NOT helpers_baseline's CUDA-randint variant) under "
        "torch.manual_seed(seed*777+snap); identical across all three models "
        "in this job (same GPU SKU). ensemble noise: base=seed*131+si, "
        "torch.manual_seed(base*10000+k) before sample k. Frames are fixed "
        "evenly-spaced val indices (i*len(val)//n_frames), not "
        "ensemble_eval.main()'s rng.choice.")

    def run_protocol(n_obs: int):
        """Evaluate all frames at one sensor count. Returns (per_snap, cost)."""
        per_snap, timings, mems = [], [], []
        fig_dir = out_dir / f"figs_{args.model}_K{args.K}_nfe{nfe}_n{n_obs}"
        for si, snap in enumerate(frames):
            item = dataset[int(snap)]
            coords = item["coords"].unsqueeze(0).to(device)
            fields = item["fields"].unsqueeze(0).to(device)

            torch.manual_seed(args.seed * 777 + int(snap))
            oc, ov, om, oi, ofid = build_sparse_condition(
                coords_full=coords, fields_full=fields,
                cond_fields=[0], n_obs_min=[n_obs], n_obs_max=[n_obs])
            _ = torch.Generator(device=ov.device).manual_seed(
                args.op_seed + int(snap))  # protocol symmetry; no-op
            sensors = int(om.sum())
            idx_sum = int(oi[om.bool()].sum())
            print(f"[seedcheck] snap={snap} n_obs={n_obs} sensors={sensors} "
                  f"idx_sum={idx_sum}", flush=True)

            base = args.seed * 131 + si
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            if args.model == "dmfgen":
                from ensemble_eval import sample_ensemble
                obs = {"coords": oc, "values": ov, "mask": om,
                       "indices": oi, "field_ids": ofid}
                t0 = time.perf_counter()
                ens = sample_ensemble(model, coords, obs, K=args.K,
                                      n_steps=nfe, chunk=args.chunk,
                                      clamp_hard=True, seed=base)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) / args.K)
                ens = ens.numpy()
            elif args.model == "senseiver":
                # Deterministic: one chunked forward (mirrors
                # eval_senseiver_iclr.predict_full_field), scored as two
                # identical members so fair CRPS == MAE and spread == 0.
                n_q = coords.shape[1]
                with torch.no_grad():
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    pred = torch.empty(1, n_q, bundle.model.n_fields,
                                       device=coords.device, dtype=coords.dtype)
                    for s in range(0, n_q, args.chunk):
                        e = min(s + args.chunk, n_q)
                        pred[:, s:e] = bundle.model(coords[:, s:e], oc, ov,
                                                    om, ofid)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - t0)
                ens = np.repeat(pred[0].detach().float().cpu().numpy()[None],
                                2, axis=0)
            else:
                if args.model == "latent_fm":
                    gv, gm = HB.build_obs_grid_mask(
                        ov, om, ofid, oi, n_fields, n_pts,
                        num_y, num_x, num_y, num_x)
                    cond = {"obs_value_grid": gv, "obs_mask_grid": gm}

                    def _draw():
                        grid = bundle.model.sample(cond, n_steps=nfe,
                                                   ode_solver=solver)
                        return HB.grid_to_pointcloud(grid, num_y, num_x)
                else:  # sit (patch tokenizer, grid path)
                    h_pad = int(bundle.components["H_pad"])
                    w_pad = int(bundle.components["W_pad"])
                    p2g = bundle.components.get("point_to_grid")
                    gv, gm = HB.build_obs_grid_mask(
                        ov, om, ofid, oi, n_fields, n_pts,
                        num_y, num_x, h_pad, w_pad, point_to_grid=p2g)
                    if str(bundle.components["cond_mode"]) == "interp":
                        gv = HB.nearest_fill_grid(gv, gm)

                    def _draw():
                        grid = MB.sit_conditional_sample(
                            net=bundle.model,
                            transport=bundle.components["transport"],
                            shape=(1, n_fields, h_pad, w_pad),
                            obs_value_grid=gv, obs_mask_grid=gm,
                            device=device, n_steps=nfe, sampler_type=solver)
                        return HB.grid_to_pointcloud(grid, num_y, num_x,
                                                     point_to_grid=p2g)

                ens = []
                with torch.no_grad():
                    for k in range(args.K):
                        torch.manual_seed(base * 10_000 + k)
                        if k == 0:
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                        r = _draw()
                        if k == 0:
                            torch.cuda.synchronize()
                            timings.append(time.perf_counter() - t0)
                        ens.append(r[0].detach().float().cpu().numpy())
                ens = np.stack(ens, axis=0)

            mems.append(torch.cuda.max_memory_allocated() / 1024 ** 3)
            truth_np = fields[0].detach().float().cpu().numpy()
            m = ensemble_metrics(ens, truth_np, field_names)
            if args.model == "senseiver":
                m = null_dispersion(m)
            m["snapshot"] = int(snap)
            m["si"] = si
            m["sensors"] = sensors
            m["idx_sum"] = idx_sum
            m["sample_seconds"] = timings[-1]
            m["peak_gpu_gb"] = mems[-1]
            per_snap.append(m)

            if args.model in ("latent_fm", "sit", "senseiver"):
                crps_path = out_dir / f"crps_snap{int(snap)}.json"
                crps_path.write_text(json.dumps(m, indent=1))

            if (not args.no_figs) and (si % max(1, args.fig_every) == 0):
                fig_dir.mkdir(parents=True, exist_ok=True)
                try:
                    cr = item.get("coords_raw")
                    cr = cr.cpu().numpy() if cr is not None else coords[0].cpu().numpy()
                    save_ensemble_figure(
                        ens, truth_np, cr, field_names,
                        fig_dir / f"spread_snap{int(snap)}.png",
                        tag=(f"{args.model} {args.ckpt}  snap {snap}  "
                             f"K={args.K} NFE={nfe} n_obs={n_obs}  "
                             f"relL2={m['aggregate']['rel_l2_mean']:.4f}"))
                except Exception as exc:  # a plot must never kill an eval
                    print(f"  [warn] figure snap {snap} failed: {exc}", flush=True)

            agg = m["aggregate"]
            print(f"[ensemble] snap={snap} n_obs={n_obs} K={args.K} " + " ".join(
                f"{k}={v:.5f}" for k, v in agg.items() if v is not None),
                flush=True)

        cost = {
            "inference_seconds_per_field_mean": float(np.mean(timings)),
            "inference_seconds_per_field_std": float(np.std(timings)),
            "inference_peak_gpu_gb": float(np.max(mems)),
            "timing_note": ("dmfgen: total sample_ensemble wall-clock / K "
                            "(all K samples timed); generative baselines: "
                            "k==0 draw wall-clock; senseiver: the single "
                            "deterministic forward. CUDA-synchronized; "
                            "diagnostic figures excluded."),
        }
        return per_snap, cost

    deterministic = args.model == "senseiver"

    def payload_common(n_obs: int, per_snap: list[dict], cost: dict) -> dict:
        return {
            "protocol": "kolm2d_matched_v1",
            "model": args.model,
            "run_dir": str(run_dir),
            "ckpt": args.ckpt,
            "split": args.split,
            "split_env": {"JHU_SPLIT_MODE": "block", "JHU_SPLIT_GAP": "0"},
            "val_len": n_total,
            "n_frames": len(frames),
            "frame_indices": frames,
            "K": 1 if deterministic else args.K,
            "deterministic": deterministic,
            **({"note_K": "deterministic model; metrics computed with two "
                          "identical members so the fair CRPS estimator is "
                          "well defined: CRPS == MAE exactly, "
                          "rel_l2_single == rel_l2_mean; dispersion fields "
                          "(spread, spread_error_ratio, coverage, rank_hist) "
                          "are null"} if deterministic else {}),
            "nfe": nfe,
            "ode_solver": solver,
            "seed": args.seed,
            "op_seed": args.op_seed,
            "cond_fields": [0],
            "n_obs": [n_obs],
            "field_names": field_names,
            "gpu": gpu_name,
            "seeding": seeding_note,
            "single_field_note": ("single-field dataset (vorticity): "
                                  "aggregate == per-field values"),
            **cost,
            "summary": summarize(per_snap, field_names),
            "snapshots": per_snap,
        }

    if args.model == "dmfgen":
        rel_l2_by_n = {}
        cost_by_n = {}
        for n_obs in args.n_obs_list:
            per_snap, cost = run_protocol(n_obs)
            payload = payload_common(n_obs, per_snap, cost)
            sweep_path = out_dir / f"sensor_sweep_dmfgen_n{n_obs}.json"
            sweep_path.write_text(json.dumps(payload, indent=1))
            print(f"[out] wrote {sweep_path}", flush=True)
            if n_obs == 655:
                main_path = out_dir / f"kolm_fleet_dmfgen_K{args.K}_nfe{nfe}.json"
                main_path.write_text(json.dumps(payload, indent=1))
                print(f"[out] wrote {main_path}", flush=True)
            rel_l2_by_n[str(n_obs)] = payload["summary"]["aggregate"]["rel_l2_mean"]
            cost_by_n[str(n_obs)] = {
                "inference_seconds_per_field_mean":
                    cost["inference_seconds_per_field_mean"],
                "inference_peak_gpu_gb": cost["inference_peak_gpu_gb"],
            }
        combined = {
            "protocol": "kolm2d_matched_v1",
            "model": "dmfgen",
            "run_dir": str(run_dir),
            "ckpt": args.ckpt,
            "K": args.K,
            "nfe": nfe,
            "n_frames": len(frames),
            "frame_indices": frames,
            "seed": args.seed,
            "seeding": seeding_note,
            "metric": "rel_l2_mean (ensemble-mean relative L2, aggregate == "
                      "vorticity, mean over frames)",
            "rel_l2_by_n": rel_l2_by_n,
            "cost_by_n": cost_by_n,
        }
        comb_path = out_dir / "sensor_sweep_dmfgen.json"
        comb_path.write_text(json.dumps(combined, indent=1))
        print(f"[out] wrote {comb_path}", flush=True)
        print(f"[RESULT] dmfgen rel_l2_by_n={rel_l2_by_n}", flush=True)
    else:
        n_obs = args.n_obs_list[0]
        with adapter.evaluation_weights(bundle):
            bundle.model.eval()
            per_snap, cost = run_protocol(n_obs)
        payload = payload_common(n_obs, per_snap, cost)
        if args.model == "senseiver":
            main_path = out_dir / "kolm_fleet_senseiver_K1.json"
        else:
            tag = "latentfm" if args.model == "latent_fm" else "sit"
            main_path = out_dir / f"kolm_fleet_{tag}_K{args.K}_nfe{nfe}.json"
        main_path.write_text(json.dumps(payload, indent=1))
        s = payload["summary"]["aggregate"]
        disp = ("deterministic (dispersion null)" if deterministic else
                f"spread/err={s['spread_error_ratio']:.3f} "
                f"cov90={s['coverage_90']:.3f}")
        print(f"[RESULT] {args.model} relL2={s['rel_l2_mean']:.5f} "
              f"crps={s['crps']:.5f} {disp} "
              f"infer={cost['inference_seconds_per_field_mean']:.3f}s "
              f"peak={cost['inference_peak_gpu_gb']:.2f}GB", flush=True)
        print(f"[out] wrote {main_path} and "
              f"{len(per_snap)} crps_snapN.json files", flush=True)


if __name__ == "__main__":
    main()
