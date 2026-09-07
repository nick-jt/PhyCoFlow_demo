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

Datasets: originally 2D Kolmogorov; generalized to the cylinder2d fleet.
Each run's own config supplies its data path, so mesh-trained models
(dmfgen / senseiver / mlp_rbf on Cylinder2D_mesh.h5, 23,800 points) and
grid-trained models (sit / s3gm / geofno / latent_fm on Cylinder2D_grid.h5,
400x200) each evaluate on their training discretization; sensor layouts are
bit-identical WITHIN each discretization family (same n_pts, same seeds) and
count-matched (same n_obs) across families -- recorded per-JSON via
data_path / num_points. --stratify-blocks B splits the val block into B
equal sub-blocks with even spacing inside each (cylinder: B=2 for the two
held-out Re trajectories). --cond-fields generalizes the observed-channel
set (cylinder: [0, 1] = Ux, Uy observed; p unobserved = the 2D
identifiability probe, reported per-field).

Deterministic models (--model senseiver | mlp_rbf | geofno): one forward per
snapshot,
scored the way the JHU canonical eval scores deterministic rows
(eval_senseiver_iclr.det_ensemble): the prediction is tiled into TWO identical
ensemble members so ensemble_eval's fair CRPS estimator and ddof=1 spread are
well defined -- the pair term vanishes exactly, so crps == MAE and
spread == 0 -- then the dispersion fields (spread, spread_error_ratio,
coverage_50/90, rank_hist) are set to null in the output because they carry
no information for a point predictor. K is reported as 1.

S3GM (--model s3gm): its own predictor-corrector DPS sampler
(model_baseline.dps_sample) with the RUN CONFIG's sampling params
(sampling_N / snr / n_corrector_steps / alpha_obs -- i.e. the 2D run's own
values, never 3D-tuned defaults; they are recorded in the JSON). ONE
documented deviation from the per-k seeding convention: the K samples are
drawn as a single batched PC chain (shape_5d[0] = K) seeded once with
torch.manual_seed(base * 10_000), because K sequential 1200-net-eval DPS
chains per snapshot would not fit any backfill wall. dps_sample broadcasts
the (1, ...) observation grids over the K-batch, so members differ only in
their noise draws, exactly as sequential chains would.

--resume skips any snapshot whose crps_snapN.json already exists (baselines
only), so a timed-out job can be resubmitted and continues where it stopped.

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
                   choices=["dmfgen", "latent_fm", "sit", "senseiver",
                            "mlp_rbf", "mlprbf", "geofno", "s3gm"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--nfe", type=int, default=None,
                   help="ODE steps. Default: 4 for dmfgen/latent_fm; the run "
                        "config's sampling_N for sit/s3gm (their configured "
                        "benchmark step counts); 1 for deterministic models.")
    p.add_argument("--n-obs-list", type=int, nargs="+", default=[655],
                   help="Sensor counts to evaluate (per conditioned field). "
                        "More than one entry = a sweep (dmfgen only).")
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0],
                   help="Observed field ids (kolm: [0]; cylinder: [0, 1]).")
    p.add_argument("--n-frames", type=int, default=50)
    p.add_argument("--stratify-blocks", type=int, default=1,
                   help="Split the val block into B equal sub-blocks and "
                        "space frames evenly inside each (cylinder: 2, one "
                        "per held-out Re).")
    p.add_argument("--out-prefix", default="kolm_fleet",
                   help="Combined-JSON prefix (kolm_fleet / cyl_fleet).")
    p.add_argument("--resume", action="store_true",
                   help="Skip snapshots whose crps_snapN.json already exists "
                        "(baselines only); their stored metrics are reused.")
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
    # ---- field-dump mode (additive; no behavior change without the flags) ----
    p.add_argument("--dump-frame", type=int, default=None,
                   help="Dump mode: index INTO THE SPLIT of the single frame "
                        "to reconstruct and save to --dump-npz. The canonical "
                        "sensor draw (torch.manual_seed(seed*777+snap)) and, "
                        "when the frame is one of the canonical evenly-spaced "
                        "eval frames, the canonical ensemble-noise seeding "
                        "(base = seed*131 + si) are reused, so the dumped "
                        "sample/mean/std cross-reference the eval JSONs "
                        "exactly. Skips the 50-frame protocol loop entirely.")
    p.add_argument("--dump-npz", default=None,
                   help="Dump mode: output .npz path (required with "
                        "--dump-frame).")
    p.add_argument("--cond-fields", type=int, nargs="+", default=None,
                   help="Dump mode only: observed field indices (default "
                        "[0], the 2D Kolmogorov protocol). --n-obs-list[0] "
                        "is broadcast per conditioned field (e.g. cylinder: "
                        "--cond-fields 0 1 --n-obs-list 238).")
    return p.parse_args()


def require_block_split() -> None:
    if os.environ.get("JHU_SPLIT_MODE") != "block" or \
       os.environ.get("JHU_SPLIT_GAP") != "0":
        raise SystemExit(
            "[guard] JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 must be exported "
            "(protocol: trajectory-holdout val block). Refusing to run.")


def frame_list(n_total: int, n_frames: int, blocks: int = 1) -> list[int]:
    """Evenly spaced frames; with blocks > 1, spaced evenly inside each of
    `blocks` equal sub-blocks (stratified across held-out trajectories)."""
    if blocks <= 1:
        return [(i * n_total) // n_frames for i in range(n_frames)]
    if n_total % blocks or n_frames % blocks:
        raise SystemExit(f"[guard] stratify: {n_total} frames / {n_frames} "
                         f"picks not divisible by {blocks} blocks.")
    span, per = n_total // blocks, n_frames // blocks
    return [b * span + (i * span) // per
            for b in range(blocks) for i in range(per)]


DISPERSION_KEYS = ("spread", "spread_error_ratio", "coverage_50", "coverage_90")

# Deterministic point predictors: single forward, det-tiled scoring,
# dispersion fields nulled, K reported as 1.
DET_MODELS = {"senseiver", "mlp_rbf", "geofno"}


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
    if args.model == "mlprbf":          # accept both spellings
        args.model = "mlp_rbf"
    require_block_split()
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "Evaluation"

    if args.model != "dmfgen" and len(args.n_obs_list) != 1:
        raise SystemExit("[guard] the sensor-count sweep is dmfgen-only.")
    if (args.dump_frame is None) != (args.dump_npz is None):
        raise SystemExit("[guard] --dump-frame and --dump-npz go together.")
    if args.cond_fields is not None and args.dump_frame is None:
        raise SystemExit("[guard] --cond-fields is dump-mode only (the "
                         "protocol loop is fixed to cond_fields=[0]).")
    if args.dump_frame is not None and len(args.n_obs_list) != 1:
        raise SystemExit("[guard] dump mode takes exactly one --n-obs-list "
                         "value (broadcast per conditioned field).")

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
            data_path = cfg["data"]
        else:
            import model_baseline as MB
            cfg = MB.validate_and_normalize_config(
                MB.load_yaml(run_dir / "run_config.yaml"))
            cfg["training_stage"] = 2 if args.model == "latent_fm" else 1
            dataset = MB.build_dataset(cfg, split=args.split,
                                       stats_path=run_dir / "dataset_stats.pt")
            ckpt_file = run_dir / f"{args.ckpt}.pt"
            data_path = cfg["shared"]["paths"]["data_path"]
            if args.model in DET_MODELS:
                nfe = 1  # deterministic: single forward, no ODE
            elif args.nfe is not None:
                nfe = args.nfe
            elif args.model in ("sit", "s3gm"):
                nfe = int(MB.resolve_stage_config(cfg)["sampling"]["sampling_N"])
            else:
                nfe = 4
        n = len(dataset)
        frames = frame_list(n, args.n_frames, args.stratify_blocks)
        print(f"[dry-run] model={args.model} run={run_dir}")
        print(f"[dry-run] ckpt={ckpt_file} exists={ckpt_file.is_file()}")
        print(f"[dry-run] data={data_path}")
        print(f"[dry-run] split={args.split} len={n} "
              f"(expected {args.expect_val_len}) "
              f"fields={list(dataset.field_names)} points={dataset.num_points}")
        print(f"[dry-run] K={args.K} nfe={nfe} n_obs_list={args.n_obs_list} "
              f"cond_fields={args.cond_fields} seed={args.seed} "
              f"stratify_blocks={args.stratify_blocks} prefix={args.out_prefix}")
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

    s3gm_params = None
    if args.model == "dmfgen":
        device = "cuda:0"
        model, dataset, cfg = load_dmfgen(run_dir, args.ckpt, device)
        nfe = args.nfe if args.nfe is not None else 4
        solver = "euler"
        data_path = cfg["data"]
    else:
        MB, adapter, bundle, dataset, cfg, device = load_baseline(
            run_dir, args.ckpt, args.model, args.split)
        import helpers_baseline as HB
        data_path = cfg["shared"]["paths"]["data_path"]
        if args.model in DET_MODELS:
            solver = "none"
            nfe = 1  # deterministic: single forward, no ODE
        elif args.model == "s3gm":
            sampling_cfg = MB.resolve_stage_config(cfg)["sampling"]
            nfe = int(args.nfe if args.nfe is not None
                      else sampling_cfg["sampling_N"])
            solver = "pc_dps"
            s3gm_params = {
                "snr": float(sampling_cfg["snr"]),
                "n_corrector_steps": int(sampling_cfg["n_corrector_steps"]),
                "alpha_obs": float(sampling_cfg["alpha_obs"]),
                "note": ("guidance/sampling params taken from this run's own "
                         "config (2D values), not any 3D-tuned defaults; K "
                         "samples drawn as one batched PC chain seeded with "
                         "torch.manual_seed(base*10000) -- documented "
                         "deviation from per-k seeding"),
            }
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
    frames = frame_list(n_total, args.n_frames, args.stratify_blocks)
    field_names = list(dataset.field_names)
    print(f"[protocol] model={args.model} split={args.split} n={n_total} "
          f"data={data_path} frames={frames}", flush=True)
    print(f"[protocol] K={args.K} nfe={nfe} solver={solver} seed={args.seed} "
          f"op_seed={args.op_seed} n_obs_list={args.n_obs_list} "
          f"cond_fields={args.cond_fields} gpu={gpu_name}", flush=True)

    seeding_note = (
        "sensors: helpers.build_sparse_condition (canonical CPU-randint variant, "
        "NOT helpers_baseline's CUDA-randint variant) under "
        "torch.manual_seed(seed*777+snap); bit-identical across every model "
        "evaluated on the same data file / point count (same GPU SKU), and "
        "count-matched across discretizations. ensemble noise: "
        "base=seed*131+si, torch.manual_seed(base*10000+k) before sample k "
        "(s3gm: one batched chain seeded at base*10000 -- see s3gm_sampling). "
        "Frames are fixed evenly-spaced val indices "
        "(i*len(val)//n_frames, stratified across --stratify-blocks equal "
        "sub-blocks), not ensemble_eval.main()'s rng.choice.")

    def run_protocol(n_obs: int):
        """Evaluate all frames at one sensor count. Returns (per_snap, cost)."""
        per_snap, timings, mems = [], [], []
        fig_dir = out_dir / f"figs_{args.model}_K{args.K}_nfe{nfe}_n{n_obs}"
        for si, snap in enumerate(frames):
            crps_path = out_dir / f"crps_snap{int(snap)}.json"
            if (args.resume and args.model != "dmfgen"
                    and crps_path.exists() and crps_path.stat().st_size > 0):
                m = json.loads(crps_path.read_text())
                per_snap.append(m)
                if m.get("sample_seconds") is not None:
                    timings.append(float(m["sample_seconds"]))
                if m.get("peak_gpu_gb") is not None:
                    mems.append(float(m["peak_gpu_gb"]))
                print(f"[resume] snap={snap} loaded {crps_path.name}", flush=True)
                continue

            item = dataset[int(snap)]
            coords = item["coords"].unsqueeze(0).to(device)
            fields = item["fields"].unsqueeze(0).to(device)

            n_cf = len(args.cond_fields)
            torch.manual_seed(args.seed * 777 + int(snap))
            oc, ov, om, oi, ofid = build_sparse_condition(
                coords_full=coords, fields_full=fields,
                cond_fields=list(args.cond_fields),
                n_obs_min=[n_obs] * n_cf, n_obs_max=[n_obs] * n_cf)
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
            elif args.model in DET_MODELS:
                # Deterministic: one forward (pointwise chunked for
                # senseiver / mlp_rbf, mirroring
                # eval_senseiver_iclr.predict_full_field; dense grid pass for
                # geofno, mirroring visualize_reconstruction_deterministic),
                # scored as two identical members so fair CRPS == MAE and
                # spread == 0.
                with torch.no_grad():
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if args.model == "geofno":
                        gv, gm = HB.build_obs_grid_mask(
                            ov, om, ofid, oi, n_fields, n_pts,
                            num_y, num_x, num_y, num_x,
                            point_to_grid=bundle.components.get("point_to_grid"))
                        pred_grid = bundle.model(gv, gm)
                        pred = HB.grid_to_pointcloud(
                            pred_grid, num_y, num_x,
                            point_to_grid=bundle.components.get("point_to_grid"))
                    else:  # senseiver / mlp_rbf: pointwise
                        n_q = coords.shape[1]
                        pred = torch.empty(1, n_q, n_fields,
                                           device=coords.device,
                                           dtype=coords.dtype)
                        for s in range(0, n_q, args.chunk):
                            e = min(s + args.chunk, n_q)
                            pred[:, s:e] = bundle.model(coords[:, s:e], oc, ov,
                                                        om, ofid)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - t0)
                ens = np.repeat(pred[0].detach().float().cpu().numpy()[None],
                                2, axis=0)
            elif args.model == "s3gm":
                # K samples as ONE batched predictor-corrector DPS chain
                # (documented deviation from per-k seeding; see module
                # docstring). dps_sample broadcasts the (1, ...) obs grids
                # over the K-batch and needs autograd for the DPS gradient.
                gv, gm = HB.build_obs_grid_mask(
                    ov, om, ofid, oi, n_fields, n_pts, num_y, num_x,
                    int(bundle.components["H_pad"]),
                    int(bundle.components["W_pad"]),
                    point_to_grid=bundle.components.get("point_to_grid"))
                torch.manual_seed(base * 10_000)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.enable_grad():
                    grid = MB.dps_sample(
                        net=bundle.model, sde=bundle.components["sde"],
                        shape_5d=(args.K, 1, n_fields,
                                  int(bundle.components["H_pad"]),
                                  int(bundle.components["W_pad"])),
                        obs_value_grid=gv, obs_mask_grid=gm, device=device,
                        N_steps=nfe, snr=s3gm_params["snr"],
                        n_corrector_steps=s3gm_params["n_corrector_steps"],
                        alpha_obs=s3gm_params["alpha_obs"])
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) / args.K)
                recon = HB.grid_to_pointcloud(
                    grid, num_y, num_x,
                    point_to_grid=bundle.components.get("point_to_grid"))
                ens = recon.detach().float().cpu().numpy()
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
            if args.model in DET_MODELS:
                m = null_dispersion(m)
            m["snapshot"] = int(snap)
            m["si"] = si
            m["sensors"] = sensors
            m["idx_sum"] = idx_sum
            m["K"] = 1 if args.model in DET_MODELS else args.K
            m["nfe"] = nfe
            m["n_obs"] = n_obs
            m["sample_seconds"] = timings[-1]
            m["peak_gpu_gb"] = mems[-1]
            per_snap.append(m)

            if args.model != "dmfgen":
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

    # ------------------------------------------------------------------
    # dump mode: one frame -> npz (truth / sample / mean / std / sensors)
    # ------------------------------------------------------------------
    def dump_one(snap: int, n_obs: int, cond_fields: list[int],
                 out_path: Path) -> None:
        """Reconstruct ONE split frame and save the fields to npz.

        Mirrors the per-snapshot body of run_protocol exactly (same canonical
        sensor draw, same ensemble-noise convention) so the dumped fields are
        the very samples the eval JSONs scored -- provided (seed, K, nfe,
        n_obs, cond_fields) match and `snap` is one of the canonical
        evenly-spaced frames (then si = frames.index(snap)).
        """
        snap = int(snap)
        canonical = snap in frames
        si = frames.index(snap) if canonical else snap
        if not canonical:
            print(f"[dump] WARNING: frame {snap} is not in the canonical "
                  f"{len(frames)}-frame list; ensemble noise base uses "
                  f"si={si} (=snap) and does NOT cross-reference the eval "
                  "JSONs.", flush=True)
        item = dataset[snap]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)

        torch.manual_seed(args.seed * 777 + snap)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=list(cond_fields),
            n_obs_min=[n_obs] * len(cond_fields),
            n_obs_max=[n_obs] * len(cond_fields))
        valid = om[0].bool()
        sensors = int(om.sum())
        idx_sum = int(oi[om.bool()].sum())
        print(f"[seedcheck] snap={snap} n_obs={n_obs} "
              f"cond_fields={list(cond_fields)} sensors={sensors} "
              f"idx_sum={idx_sum}", flush=True)

        base = args.seed * 131 + si
        t0 = time.perf_counter()
        if args.model == "dmfgen":
            from ensemble_eval import sample_ensemble
            obs = {"coords": oc, "values": ov, "mask": om,
                   "indices": oi, "field_ids": ofid}
            ens = sample_ensemble(model, coords, obs, K=args.K,
                                  n_steps=nfe, chunk=args.chunk,
                                  clamp_hard=True, seed=base).numpy()
        elif args.model == "senseiver":
            n_q = coords.shape[1]
            with torch.no_grad():
                pred = torch.empty(1, n_q, bundle.model.n_fields,
                                   device=coords.device, dtype=coords.dtype)
                for s in range(0, n_q, args.chunk):
                    e = min(s + args.chunk, n_q)
                    pred[:, s:e] = bundle.model(coords[:, s:e], oc, ov,
                                                om, ofid)
            ens = pred[0].detach().float().cpu().numpy()[None]
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
                    ens.append(_draw()[0].detach().float().cpu().numpy())
                    print(f"[dump] member {k + 1}/{args.K} "
                          f"({time.perf_counter() - t0:.1f}s)", flush=True)
            ens = np.stack(ens, axis=0)
        K_eff = ens.shape[0]
        print(f"[dump] ensemble {ens.shape} in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)

        truth_np = fields[0].detach().float().cpu().numpy()
        # scored exactly like the eval (senseiver: 2 identical members)
        m = ensemble_metrics(np.repeat(ens, 2, axis=0) if K_eff == 1 else ens,
                             truth_np, field_names)
        if args.model == "senseiver":
            m = null_dispersion(m)
        print(f"[dump] snap={snap} " + " ".join(
            f"{k}={v:.5f}" for k, v in m["aggregate"].items()
            if isinstance(v, float)), flush=True)

        pred_mean = ens.mean(axis=0)
        pred_std = (ens.std(axis=0, ddof=1) if K_eff > 1
                    else np.zeros_like(ens[0]))

        ds_mean = getattr(dataset, "mean", None)
        ds_std = getattr(dataset, "std", None)
        try:
            abs_frame = int(np.asarray(dataset.indices)[snap])
        except Exception:
            abs_frame = None
        meta = {
            "protocol": "kolm2d_matched_v1_dump",
            "model": args.model,
            "run_dir": str(run_dir),
            "ckpt": args.ckpt,
            "split": args.split,
            "split_env": {k: os.environ.get(k)
                          for k in ("JHU_SPLIT_MODE", "JHU_SPLIT_GAP")},
            "split_len": n_total,
            "snapshot_index": snap,
            "absolute_frame": abs_frame,
            "canonical_frame": canonical,
            "si": si,
            "seed": args.seed,
            "sensor_seed_formula": f"torch.manual_seed({args.seed}*777+{snap})",
            "sensor_device": str(coords.device),
            "cond_fields": list(cond_fields),
            "n_obs": n_obs,
            "n_sensors": sensors,
            "sensor_idx_sum": idx_sum,
            "K": K_eff,
            "nfe": nfe,
            "ode_solver": solver,
            "noise_seeding": (f"base={args.seed}*131+{si}; "
                              "torch.manual_seed(base*10000+k) per member"
                              if args.model != "dmfgen" else
                              f"sample_ensemble(seed={base})"),
            "deterministic": args.model == "senseiver",
            "units": "STANDARDIZED (z-score with this run's train stats; "
                     "physical = arr * norm_std + norm_mean)",
            "pred_std_ddof": 1,
            "gpu": gpu_name,
            "metrics": m,
            "field_names": field_names,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save = dict(
            truth=truth_np.astype(np.float32),
            pred_mean=pred_mean.astype(np.float32),
            pred_sample=ens[0].astype(np.float32),
            pred_std=pred_std.astype(np.float32),
            coords=item["coords"].numpy().astype(np.float32),
            sensor_indices=oi[0, valid].long().cpu().numpy(),
            sensor_field_ids=ofid[0, valid].long().cpu().numpy(),
            names=np.array(field_names),
            meta=np.array(json.dumps(meta, indent=1)),
        )
        if ds_mean is not None and ds_std is not None:
            save["norm_mean"] = ds_mean.cpu().numpy().ravel().astype(np.float32)
            save["norm_std"] = ds_std.cpu().numpy().ravel().astype(np.float32)
        cr = item.get("coords_raw")
        if cr is not None:
            save["coords_raw"] = cr.numpy().astype(np.float32)
        np.savez_compressed(out_path, **save)
        print(f"[dump] wrote {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    if args.dump_frame is not None:
        cf = args.cond_fields if args.cond_fields is not None else [0]
        if args.model == "dmfgen":
            dump_one(args.dump_frame, args.n_obs_list[0], cf,
                     Path(args.dump_npz))
        else:
            with adapter.evaluation_weights(bundle):
                bundle.model.eval()
                dump_one(args.dump_frame, args.n_obs_list[0], cf,
                         Path(args.dump_npz))
        return

    deterministic = args.model in DET_MODELS
    observed = [field_names[c] for c in args.cond_fields if c < len(field_names)]
    unobserved = [f for f in field_names if f not in observed]
    if len(field_names) == 1:
        field_note = ("single-field dataset: aggregate == per-field values")
    else:
        field_note = (f"multi-field dataset: observed={observed} "
                      f"unobserved={unobserved}; the unobserved per_field "
                      "columns are the 2D identifiability probe -- report "
                      "them alongside the aggregate")

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
            "cond_fields": list(args.cond_fields),
            "n_obs": [n_obs] * len(args.cond_fields),
            "stratify_blocks": args.stratify_blocks,
            "data_path": str(data_path),
            "num_points": int(dataset.num_points),
            "field_names": field_names,
            "observed_fields": observed,
            "unobserved_fields": unobserved,
            "gpu": gpu_name,
            "seeding": seeding_note,
            "field_note": field_note,
            **({"s3gm_sampling": s3gm_params} if s3gm_params else {}),
            **cost,
            "summary": summarize(per_snap, field_names),
            "snapshots": per_snap,
        }

    if args.model == "dmfgen" and len(args.n_obs_list) == 1:
        # Single-density protocol run (e.g. the cylinder fleet): main JSON only.
        n_obs = args.n_obs_list[0]
        per_snap, cost = run_protocol(n_obs)
        payload = payload_common(n_obs, per_snap, cost)
        main_path = out_dir / f"{args.out_prefix}_dmfgen_K{args.K}_nfe{nfe}.json"
        main_path.write_text(json.dumps(payload, indent=1))
        s = payload["summary"]["aggregate"]
        print(f"[RESULT] dmfgen relL2={s['rel_l2_mean']:.5f} "
              f"crps={s['crps']:.5f} "
              f"infer={cost['inference_seconds_per_field_mean']:.3f}s "
              f"peak={cost['inference_peak_gpu_gb']:.2f}GB", flush=True)
        print(f"[out] wrote {main_path}", flush=True)
    elif args.model == "dmfgen":
        rel_l2_by_n = {}
        cost_by_n = {}
        for n_obs in args.n_obs_list:
            per_snap, cost = run_protocol(n_obs)
            payload = payload_common(n_obs, per_snap, cost)
            sweep_path = out_dir / f"sensor_sweep_dmfgen_n{n_obs}.json"
            sweep_path.write_text(json.dumps(payload, indent=1))
            print(f"[out] wrote {sweep_path}", flush=True)
            if n_obs == 655:
                main_path = out_dir / f"{args.out_prefix}_dmfgen_K{args.K}_nfe{nfe}.json"
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
        tag = {"latent_fm": "latentfm", "mlp_rbf": "mlprbf"}.get(
            args.model, args.model)
        if deterministic:
            main_path = out_dir / f"{args.out_prefix}_{tag}_K1.json"
        else:
            main_path = out_dir / f"{args.out_prefix}_{tag}_K{args.K}_nfe{nfe}.json"
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
