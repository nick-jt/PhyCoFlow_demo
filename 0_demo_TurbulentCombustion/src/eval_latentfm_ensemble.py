#!/usr/bin/env python3
"""Seeded posterior-ensemble evaluation for the latent flow-matching baseline.

WHY THIS FILE EXISTS
--------------------
`ensemble_eval.py`'s CLI driver (`main()`) cannot evaluate a latent-FM run:
its `load_run()` reads `<run>/args.json` (baseline runs write
`run_config.yaml`), builds a PointCloudFFM, and its `sample_ensemble()` calls
`model.sample_source(...)`, none of which exists on `LatentFlowMatching`.
Only its *metric function*, `ensemble_eval.ensemble_metrics`, is
model-agnostic, and that is what the shared protocol pins.

This driver therefore reproduces `ensemble_eval.main()`'s SEEDING SEMANTICS
exactly, so sensor layouts and snapshot selection are bit-identical to the
canonical path, while driving latent FM's own sampler:

  snapshot selection : rng = np.random.default_rng(seed)
                       snap_ids = rng.choice(len(dataset), n_snapshots, replace=False)
  sensor layout      : torch.manual_seed(seed * 777 + snap)  immediately before
                       build_sparse_condition(...)
  ensemble noise     : per-snapshot base = seed * 131 + si, then
                       torch.manual_seed(base * 10_000 + k) before sample k
                       (ensemble_eval.py:95 receives base as its `seed` arg)

IMPORTANT -- it imports `build_sparse_condition` from **helpers.py**, the same
implementation `ensemble_eval.py` uses, NOT the `helpers_baseline.py` variant
that `model_baseline.py` binds. The two are not RNG-equivalent: the baseline
variant draws its per-field count with `torch.randint(..., device=device)`
(CUDA generator) whereas helpers.py draws it on CPU, so the following
`torch.randperm(n_pts, device=cuda)` starts from a different CUDA RNG state
and yields a DIFFERENT sensor set for the same seed (~2% index overlap; see
eval_sit_ensemble.py:26-32). Importing the canonical one is what makes the
layouts actually match across every model in the paper.

FINGERPRINT GATE
----------------
torch.randperm on CUDA is not portable across GPU SKU (H100 PCIe on the login
node vs H100 SXM on compute nodes give different layouts for the same seed),
so this script (a) refuses to run on a PCIe/login GPU and (b) verifies, before
any sampling, that the canonical protocol (--seed 0 --n-obs 19531 19531
--cond-fields 0 2) reproduces the shared fingerprint on snapshot 29:

    sensors=39062  idx_sum=37987162596

and aborts on mismatch. The fingerprint pre-check is RNG-safe: every
snapshot's sensor draw re-seeds `torch.manual_seed(seed*777+snap)` immediately
before its own `build_sparse_condition`, so an extra draw beforehand cannot
perturb the canonical stream.

FIDELITY FIXES (lfm_fixes)
--------------------------
The fixed run (job 16997534) trains with `cond_mode: image_norm` and
`latent_scale_mode: per_channel`, both installed by monkey-patch from
`lfm_fixes.py` (kept out of the shared checkout while five agents edit
model_baseline.py concurrently; see its docstring). A model built WITHOUT
those patches cannot even be constructed from that run's config (`image_norm`
is rejected by the vanilla class), so this driver auto-imports `lfm_fixes`
whenever the run config requires it (search path: $LFM_FIXES_DIR, then the
training job's scratch dir, then sys.path). For shipped-config runs
(`cond_mode: image`) the patches are numerically inert no-ops; the
`latent_scale` tensor is restored from the checkpoint by the patched
`load_checkpoint`, never recomputed at eval time.

SPLIT REPRODUCTION
------------------
The cross-cube block split is controlled by env vars, exactly as in training
(job 16997534): JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0. Set them in the
launcher; JHU_AUGMENT only affects the train split and is irrelevant here.

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

# Canonical fingerprint (shared across every baseline eval in the paper).
FP_SNAP = 29
FP_SENSORS = 39062
FP_IDX_SUM = 37987162596

JHU_FIELDS = ("Ux", "Uy", "Uz", "p")

# The live checkout's src (the run dirs and the current model_baseline.py live
# with it). Used only if this file is executed from a directory that does not
# itself contain model_baseline.py (e.g. a scratch execution copy).
REAL_SRC = ("/home/ntricard/generative_reconstruction/temp/"
            "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src")

# Where the training job keeps lfm_fixes.py (monkey-patched fidelity fixes).
DEFAULT_FIXES_DIR = "/home/ntricard/.claude/jobs/3ac3fd02/tmp"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Seeded canonical ensemble eval for the latent-FM baseline")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best",
                   help="best | last | a file under <run-dir> or <run-dir>/ckpt_window "
                        "(e.g. epoch_10750.pt) | an absolute checkpoint path")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0, 2])
    p.add_argument("--n-obs", type=int, nargs="+", default=[19531, 19531])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op-seed", type=int, default=1000,
                   help="Kept for protocol symmetry; a no-op here (noise=0, no ops).")
    p.add_argument("--nfe", type=int, default=4,
                   help="ODE steps. 4 = the paper's matched flow-model setting.")
    p.add_argument("--ode-solver", default=None,
                   help="Default: the run config's sampling.ode_solver (euler).")
    p.add_argument("--fig-every", type=int, default=10)
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--out", default=None,
                   help="Default <run-dir>/Evaluation/lfm_canonical_<tag>_K<K>_nfe<N>.json")
    p.add_argument("--allow-any-gpu", action="store_true",
                   help="TESTING ONLY: skip the compute-node guard. The fingerprint "
                        "gate still aborts on a non-canonical layout.")
    return p.parse_args()


def assert_compute_node(allow_any_gpu: bool) -> str:
    """Refuse H100-PCIe / login-node execution (same rule as eval_deeponet_iclr).

    torch.randperm on CUDA differs between H100 PCIe (login) and H100 SXM
    (compute), so a login-node run would silently score a different sensor
    layout than every other model in the paper.
    """
    if not torch.cuda.is_available():
        raise SystemExit("[guard] CUDA is required; refusing CPU execution.")
    name = torch.cuda.get_device_name(0)
    if allow_any_gpu:
        print(f"[guard] OVERRIDDEN (--allow-any-gpu) on {name}", flush=True)
        return name
    if "PCIe" in name:
        raise SystemExit(
            f"[guard] refusing to run on {name!r}: this is the login-node GPU SKU. "
            "torch.randperm(CUDA) is not portable across SKUs, so the canonical "
            "sensor fingerprint cannot be reproduced here. Submit to a compute node."
        )
    print(f"[guard] compute-node GPU OK: {name}", flush=True)
    return name


def bind_src_dir() -> Path:
    """Put a src dir containing model_baseline.py at the front of sys.path."""
    cands = [os.environ.get("LFM_SRC_DIR"),
             str(Path(__file__).resolve().parent),
             REAL_SRC]
    for c in cands:
        if c and Path(c, "model_baseline.py").is_file():
            if c not in sys.path:
                sys.path.insert(0, c)
            os.chdir(c)
            print(f"[src] using {c}", flush=True)
            return Path(c)
    raise SystemExit(f"[src] model_baseline.py not found in any of {cands}")


def resolve_ckpt(run_dir: Path, spec: str) -> Path:
    cand = [Path(spec)] if os.path.isabs(spec) else []
    cand += [run_dir / spec, run_dir / "ckpt_window" / spec]
    if spec in ("best", "last"):
        cand.append(run_dir / f"{spec}.pt")
    for c in cand:
        if c.is_file():
            return c
    raise SystemExit(f"[ckpt] no checkpoint found for --ckpt {spec!r}; tried "
                     + ", ".join(str(c) for c in cand))


def maybe_import_fixes(cfg: dict) -> bool:
    """Import lfm_fixes when the run config's stage-2 settings require it."""
    s2 = cfg.get("latent_fm_params", {}).get("stage2", {})
    cond_mode = str(s2.get("conditioning", {}).get("cond_mode", "image"))
    scale_mode = str(s2.get("architecture", {}).get("latent_scale_mode", "none"))
    needed = cond_mode == "image_norm" or scale_mode not in ("none", "None")
    for d in (os.environ.get("LFM_FIXES_DIR"), DEFAULT_FIXES_DIR):
        if d and d not in sys.path and Path(d, "lfm_fixes.py").is_file():
            sys.path.append(d)
    try:
        import lfm_fixes  # noqa: F401  (installs the patches on import)
        print(f"[fixes] lfm_fixes imported (cond_mode={cond_mode}, "
              f"latent_scale_mode={scale_mode})", flush=True)
        return True
    except ImportError:
        if needed:
            raise SystemExit(
                f"[fixes] run config needs lfm_fixes (cond_mode={cond_mode}, "
                f"latent_scale_mode={scale_mode}) but the module was not found. "
                "Set LFM_FIXES_DIR to the directory containing lfm_fixes.py."
            )
        print("[fixes] lfm_fixes not found; shipped-config run, proceeding without.",
              flush=True)
        return False


def main() -> None:
    args = parse_args()
    gpu_name = assert_compute_node(args.allow_any_gpu)
    bind_src_dir()

    import model_baseline as MB  # noqa: E402

    run_dir = MB.ensure_absolute(args.run_dir)
    cfg = MB.validate_and_normalize_config(MB.load_yaml(run_dir / "run_config.yaml"))
    cfg["training_stage"] = 2
    if not cfg["shared"]["data"].get("field_names"):
        # Older run configs have field_names: null and fall back to combustion
        # labels (CH4/CO/T/U_1) -- positional, right values, wrong keys.
        cfg["shared"]["data"]["field_names"] = list(JHU_FIELDS)

    patched = maybe_import_fixes(cfg)

    # Canonical sensor draw + canonical metric schema. build_sparse_condition
    # MUST come from helpers.py (see module docstring).
    from helpers import build_sparse_condition  # noqa: E402
    from ensemble_eval import ensemble_metrics, save_ensemble_figure  # noqa: E402

    device = MB.infer_device(None, cfg["shared"]["device_ids"])
    ckpt_path = resolve_ckpt(run_dir, args.ckpt)
    ckpt = MB.safe_torch_load(ckpt_path, map_location=device)
    tag = ckpt_path.stem  # best / last / epoch_10750

    # Bind the exact stage-1 autoencoder this checkpoint trained against,
    # rather than letting find_latest_run_dir guess.
    if ckpt.get("ae_checkpoint"):
        cfg["latent_fm_params"]["stage2"]["stage1_checkpoint"] = ckpt["ae_checkpoint"]
        print(f"[eval] stage1 ckpt {ckpt['ae_checkpoint']}", flush=True)

    stats_path = run_dir / "dataset_stats.pt"
    dataset = MB.build_dataset(cfg, split=args.split, stats_path=stats_path)
    field_names = list(dataset.field_names)
    print(f"[eval] split={args.split} n={len(dataset)} fields={field_names} "
          f"ckpt={ckpt_path} epoch={ckpt.get('epoch')} "
          f"val_loss={ckpt.get('val_loss')}", flush=True)

    adapter = MB.get_baseline_adapter("latent_fm")
    bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=run_dir,
                                        train_set=dataset, val_set=dataset)
    adapter.load_checkpoint(bundle, ckpt)
    model = bundle.model

    sampling_cfg = MB.resolve_stage_config(cfg)["sampling"]
    ode_solver = str(args.ode_solver or sampling_cfg["ode_solver"])

    vn_params = sum(p.numel() for p in bundle.components["velocity_net"].parameters())
    ae_params = sum(p.numel() for p in model.ae.parameters())
    print(f"[params] ae={ae_params} velocity_net={vn_params} "
          f"combined={ae_params + vn_params}", flush=True)

    num_x = int(cfg["shared"]["data"]["num_x"])
    num_y = int(cfg["shared"]["data"]["num_y"])
    num_z = int(cfg["shared"]["data"]["num_z"])
    n_pts = num_x * num_y * num_z
    n_fields = dataset.num_fields

    with torch.no_grad():
        z_probe = model.ae.encode(torch.zeros(1, n_fields, num_z, num_y, num_x,
                                              device=device))
    latent_shape = tuple(z_probe.shape[1:])
    latent_scalars = int(np.prod(latent_shape))
    print(f"[bottleneck] latent {latent_shape} = {latent_scalars} scalars; "
          f"compression {n_fields * n_pts / latent_scalars:.2f}x", flush=True)

    def draw_condition(snap: int):
        """Canonical per-snapshot sensor draw (bit-identical to ensemble_eval)."""
        item = dataset[snap]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)
        torch.manual_seed(args.seed * 777 + snap)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=list(args.cond_fields),
            n_obs_min=list(args.n_obs), n_obs_max=list(args.n_obs),
        )
        # op_seed generator: a no-op for this protocol (noise=0, no occlusion,
        # no field dropout) but constructed so the contract is explicit.
        _ = torch.Generator(device=ov.device).manual_seed(args.op_seed + snap)
        return item, coords, fields, oc, ov, om, oi, ofid

    # --- fingerprint self-check, BEFORE any sampling --------------------------
    canonical = (args.seed == 0 and list(args.n_obs) == [19531, 19531]
                 and list(args.cond_fields) == [0, 2])
    if canonical:
        if len(dataset) <= FP_SNAP:
            raise SystemExit(f"[fingerprint] split has only {len(dataset)} snapshots; "
                             f"cannot check snapshot {FP_SNAP}.")
        _, _, _, _, _, om, oi, _ = draw_condition(FP_SNAP)
        got_sensors = int(om.sum())
        got_idx_sum = int(oi[om.bool()].sum())
        ok = got_sensors == FP_SENSORS and got_idx_sum == FP_IDX_SUM
        print(f"[fingerprint] snap={FP_SNAP} sensors={got_sensors} "
              f"idx_sum={got_idx_sum} expected sensors={FP_SENSORS} "
              f"idx_sum={FP_IDX_SUM} -> {'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            raise SystemExit(
                "[fingerprint] ABORT: canonical sensor layout NOT reproduced. "
                "Wrong GPU SKU, wrong build_sparse_condition import, or a "
                "changed RNG contract. Scores would not be comparable.")
    else:
        print("[fingerprint] non-canonical protocol args; fingerprint gate skipped "
              "(output will still be labelled with the actual args).", flush=True)

    # --- canonical snapshot selection (identical to ensemble_eval.main) -------
    rng = np.random.default_rng(args.seed)
    snap_ids = rng.choice(len(dataset), size=min(args.n_snapshots, len(dataset)),
                          replace=False)
    print(f"[seedcheck] n_dataset={len(dataset)} n_snapshots={len(snap_ids)} "
          f"seed={args.seed} op_seed={args.op_seed} K={args.K} nfe={args.nfe} "
          f"solver={ode_solver} cond_fields={args.cond_fields} n_obs={args.n_obs}",
          flush=True)

    out_dir = run_dir / "Evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (Path(args.out) if args.out
                else out_dir / f"lfm_canonical_{tag}_K{args.K}_nfe{args.nfe}.json")
    fig_dir = out_dir / f"figs_canonical_{tag}"

    per_snap, timings, mems = [], [], []
    with adapter.evaluation_weights(bundle):
        model.eval()
        for si, snap in enumerate(snap_ids):
            snap = int(snap)
            item, coords, fields, oc, ov, om, oi, ofid = draw_condition(snap)
            print(f"[seedcheck] snap={snap} sensors={int(om.sum())} "
                  f"idx_sum={int(oi[om.bool()].sum())}", flush=True)

            gv, gm = MB.build_obs_grid_mask3d(ov, om, ofid, oi, n_fields, n_pts,
                                              num_z, num_y, num_x,
                                              num_z, num_y, num_x)
            cond = {"obs_value_grid": gv, "obs_mask_grid": gm}

            # --- canonical ensemble noise seeding -------------------------
            base = args.seed * 131 + si
            ens = []
            with torch.no_grad():
                for k in range(args.K):
                    torch.manual_seed(base * 10_000 + k)
                    if k == 0:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(device)
                        t0 = time.perf_counter()
                    grid = model.sample(cond, n_steps=args.nfe,
                                        ode_solver=ode_solver)
                    if k == 0:
                        torch.cuda.synchronize()
                        timings.append(time.perf_counter() - t0)
                        mems.append(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
                    ens.append(MB.grid3d_to_pointcloud(grid, num_z, num_y, num_x)
                               [0].detach().float().cpu().numpy())
            ens = np.stack(ens, axis=0)

            truth_np = fields[0].detach().float().cpu().numpy()
            m = ensemble_metrics(ens, truth_np, field_names)
            m["snapshot"] = snap
            per_snap.append(m)

            if (not args.no_figs) and (si % max(1, args.fig_every) == 0):
                fig_dir.mkdir(parents=True, exist_ok=True)
                try:
                    cr = item.get("coords_raw")
                    cr = cr.cpu().numpy() if cr is not None else coords[0].cpu().numpy()
                    save_ensemble_figure(
                        ens, truth_np, cr, field_names,
                        fig_dir / f"spread_nfe{args.nfe}_snap{snap}.png",
                        tag=(f"latent-FM {tag}  snap {snap}  K={args.K} "
                             f"NFE={args.nfe}  "
                             f"relL2={m['aggregate']['rel_l2_mean']:.4f}"))
                except Exception as exc:      # a plot must never kill an eval
                    print(f"  [warn] spread figure snap {snap} failed: {exc}",
                          flush=True)

            agg = m["aggregate"]
            print(f"[ensemble] snap={snap} K={args.K} " + " ".join(
                f"{k}={v:.5f}" for k, v in agg.items()), flush=True)

    # --- summary --------------------------------------------------------------
    agg_keys = list(per_snap[0]["aggregate"].keys())
    fld_keys = list(per_snap[0]["per_field"][field_names[0]].keys())

    def _mean_std(path):
        vals = []
        for m in per_snap:
            d = m
            for k in path:
                d = d[k]
            vals.append(float(d))
        return float(np.mean(vals)), float(np.std(vals))

    summary = {
        "aggregate": {k: _mean_std(["aggregate", k])[0] for k in agg_keys},
        "aggregate_std": {k: _mean_std(["aggregate", k])[1] for k in agg_keys},
        "per_field": {f: {k: _mean_std(["per_field", f, k])[0] for k in fld_keys}
                      for f in field_names},
    }

    payload = {
        "protocol": "canonical",
        "baseline": "latent_fm",
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "ckpt_tag": tag,
        "epoch": int(ckpt.get("epoch", -1)),
        "fidelity_fixes_active": bool(patched),
        "cond_mode": str(ckpt.get("cond_mode", "image")),
        "latent_scale_numel": int(getattr(model, "latent_scale",
                                          torch.ones(1)).numel()),
        "K": args.K,
        "nfe": args.nfe,
        "ode_solver": ode_solver,
        "seed": args.seed,
        "op_seed": args.op_seed,
        "split": args.split,
        "cond_fields": list(args.cond_fields),
        "n_obs": list(args.n_obs),
        "n_snapshots": len(snap_ids),
        "snapshot_ids": [int(s) for s in snap_ids],
        "fingerprint_checked": bool(canonical),
        "gpu": gpu_name,
        "field_names": field_names,
        "params": {"autoencoder": ae_params, "velocity_net": vn_params,
                   "combined": ae_params + vn_params},
        "bottleneck": {"latent_shape": list(latent_shape),
                       "latent_scalars": latent_scalars,
                       "field_scalars": n_fields * n_pts,
                       "compression_ratio": n_fields * n_pts / latent_scalars},
        "inference_wallclock_s_mean": float(np.mean(timings)),
        "inference_wallclock_s_std": float(np.std(timings)),
        "inference_peak_gpu_mem_mb": float(np.max(mems)),
        "summary": summary,
        "snapshots": per_snap,
    }
    out_path.write_text(json.dumps(payload, indent=1))

    pf = summary["per_field"]
    print(f"[RESULT] ckpt={tag} epoch={payload['epoch']} K={args.K} "
          f"nfe={args.nfe} n={len(snap_ids)} "
          f"agg_relL2={summary['aggregate']['rel_l2_mean']:.5f} "
          f"crps={summary['aggregate']['crps']:.5f} | "
          + " ".join(f"{f}={pf[f]['rel_l2_mean']:.4f}" for f in field_names)
          + f" | infer={payload['inference_wallclock_s_mean']:.3f}s "
            f"peak={payload['inference_peak_gpu_mem_mb']:.0f}MB", flush=True)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
