"""Canonical cross-cube evaluation for the S3GM baseline (3-D adaptation).

Replicates ensemble_eval.py's driver seeding EXACTLY (ensemble_eval.py:291-304):

    rng = np.random.default_rng(seed)
    snap_ids = rng.choice(len(dataset), size=n_snapshots, replace=False)
    for snap in snap_ids:
        torch.manual_seed(seed * 777 + int(snap))   # immediately before the draw
        build_sparse_condition(...)                  # from helpers.py, NOT helpers_baseline

Fingerprint printed for cross-baseline agreement:
    [seedcheck] snap=29 sensors=39062 idx_sum=37987162596

MUST run on a compute node: torch.randperm on CUDA is not portable between
H100 PCIe (login) and H100 SXM (compute).
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

import model_baseline as MB
import s3gm3d as S
from helpers import build_sparse_condition
from ensemble_eval import (
    check_canonical_fingerprint,
    ensemble_metrics,
    require_compute_node,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best", help="best | last | archive/epoch_XXXX.pt | path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op-seed", type=int, default=1000)
    p.add_argument("--n-snapshots", type=int, default=50)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0, 2])
    p.add_argument("--n-obs", type=int, nargs="+", default=[19531, 19531])
    p.add_argument("--n-steps", type=int, default=None, help="denoising steps (default: config sampling_N)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    p.add_argument("--seedcheck-only", action="store_true")
    p.add_argument("--arm", choices=["jhu_tuned", "upstream"], default="jhu_tuned",
                   help="Guidance arm. 'jhu_tuned' = alpha_case 0.05 / beta 0.004 "
                        "(OUR DEVIATION, stable at JHU scale). 'upstream' = "
                        "alpha_case 0.5 / beta 0.4 (upstream-faithful, DIVERGES "
                        "at JHU scale -- run only to document the divergence).")
    a = p.parse_args()

    require_compute_node()
    run_dir = Path(a.run_dir)
    cfg = MB.validate_and_normalize_config(MB.load_yaml(run_dir / "run_config.yaml"))

    # Training may have staged the H5 to node-local NVMe and recorded that path
    # in run_config.yaml.  That path does not exist on the evaluation node, so
    # fall back to the shared copy the launcher preserved.
    _paths = cfg["shared"]["paths"]
    _dp = Path(_paths["data_path"])
    if not _dp.exists():
        _shared = _paths.get("data_path_shared")
        if _shared and Path(_shared).exists():
            print(f"[eval] staged data_path {_dp} absent; using shared {_shared}", flush=True)
            _paths["data_path"] = _shared
        else:
            raise FileNotFoundError(
                f"data_path {_dp} does not exist and no usable data_path_shared "
                f"({_shared!r}) was recorded in run_config.yaml.")

    device = torch.device(a.device)
    print(f"[eval] device={device} "
          f"({torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'})", flush=True)

    stats_path = run_dir / "dataset_stats.pt"
    dataset = MB.build_dataset(cfg, split="val", stats_path=stats_path)
    print(f"[eval] val snapshots={len(dataset)} points={dataset.num_points} "
          f"fields={list(dataset.field_names)}", flush=True)

    # --------------------------------------------------------------- model
    adapter = S.S3GM3DAdapter()
    train_set = MB.build_dataset(cfg, split="train", stats_path=stats_path)
    bundle = adapter.build_for_training(cfg, device, run_dir, train_set, dataset)
    ck_path = Path(a.ckpt) if os.path.sep in a.ckpt else run_dir / f"{a.ckpt}.pt"
    ck = MB.safe_torch_load(ck_path, map_location="cpu")
    adapter.load_checkpoint(bundle, ck)
    print(f"[eval] checkpoint={ck_path} epoch={ck.get('epoch')} "
          f"val_loss={ck.get('val_loss')} params={ck.get('trainable_params')}", flush=True)

    cmp_ = bundle.components
    Nz, Ny, Nx = cmp_["Nz"], cmp_["Ny"], cmp_["Nx"]
    H_pad, W_pad = cmp_["H_pad"], cmp_["W_pad"]
    scfg = cmp_["sampling_cfg"]
    n_steps = int(a.n_steps if a.n_steps is not None else scfg["sampling_N"])

    # ---------------------------------------------------------------- arms
    # Selected here, NOT taken from run_config.yaml, so a run trained before
    # the stabilised values were adopted cannot silently evaluate at the
    # diverging ones. Both arms are reported; neither is a substitute.
    ARMS = {
        "jhu_tuned": {"alpha_case": 0.05, "beta": 0.004,
                      "label": "JHU-TUNED DEVIATION (NOT upstream)",
                      "note": "upstream alpha_case=0.5/beta=0.4 each diverge "
                              "independently at JHU scale; stability edge alpha<1"},
        "upstream": {"alpha_case": 0.5, "beta": 0.4,
                     "label": "UPSTREAM-FAITHFUL (diverges at JHU scale)",
                     "note": "reported only to document the divergence"},
    }
    arm = ARMS[a.arm]
    scfg["alpha_case"], scfg["beta"] = arm["alpha_case"], arm["beta"]
    print("=" * 72, flush=True)
    print(f"[arm] {a.arm}: {arm['label']}", flush=True)
    print(f"[arm] alpha_case={arm['alpha_case']} beta={arm['beta']}  -- {arm['note']}", flush=True)
    print("=" * 72, flush=True)

    # ------------------------------------------------- driver seeding (copy)
    rng = np.random.default_rng(a.seed)
    snap_ids = rng.choice(len(dataset), size=min(a.n_snapshots, len(dataset)), replace=False)

    results, per_snapshot_walls = [], []
    infer_peak_mb = 0.0
    t_all = time.perf_counter()

    with adapter.evaluation_weights(bundle):
        bundle.model.eval()
        for si, snap in enumerate(snap_ids):
            item = dataset[int(snap)]
            coords = item["coords"].unsqueeze(0).to(device)
            fields = item["fields"].unsqueeze(0).to(device)

            torch.manual_seed(a.seed * 777 + int(snap))
            oc, ov, om, oi, ofid = build_sparse_condition(
                coords_full=coords, fields_full=fields,
                cond_fields=a.cond_fields, n_obs_min=a.n_obs, n_obs_max=a.n_obs)
            sel = oi[om > 0]
            print(f"[seedcheck] snap={int(snap)} sensors={int(sel.numel())} "
                  f"idx_sum={int(sel.sum().item())}", flush=True)
            check_canonical_fingerprint(int(snap), int(sel.numel()),
                                        int(sel.sum().item()), a.seed,
                                        a.cond_fields, a.n_obs)
            if a.seedcheck_only:
                continue

            yv, mv = S.obs_to_video(ov, om, ofid, oi, dataset.num_fields,
                                    dataset.num_points, Nz, Ny, Nx, H_pad, W_pad)
            obs_frac = float(int(om.sum())) / float(dataset.num_points * len(a.cond_fields))
            alpha = float(scfg["alpha_case"]) / math.sqrt(max(obs_frac, 1e-12))

            sde_s = MB.VESDE(config=None, sigma_min=cmp_["sigma_min"],
                             sigma_max=cmp_["sigma_max"], N=n_steps)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.manual_seed(a.seed * 131 + si)
            with torch.enable_grad():
                vol = S.s3gm_reconstruct_ensemble(
                    bundle.model, sde_s, yv, mv, Nz, K=a.K, chunk=1,
                    t_win=cmp_["t_win"],
                    overlap=cmp_["overlap"], alpha=alpha, beta=float(scfg["beta"]),
                    snr=float(scfg["snr"]),
                    n_corrector_steps=int(scfg["n_corrector_steps"]),
                    device=device, progress=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            per_snapshot_walls.append(wall)
            if torch.cuda.is_available():
                infer_peak_mb = max(infer_peak_mb,
                                    torch.cuda.max_memory_allocated(device) / 1024 ** 2)

            ens = S.video_to_pointcloud(vol, Nz, Ny, Nx).float().cpu().numpy()   # [K,N,C]
            m = ensemble_metrics(ens, fields[0].cpu().numpy(),
                                 [str(x) for x in dataset.field_names])
            m["snapshot"] = int(snap)
            m["wall_s"] = wall
            results.append(m)
            print(f"[eval] {si+1}/{len(snap_ids)} snap={int(snap)} "
                  f"relL2={m['aggregate']['rel_l2_mean']:.4f} "
                  f"crps={m['aggregate']['crps']:.4f} "
                  f"spread/err={m['aggregate']['spread_error_ratio']:.3f} "
                  f"wall={wall:.1f}s", flush=True)
            del vol, ens
            torch.cuda.empty_cache()

    if a.seedcheck_only:
        return

    keys = list(results[0]["aggregate"].keys())
    agg = {k: float(np.mean([r["aggregate"][k] for r in results])) for k in keys}
    fields_names = list(results[0]["per_field"].keys())
    per_field = {f: {k: float(np.mean([r["per_field"][f][k] for r in results]))
                     for k in results[0]["per_field"][f]} for f in fields_names}

    wall_per_field = float(np.mean(per_snapshot_walls)) / max(a.K, 1)
    instr = {
        "train_step_s_excl_val": ck.get("train_step_s"),
        "train_peak_gpu_mb": ck.get("train_peak_gpu_mb"),
        "infer_wall_s_one_field": wall_per_field,
        "infer_wall_s_ensemble_K": float(np.mean(per_snapshot_walls)),
        "infer_peak_gpu_mb": infer_peak_mb,
        "n_denoising_steps": n_steps,
        "K": a.K,
        "trainable_params": ck.get("trainable_params"),
    }
    print("[instr] " + " ".join(f"{k}={v}" for k, v in instr.items()), flush=True)
    print(f"[agg] " + " ".join(f"{k}={v:.5f}" for k, v in agg.items()), flush=True)

    out = Path(a.out) if a.out else run_dir / f"eval_s3gm3d_{a.arm}_{a.ckpt.replace('/', '_')}.json"
    with open(out, "w") as fh:
        json.dump({
            "run_dir": str(run_dir), "checkpoint": str(ck_path),
            "epoch": ck.get("epoch"),
            "arm": a.arm,
            "arm_label": arm["label"],
            "arm_note": arm["note"],
            "is_upstream_faithful": (a.arm == "upstream"),
            "guidance": {"alpha_case": arm["alpha_case"], "beta": arm["beta"],
                         "upstream_alpha_case": 0.5, "upstream_beta": 0.4},
            "protocol": {"seed": a.seed, "op_seed": a.op_seed,
                         "n_snapshots": int(len(snap_ids)), "K": a.K,
                         "cond_fields": a.cond_fields, "n_obs": a.n_obs,
                         "n_denoising_steps": n_steps},
            "instrumentation": instr,
            "aggregate": agg, "per_field": per_field,
            "per_snapshot": results,
            "total_wall_s": time.perf_counter() - t_all,
        }, fh, indent=2)
    print(f"[eval] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
