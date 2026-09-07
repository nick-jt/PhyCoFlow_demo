#!/usr/bin/env python3
"""Split-design ablation for the 2D Kolmogorov benchmark: quantify how much of
the gap between our benchmark numbers and literature-style figures is PROTOCOL
rather than architecture.

Arms (same checkpoint, same sensor count, same metric machinery everywhere):

  Protocol A (ours, already measured -- reference only): held-out-TRAJECTORY
      frames; the existing kolm_fleet_dmfgen_K8_nfe4.json numbers.

  Protocol B ("literature-style: unseen frame, seen trajectory"): frames drawn
      from the raw stride-1 npy at stride-4 offset +2 -- raw frame 4k+2 sits
      exactly BETWEEN two frames the model trained on (4k and 4k+4) of a TRAIN
      trajectory: maximally unseen-but-adjacent. 50 such frames evenly spaced
      over the (train_traj, k) candidate grid, normalized with the H5 train
      stats from kolmogorov2d_manifest.json, on coordinates identical to
      convert_kolmogorov.build_coordinates (taken directly from the run's own
      dataset object, so they are bit-identical to Protocol A's).
      -> kolm_litprotocol_{dmfgen,idw}.json, protocol
         "seen_trajectory_stride_offset2".

  Uniform-grid arm (Protocol A frames, lattice sensors): the fleet's exact 50
      held-out val frames, but sensors on a fixed 25x26 uniform lattice
      (650 sensors ~ the 655 scattered budget) instead of the seeded random
      scatter -- quantifies the scattered-vs-uniform information gap.
      -> kolm_uniformgrid_{dmfgen,idw}.json, protocol
         "heldout_trajectory_uniform_grid".

Models per arm: DMF-Gen (best.pt, K=8, NFE=4, canonical clamped chunked
sampler via ensemble_eval.sample_ensemble) and CPU IDW k=8 imported from
baseline_classical_2d (kd_predict, NON-PERIODIC coords via build_coords_box
periodic=False -- matching the fleet's classical_baselines_*_nonperiodic_2d
convention used by figs_pof/fig_perf_vs_sensors.py). IDW is deterministic and
scored exactly like the fleet's deterministic rows: two identical members
(fair CRPS == MAE), dispersion fields nulled.

Seeding:
  Protocol B sensors : torch.manual_seed(seed*777 + p) immediately before
      helpers.build_sparse_condition, where p = train-candidate index
      traj_pos*80 + k (these frames do not exist in the H5, so p replaces the
      H5 snapshot index in the canonical formula; recorded per snapshot).
      DMF-Gen and IDW see the IDENTICAL draw (drawn once, shared).
  Uniform arm sensors: no RNG -- one fixed lattice, identical for both models
      and every frame.
  Ensemble noise     : canonical base = seed*131 + si (si = list position),
      sample_ensemble(seed=base). The uniform arm's si ordering equals the
      fleet's, so its noise seeds match the fleet run frame-for-frame.

Touches no shared module. Outputs land in
Save_TrainedModel/kolmogorov2d/litprotocol/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

MANIFEST = ("/projects/ammoniacomb/generative_reconstruction/kolmogorov2d/"
            "kolmogorov2d_manifest.json")
NPY = ("/projects/ammoniacomb/generative_reconstruction/baselines/"
       "sparse-reconstruction/data/kolmogorov_shu.npy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Kolmogorov split-design ablation")
    p.add_argument("--run-dir", required=True, help="DMF-Gen run dir (best.pt)")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--nfe", type=int, default=4)
    p.add_argument("--n-obs", type=int, default=655)
    p.add_argument("--n-frames", type=int, default=50)
    p.add_argument("--idw-k", type=int, default=8)
    p.add_argument("--lattice", type=int, nargs=2, default=[25, 26],
                   help="Uniform-arm lattice (rows cols); 25x26 = 650 sensors.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=262_144)
    p.add_argument("--fig-every", type=int, default=10)
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--skip-uniform", action="store_true")
    p.add_argument("--out-dir", default=None,
                   help="Default: <repo>/Save_TrainedModel/kolmogorov2d/litprotocol")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def lit_b_frames(manifest: dict, n_frames: int) -> list[dict]:
    """50 (train_traj, raw_frame=4k+2) picks, evenly spaced over candidates."""
    train = list(manifest["train_trajs"])
    fpt = int(manifest["frames_per_traj"])          # 80 strided frames/traj
    stride = int(manifest["stride"])                # 4
    total = len(train) * fpt                        # 2560 candidates
    out = []
    for j in range(n_frames):
        p = (j * total) // n_frames
        k = p % fpt
        out.append({"p": p, "traj": int(train[p // fpt]), "k": int(k),
                    "raw_frame": int(stride * k + stride // 2)})
    return out


def uniform_lattice(ny: int, nx: int, rows: int, cols: int) -> np.ndarray:
    """Row-major flat indices of a centered rows x cols lattice."""
    iy = ((2 * np.arange(rows) + 1) * ny) // (2 * rows)
    ix = ((2 * np.arange(cols) + 1) * nx) // (2 * cols)
    return (iy[:, None] * nx + ix[None, :]).ravel().astype(np.int64)


def main() -> None:
    args = parse_args()
    if os.environ.get("JHU_SPLIT_MODE") != "block" or \
       os.environ.get("JHU_SPLIT_GAP") != "0":
        raise SystemExit("[guard] export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0 "
                         "(the run's dataset/stats depend on it).")

    run_dir = Path(args.run_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    out_dir = (Path(args.out_dir) if args.out_dir else
               script_dir.parent / "Save_TrainedModel/kolmogorov2d/litprotocol")
    manifest = json.load(open(MANIFEST))
    b_frames = lit_b_frames(manifest, args.n_frames)
    rows, cols = args.lattice
    lattice = uniform_lattice(*manifest["grid"], rows, cols)

    if args.dry_run:
        print(f"[dry-run] run={run_dir} ckpt exists="
              f"{(run_dir / args.ckpt).is_file()}")
        print(f"[dry-run] npy={NPY} exists={os.path.exists(NPY)}")
        print(f"[dry-run] out_dir={out_dir}")
        print(f"[dry-run] B frames ({len(b_frames)}): "
              + " ".join(f"t{f['traj']}r{f['raw_frame']}" for f in b_frames))
        print(f"[dry-run] uniform lattice {rows}x{cols} = {lattice.size} "
              f"sensors (scattered budget {args.n_obs}); "
              f"first idx {lattice[:4].tolist()} last {lattice[-2:].tolist()}")
        assert all(f["raw_frame"] % 4 == 2 for f in b_frames)
        assert all(f["raw_frame"] < 320 for f in b_frames)
        print("[dry-run] OK")
        return

    from helpers import build_sparse_condition
    from ensemble_eval import (ensemble_metrics, load_run,
                               require_compute_node, sample_ensemble,
                               save_ensemble_figure)
    from eval_kolm_ensemble import frame_list, null_dispersion, summarize
    from baseline_classical_2d import build_coords_box, kd_predict

    require_compute_node()
    gpu_name = torch.cuda.get_device_name(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    model, dataset, cfg = load_run(str(run_dir), args.ckpt, device)
    field_names = list(dataset.field_names)
    coords_dev = dataset.coords.unsqueeze(0).to(device)          # [1, N, 3]
    coords_raw = dataset.coords_raw.cpu().numpy()

    # Stats cross-check: manifest train_stats vs the run's dataset_stats.pt.
    m_mean = float(manifest["train_stats"]["vorticity"]["mean"])
    m_std = float(manifest["train_stats"]["vorticity"]["std"])
    d_mean = float(dataset.mean.ravel()[0])
    d_std = float(dataset.std.ravel()[0])
    print(f"[stats] manifest mean={m_mean:.6e} std={m_std:.6f} | "
          f"run stats mean={d_mean:.6e} std={d_std:.6f}", flush=True)
    if not (abs(m_std - d_std) < 1e-3 * m_std and abs(m_mean - d_mean) < 1e-3 * m_std):
        raise SystemExit("[guard] manifest train stats disagree with the run's "
                         "dataset_stats.pt -- normalization would not match.")

    coords_box, boxsize = build_coords_box(coords_raw[:, :2], periodic=False)
    idw_note = ("IDW k=%d, NON-PERIODIC distances (build_coords_box periodic="
                "False), matching the fleet's classical_baselines_*_"
                "nonperiodic_2d convention" % args.idw_k)

    def dmfgen_draw(fields_dev, obs, base):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        ens = sample_ensemble(model, coords_dev, obs, K=args.K,
                              n_steps=args.nfe, chunk=args.chunk,
                              clamp_hard=True, seed=base).numpy()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.K
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        return ens, dt, peak

    def idw_predict(sensor_idx, sensor_val):
        t0 = time.perf_counter()
        pred = kd_predict(coords_box, {0: (sensor_idx, sensor_val)},
                          n_ch=1, mode="idw", k=args.idw_k, boxsize=boxsize)
        return pred, time.perf_counter() - t0

    def make_payload(protocol, model_name, frames_desc, per_snap, cost,
                     extra=None):
        pay = {
            "protocol": protocol,
            "model": model_name,
            "run_dir": str(run_dir),
            "ckpt": args.ckpt,
            "K": 1 if model_name == "idw" else args.K,
            "deterministic": model_name == "idw",
            "nfe": None if model_name == "idw" else args.nfe,
            "seed": args.seed,
            "n_frames": len(per_snap),
            "frames": frames_desc,
            "field_names": field_names,
            "normalization": {"mean": m_mean, "std": m_std,
                              "source": "kolmogorov2d_manifest.json train_stats "
                                        "(== the run's dataset_stats.pt)"},
            "gpu": gpu_name,
            **(extra or {}),
            **cost,
            "summary": summarize(per_snap, field_names),
            "snapshots": per_snap,
        }
        return pay

    def score_det(pred, truth):
        m = null_dispersion(ensemble_metrics(np.repeat(pred[None], 2, axis=0),
                                             truth, field_names))
        return m

    def write(name, payload):
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=1))
        s = payload["summary"]["aggregate"]
        print(f"[out] {path.name}: relL2={s['rel_l2_mean']:.5f} "
              f"crps={s['crps']:.5f}", flush=True)

    # ================= Protocol B: seen trajectory, unseen frame ============
    raw = np.load(NPY, mmap_mode="r")
    per_dmf, per_idw = [], []
    t_dmf, m_dmf, t_idw = [], [], []
    fig_dir = out_dir / "figs_litprotocol"
    for si, spec in enumerate(b_frames):
        frame = np.asarray(raw[spec["traj"], spec["raw_frame"]],
                           dtype=np.float32)
        truth = ((frame.reshape(-1, 1) - m_mean) / m_std).astype(np.float32)
        fields_dev = torch.from_numpy(truth).unsqueeze(0).to(device)

        torch.manual_seed(args.seed * 777 + spec["p"])
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords_dev, fields_full=fields_dev,
            cond_fields=[0], n_obs_min=[args.n_obs], n_obs_max=[args.n_obs])
        sensors = int(om.sum())
        idx_sum = int(oi[om.bool()].sum())
        print(f"[seedcheck] B traj={spec['traj']} raw={spec['raw_frame']} "
              f"p={spec['p']} sensors={sensors} idx_sum={idx_sum}", flush=True)

        base = args.seed * 131 + si
        obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
               "field_ids": ofid}
        ens, dt, peak = dmfgen_draw(fields_dev, obs, base)
        t_dmf.append(dt)
        m_dmf.append(peak)
        m = ensemble_metrics(ens, truth, field_names)
        m.update(spec)
        m["sensors"] = sensors
        m["idx_sum"] = idx_sum
        per_dmf.append(m)

        v = om[0].bool()
        s_idx = oi[0, v].long().cpu().numpy()
        s_val = ov[0, v, 0].float().cpu().numpy()
        pred, dt_i = idw_predict(s_idx, s_val)
        t_idw.append(dt_i)
        mi = score_det(pred, truth)
        mi.update(spec)
        mi["sensors"] = sensors
        mi["idx_sum"] = idx_sum
        per_idw.append(mi)

        if (not args.no_figs) and (si % max(1, args.fig_every) == 0):
            fig_dir.mkdir(parents=True, exist_ok=True)
            try:
                save_ensemble_figure(
                    ens, truth, coords_raw, field_names,
                    fig_dir / f"B_t{spec['traj']}r{spec['raw_frame']}.png",
                    tag=(f"lit-protocol B traj{spec['traj']} raw{spec['raw_frame']} "
                         f"relL2={m['aggregate']['rel_l2_mean']:.4f}"))
            except Exception as exc:
                print(f"  [warn] figure failed: {exc}", flush=True)
        print(f"[B] {si + 1}/{len(b_frames)} "
              f"dmfgen relL2={m['aggregate']['rel_l2_mean']:.4f} "
              f"idw relL2={mi['aggregate']['rel_l2_mean']:.4f}", flush=True)

    seed_note_b = ("sensors: torch.manual_seed(seed*777 + p) + canonical "
                   "helpers.build_sparse_condition on CUDA, p = train-"
                   "candidate index traj_pos*80 + k (frames absent from the "
                   "H5); DMF-Gen and IDW share the identical draw. noise: "
                   "base=seed*131+si, sample_ensemble(seed=base).")
    frames_desc = b_frames
    write("kolm_litprotocol_dmfgen.json", make_payload(
        "seen_trajectory_stride_offset2", "dmfgen", frames_desc, per_dmf,
        {"inference_seconds_per_field_mean": float(np.mean(t_dmf)),
         "inference_seconds_per_field_std": float(np.std(t_dmf)),
         "inference_peak_gpu_gb": float(np.max(m_dmf)),
         "timing_note": "total sample_ensemble wall-clock / K, CUDA-synced"},
        {"n_obs": [args.n_obs], "cond_fields": [0], "seeding": seed_note_b,
         "npy": NPY,
         "frame_note": "raw_frame = 4k+2: exactly between two stride-4 "
                       "TRAIN frames of a TRAIN trajectory"}))
    write("kolm_litprotocol_idw.json", make_payload(
        "seen_trajectory_stride_offset2", "idw", frames_desc, per_idw,
        {"inference_seconds_per_field_mean": float(np.mean(t_idw)),
         "inference_seconds_per_field_std": float(np.std(t_idw)),
         "inference_peak_gpu_gb": None,
         "timing_note": "CPU cKDTree IDW; no GPU cost"},
        {"n_obs": [args.n_obs], "cond_fields": [0], "seeding": seed_note_b,
         "npy": NPY, "estimator": idw_note}))

    # ================= Uniform-lattice arm (Protocol A frames) ==============
    if not args.skip_uniform:
        frames = frame_list(len(dataset), args.n_frames)
        lat_t = torch.from_numpy(lattice).to(device)
        oi_u = lat_t.unsqueeze(0)                                # [1, M]
        oc_u = coords_dev[:, lat_t]                              # [1, M, 3]
        om_u = torch.ones(1, lattice.size, device=device)
        ofid_u = torch.zeros(1, lattice.size, dtype=torch.long, device=device)
        per_dmf, per_idw = [], []
        t_dmf, m_dmf, t_idw = [], [], []
        fig_dir = out_dir / "figs_uniformgrid"
        for si, snap in enumerate(frames):
            item = dataset[int(snap)]
            fields_dev = item["fields"].unsqueeze(0).to(device)
            truth = item["fields"].cpu().numpy()
            ov_u = fields_dev[:, lat_t, 0:1]                     # [1, M, 1]
            obs = {"coords": oc_u, "values": ov_u, "mask": om_u,
                   "indices": oi_u, "field_ids": ofid_u}
            base = args.seed * 131 + si
            ens, dt, peak = dmfgen_draw(fields_dev, obs, base)
            t_dmf.append(dt)
            m_dmf.append(peak)
            m = ensemble_metrics(ens, truth, field_names)
            m["snapshot"] = int(snap)
            m["sensors"] = int(lattice.size)
            per_dmf.append(m)

            pred, dt_i = idw_predict(lattice,
                                     ov_u[0, :, 0].float().cpu().numpy())
            t_idw.append(dt_i)
            mi = score_det(pred, truth)
            mi["snapshot"] = int(snap)
            mi["sensors"] = int(lattice.size)
            per_idw.append(mi)

            if (not args.no_figs) and (si % max(1, args.fig_every) == 0):
                fig_dir.mkdir(parents=True, exist_ok=True)
                try:
                    save_ensemble_figure(
                        ens, truth, coords_raw, field_names,
                        fig_dir / f"U_snap{int(snap)}.png",
                        tag=(f"uniform {rows}x{cols} snap {snap} "
                             f"relL2={m['aggregate']['rel_l2_mean']:.4f}"))
                except Exception as exc:
                    print(f"  [warn] figure failed: {exc}", flush=True)
            print(f"[U] {si + 1}/{len(frames)} snap={snap} "
                  f"dmfgen relL2={m['aggregate']['rel_l2_mean']:.4f} "
                  f"idw relL2={mi['aggregate']['rel_l2_mean']:.4f}", flush=True)

        seed_note_u = (f"sensors: FIXED {rows}x{cols} centered lattice "
                       f"({lattice.size} sensors vs the {args.n_obs} scattered "
                       "budget), identical for every frame and both models; "
                       "no sensor RNG. Frames and noise seeds identical to "
                       "the kolm_fleet run (same si ordering).")
        write("kolm_uniformgrid_dmfgen.json", make_payload(
            "heldout_trajectory_uniform_grid", "dmfgen", frames, per_dmf,
            {"inference_seconds_per_field_mean": float(np.mean(t_dmf)),
             "inference_seconds_per_field_std": float(np.std(t_dmf)),
             "inference_peak_gpu_gb": float(np.max(m_dmf)),
             "timing_note": "total sample_ensemble wall-clock / K, CUDA-synced"},
            {"n_obs": [int(lattice.size)], "cond_fields": [0],
             "lattice": [rows, cols], "seeding": seed_note_u}))
        write("kolm_uniformgrid_idw.json", make_payload(
            "heldout_trajectory_uniform_grid", "idw", frames, per_idw,
            {"inference_seconds_per_field_mean": float(np.mean(t_idw)),
             "inference_seconds_per_field_std": float(np.std(t_idw)),
             "inference_peak_gpu_gb": None,
             "timing_note": "CPU cKDTree IDW; no GPU cost"},
            {"n_obs": [int(lattice.size)], "cond_fields": [0],
             "lattice": [rows, cols], "seeding": seed_note_u,
             "estimator": idw_note}))

    print("[done] all arms written to", out_dir, flush=True)


if __name__ == "__main__":
    main()
