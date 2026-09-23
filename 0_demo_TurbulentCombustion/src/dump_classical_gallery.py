#!/usr/bin/env python3
"""Classical-baseline field dumps for the 2D reconstruction galleries.

CPU-only companion of src/dump_kolm_gallery.sh: reconstructs the SAME single
gallery frames with the training-free anchors (kdtree / IDW / gappy POD) by
importing the machinery of baseline_classical_2d.py (estimators, canonical
sensor draw, train-stat z-scoring), and writes per-dataset npz files next to
the learned dumps:

    Save_TrainedModel_pof/field_dumps/kolm_classical.npz   (val frame 256)
    Save_TrainedModel_pof/field_dumps/cyl_classical.npz    (val frame 300)

Sensor matching
---------------
The learned dumps draw sensors with torch.manual_seed(seed*777+snap) on a
CUDA device; torch.randperm on CPU gives a DIFFERENT permutation from the
same seed, so a login-node draw cannot be bit-identical.  This script
therefore prefers the EXACT sensor set: if a learned npz for the same
dataset (kolm_dmfgen.npz / cyl_dmfgen.npz, falling back to any
same-protocol leg) already exists, its stored `sensor_indices`/`sensor_field_ids` are
reused verbatim (meta.sensor_source = "learned_npz:<file>").  Until the GPU
dump lands, the fallback is the same-seed CPU draw, flagged
meta.sensor_source = "cpu_same_seed_APPROXIMATE" -- same seed, same count,
same protocol, different sites.  RERUN THIS SCRIPT after the GPU job
finishes to upgrade the classical panels to the exact sensor sets (it
overwrites its outputs unconditionally).

Everything is in z-score units of the TRAIN-split stats (identical formula
to helpers.TurbulentCombustionH5Dataset), matching both the learned dumps'
standardized arrays and the classical-anchor JSONs.

Cylinder note: only Ux,Uy carry sensors (cond_fields=[0,1]); the p channel
is unobserved.  kdtree/IDW predict the train mean (=0) there by
construction; gappy POD reconstructs p through the POD basis's cross-channel
correlations -- that contrast is the figure's punchline.  Ranks 20 AND 80
are dumped (POD-20 is the table headline).

Usage (login node OK; no GPU, no compute-node gate):
    python dump_classical_gallery.py [--datasets kolm cyl] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import baseline_classical_2d as BC

WT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = WT / "Save_TrainedModel_pof" / "field_dumps"

KOLM = dict(
    h5="/projects/ammoniacomb/generative_reconstruction/kolmogorov2d/"
       "Kolmogorov2D_shu_stride4.h5",
    train_frames=2560,
    field_names=["vorticity"],
    cond_fields=[0],
    n_obs=655,
    snap=256,            # 0-based position in the test split (= val index)
    periodic=True,       # Kolmogorov [0,2pi)^2 is genuinely periodic
    pod_ranks=[80],      # the classical-anchor rank
    learned_npz=["kolm_dmfgen.npz", "kolm_latent_fm.npz", "kolm_sit.npz",
                 "kolm_senseiver.npz"],
    out="kolm_classical.npz",
)
CYL = dict(
    h5="/projects/ammoniacomb/generative_reconstruction/cylinder2d/"
       "Cylinder2D_mesh.h5",
    train_frames=1200,
    field_names=["Ux", "Uy", "p"],
    cond_fields=[0, 1],
    n_obs=238,
    snap=300,            # val index; absolute frame 1500 (held-out Re250)
    periodic=False,      # unstructured mesh
    pod_ranks=[20, 80],  # POD-20 (0.056 relL2) is the headline; 80 for contrast
    learned_npz=["cyl_dmfgen.npz", "cyl_senseiver.npz"],  # mesh-protocol legs
    out="cyl_classical.npz",
)


def sensors_from_learned(out_dir: Path, cfg: dict, Z: np.ndarray):
    """Exact sensor sets from a landed learned dump, if any.

    Returns (sensors dict {fld: (idx, val)}, source string) or (None, None).
    Values are re-read from OUR z-scored frame at the stored indices (the
    z-scoring conventions are identical, but this keeps the classical dump
    self-consistent even if a run's stats file drifted).
    """
    for name in cfg["learned_npz"]:
        p = out_dir / name
        if not p.is_file():
            continue
        try:
            with np.load(p, allow_pickle=False) as d:
                meta = json.loads(str(d["meta"]))
                if int(meta.get("snapshot_index", -1)) != cfg["snap"] or \
                        list(meta.get("cond_fields", [])) != cfg["cond_fields"] or \
                        int(meta.get("n_obs", -1)) != cfg["n_obs"]:
                    print(f"[sensors] {name}: protocol mismatch "
                          f"({meta.get('snapshot_index')}, "
                          f"{meta.get('cond_fields')}, {meta.get('n_obs')}); "
                          "skipping")
                    continue
                idx = np.asarray(d["sensor_indices"], dtype=np.int64)
                fid = np.asarray(d["sensor_field_ids"], dtype=np.int64)
        except Exception as e:
            print(f"[sensors] failed to read {p}: {e}")
            continue
        sens = {}
        for fld in cfg["cond_fields"]:
            m = fid == fld
            sens[int(fld)] = (idx[m], Z[idx[m], fld].astype(np.float32))
        n = sum(v[0].size for v in sens.values())
        s = sum(int(v[0].sum()) for v in sens.values())
        print(f"[sensors] EXACT set from {name}: sensors={n} idx_sum={s}")
        return sens, f"learned_npz:{name}"
    return None, None


def run_dataset(cfg: dict, out_dir: Path, idw_k: int = 8) -> None:
    t_all = time.perf_counter()
    BC.enable_percentile_fastpath()
    craw, T, C = BC.read_layout(cfg["h5"])
    N = craw.shape[0]
    names = cfg["field_names"]
    assert len(names) == C, (names, C)
    train_frames = cfg["train_frames"]
    train_idx = np.arange(train_frames)
    abs_frame = train_frames + cfg["snap"]
    assert abs_frame < T
    mean, std = BC.compute_train_stats(cfg["h5"], train_idx, C)
    print(f"[data] {Path(cfg['h5']).name}: N={N} C={C} T={T} "
          f"train={train_frames} snap={cfg['snap']} (abs {abs_frame}) "
          f"mean={mean} std={std}")
    Z = BC.load_split(cfg["h5"], [abs_frame], N, C, mean, std)[0]  # [N, C]

    coords_box, boxsize = BC.build_coords_box(craw, periodic=cfg["periodic"])

    # ---- sensors: exact (from a landed learned dump) or same-seed CPU ------
    sens, source = sensors_from_learned(out_dir, cfg, Z)
    if sens is None:
        sens = BC.draw_sensors(
            torch.from_numpy(coords_box.astype(np.float32)),
            torch.from_numpy(Z), snap=cfg["snap"], n_obs=cfg["n_obs"],
            seed=0, device="cpu", cond_fields=cfg["cond_fields"])
        source = "cpu_same_seed_APPROXIMATE"
        n = sum(v[0].size for v in sens.values())
        s = sum(int(v[0].sum()) for v in sens.values())
        print(f"[sensors] APPROXIMATE CPU draw (torch.manual_seed(0*777+"
              f"{cfg['snap']}), device=cpu): sensors={n} idx_sum={s} -- "
              "NOT bit-identical to the CUDA draw of the learned dumps; "
              "rerun after the GPU dump lands for the exact set.")

    save = dict(truth=Z.astype(np.float32),
                coords=coords_box.astype(np.float32),
                coords_raw=craw.astype(np.float32),
                names=np.array(names),
                norm_mean=mean.astype(np.float32),
                norm_std=std.astype(np.float32),
                sensor_indices=np.concatenate(
                    [sens[f][0] for f in sorted(sens)]).astype(np.int64),
                sensor_field_ids=np.concatenate(
                    [np.full(sens[f][0].size, f, dtype=np.int64)
                     for f in sorted(sens)]))
    metrics = {}

    def add(key: str, pred: np.ndarray):
        m = BC.score(pred, Z, names)
        metrics[key] = m
        save[f"pred_{key}"] = pred.astype(np.float32)
        pf = " ".join(f"{f}:{m['per_field'][f]['rel_l2_mean']:.4f}"
                      for f in names)
        print(f"[{key}] agg relL2={m['aggregate']['rel_l2_mean']:.4f} | {pf}")

    # ---- kdtree / IDW ------------------------------------------------------
    add("kdtree", BC.kd_predict(coords_box, sens, C, "nn", idw_k, boxsize))
    add("idw", BC.kd_predict(coords_box, sens, C, "idw", idw_k, boxsize))

    # ---- gappy POD ---------------------------------------------------------
    t0 = time.perf_counter()
    Xtr = BC.load_split(cfg["h5"], train_idx, N, C, mean, std)
    Xtr = Xtr.reshape(train_frames, -1)
    mu = Xtr.mean(axis=0)
    Xtr -= mu
    pod = BC.GappyPOD(Xtr, mu, rank=max(cfg["pod_ranks"]))
    print(f"[pod] fit ({Xtr.shape}) in {time.perf_counter() - t0:.1f}s, "
          f"peak RSS {BC.peak_rss_gb():.1f} GB")
    cols, vals = BC.obs_columns(sens, C)
    for r in cfg["pod_ranks"]:
        pod.set_rank(r)
        add(f"gappy_pod_r{pod.r}",
            pod.reconstruct(cols, vals).reshape(N, C))

    meta = {
        "protocol": "classical gallery dump (baseline_classical_2d machinery)",
        "data": cfg["h5"],
        "train_frames": train_frames,
        "snapshot_index": cfg["snap"],
        "absolute_frame": int(abs_frame),
        "field_names": names,
        "cond_fields": cfg["cond_fields"],
        "n_obs_per_observed_field": cfg["n_obs"],
        "sensor_source": source,
        "sensor_seed_formula": f"torch.manual_seed(0*777+{cfg['snap']})",
        "periodic_kdtree": boxsize is not None,
        "idw_k": idw_k,
        "pod_ranks": [int(r) for r in cfg["pod_ranks"]],
        "units": "STANDARDIZED (z-score, train-split stats; physical = "
                 "arr*norm_std+norm_mean)",
        "unobserved_channels": "kdtree/idw predict 0 (train mean) on "
                               "channels without sensors; gappy POD "
                               "reconstructs them from the basis",
        "metrics": metrics,
    }
    save["meta"] = np.array(json.dumps(meta, indent=1))
    out = out_dir / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **save)
    print(f"[done] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) "
          f"in {time.perf_counter() - t_all:.1f}s total\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cyl", "kolm"],
                   choices=["kolm", "cyl"])
    p.add_argument("--out-dir", default=str(OUT_DEFAULT))
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    for d in args.datasets:
        run_dataset(CYL if d == "cyl" else KOLM, out_dir)


if __name__ == "__main__":
    main()
