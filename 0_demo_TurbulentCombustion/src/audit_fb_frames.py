#!/usr/bin/env python
"""FireBench operator-matrix frame audit (2026-09-05).

Reproduces the val-index -> absolute-frame logic of BOTH dataset classes
(src/helpers.py TurbulentCombustionH5Dataset and src/helpers_baseline.py
TurbulentCombustionH5Dataset) for the FireBench H5, under BOTH boundary
semantics:

  * "int"   : n_val = max(1, int(N * (1-train_ratio)))          <- $MAIN as of
              2026-09-05 (helpers.py L239, helpers_baseline.py L159); this is
              what eval jobs 16726561 / 16764760 ran under.
  * "round" : n_val = max(1, int(round(N * (1-train_ratio))))   <- worktree fix
              (helpers.py L242, helpers_baseline.py L162), NOT in $MAIN.

Both classes are otherwise identical in block mode:
  all_indices = arange(0, num_times, time_stride)   # stride 1 here
  n_train     = max(1, N - n_val - gap)
  train       = all_indices[:n_train]
  val/test    = all_indices[n_train + gap:]         # gap frames discarded

Split parameters (verified from run configs, not memory):
  ours (N18)   train_ratio 0.9   args.json of iclr_firebench_v4_DemoN18_*
  LFM          train_ratio 0.75  run_config.yaml of Baseline_latent_fm_Stage2_DemoN35_*
  Senseiver    train_ratio 0.75  run_config.yaml of Baseline_senseiver_Stage1_DemoN36_153857
  all three    JHU_SPLIT_MODE=block JHU_SPLIT_GAP=10 (train launchers + eval script)

Scored val indices in the operator-matrix jobs (from eval_firebench_ops_16764760.log
"matched snapshots:" line, chosen by ensemble_eval seed 0 from N18's val split):
  SNAPS = 3 9 10 5 2 1 0 4

Run:  ~/venvs/jhtdb_env/bin/python audit_fb_frames.py
CPU-only; reads H5 metadata + a strided point subsample for content stats.
"""
import json
import os
import sys

import numpy as np

H5 = "/projects/ammoniacomb/generative_reconstruction/firebench3d/FireBench_u10u12_merged.h5"
MAIN = "/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion"
N18 = MAIN + "/Save_TrainedModel/firebench/pointcloud_ffm/iclr_firebench_v4_DemoN18_20260819_083221"
LFM = MAIN + "/Save_TrainedModel/firebench/baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN35_20260823_234826"
SEN = MAIN + "/Save_TrainedModel/firebench/baseline_senseiver/Baseline_senseiver_Stage1_DemoN36_20260823_153857"

SNAPS = [3, 9, 10, 5, 2, 1, 0, 4]  # val indices passed identically to all models
OPS = ["clean", "noise01", "noise03", "slab25", "dropvw"]
GAP = 10
STRIDE = 1


def block_split(num_times, train_ratio, gap, semantics):
    """Reproduce helpers/helpers_baseline block split. Returns (train, gapf, val)."""
    all_indices = np.arange(0, num_times, STRIDE, dtype=np.int64)
    n = len(all_indices)
    frac = n * (1.0 - train_ratio)
    if semantics == "int":
        n_val = max(1, int(frac))
    elif semantics == "round":
        n_val = max(1, int(round(frac)))
    else:
        raise ValueError(semantics)
    n_train = max(1, n - n_val - gap)
    train = all_indices[:n_train]
    val = all_indices[n_train + gap:]
    gapf = all_indices[n_train:n_train + gap]
    return train, gapf, val


def fmt_range(a):
    a = np.asarray(a)
    return f"[{a.min()}..{a.max()}] (n={len(a)})"


def main():
    import h5py
    with h5py.File(H5, "r") as f:
        num_times = int(f["fields"].shape[1])
        times = f["time"][:].astype(float)
        cases = f.attrs.get("cases", "?")
    print(f"H5: {H5}")
    print(f"num_times={num_times}  cases attr: {cases}")
    # case boundary: time array restarts
    restart = int(np.argmin(np.diff(times))) + 1
    print(f"case boundary at frame {restart} (frames 0..{restart-1} = u10, "
          f"{restart}..{num_times-1} = u12; each t={times[0]:g}..{times[restart-1]:g}s)")

    splits = {}
    print("\n=== Split reconstruction (block mode, gap=10, stride=1) ===")
    for name, ratio in [("ours(0.9)", 0.9), ("baseline(0.75)", 0.75)]:
        for sem in ["int", "round"]:
            tr, gp, va = block_split(num_times, ratio, GAP, sem)
            splits[(name, sem)] = (tr, gp, va)
            print(f"{name:15s} {sem:5s}: n_val={len(va):2d} train={fmt_range(tr)} "
                  f"gap={fmt_range(gp)} val={fmt_range(va)}")

    # Which semantics did the eval jobs actually run under? ensemble_eval.py
    # drew snap_ids = default_rng(seed=0).choice(len(val_dataset), 8, replace=False)
    # and the 16764760 log records "matched snapshots: 3 9 10 5 2 1 0 4".
    print("\n=== Runtime-semantics fingerprint (ensemble_eval snapshot draw) ===")
    logged = [3, 9, 10, 5, 2, 1, 0, 4]
    for sem in ["int", "round"]:
        n = len(splits[("ours(0.9)", sem)][2])
        draw = list(np.random.default_rng(0).choice(n, size=8, replace=False))
        match = "MATCHES log" if [int(v) for v in draw] == logged else "does not match"
        print(f"  {sem:5s} (n_val={n}): draw={[int(v) for v in draw]}  -> {match}")

    print("\n=== Val-index -> absolute-frame map (frames also as u12 sim time) ===")
    for key in [("ours(0.9)", "int"), ("ours(0.9)", "round"),
                ("baseline(0.75)", "int"), ("baseline(0.75)", "round")]:
        va = splits[key][2]
        mp = ", ".join(f"{i}->{v}" for i, v in enumerate(va))
        print(f"{key[0]:15s} {key[1]:5s}: {mp}")

    # As-run semantics = int (that is what $MAIN/src had on 2026-08-25 and still has)
    ours_val = splits[("ours(0.9)", "int")][2]
    base_val = splits[("baseline(0.75)", "int")][2]
    ours_train = splits[("ours(0.9)", "int")][0]
    ours_gap = splits[("ours(0.9)", "int")][1]
    base_train = splits[("baseline(0.75)", "int")][0]

    ours_scored = [int(ours_val[s]) for s in SNAPS]
    base_scored = [int(base_val[s]) for s in SNAPS]

    def t12(fr):  # u12 sim time of an absolute frame
        return times[fr]

    print("\n=== Frames actually scored (as-run, int semantics) ===")
    print(f"{'val idx':>7s} {'ours frame':>10s} {'ours t[s]':>9s} "
          f"{'base frame':>10s} {'base t[s]':>9s}")
    for s in sorted(SNAPS):
        of, bf = int(ours_val[s]), int(base_val[s])
        print(f"{s:7d} {of:10d} {t12(of):9.0f} {bf:10d} {t12(bf):9.0f}")

    print("\n=== Overlap / contamination ===")
    inter = sorted(set(ours_scored) & set(base_scored))
    print(f"ours scored frames    : {sorted(ours_scored)}")
    print(f"baseline scored frames: {sorted(base_scored)}")
    print(f"scored-frame intersection: {inter if inter else 'NONE'}")
    print(f"ours val subset of baseline val: {set(ours_val) <= set(base_val)}")
    b_in_otrain = sorted(set(base_scored) & set(ours_train.tolist()))
    b_in_ogap = sorted(set(base_scored) & set(ours_gap.tolist()))
    o_in_btrain = sorted(set(ours_scored) & set(base_train.tolist()))
    print(f"baseline-scored frames inside OURS' train block: {b_in_otrain}")
    print(f"baseline-scored frames inside OURS' gap:         {b_in_ogap}")
    print(f"ours-scored frames inside BASELINES' train:      {o_in_btrain if o_in_btrain else 'NONE'}")
    d_o = [f - int(ours_train.max()) for f in sorted(ours_scored)]
    d_b = [f - int(base_train.max()) for f in sorted(base_scored)]
    print(f"distance (frames) past own train end -- ours: {d_o}  baselines: {d_b}")

    print("\n=== Remap table for matched re-eval (baseline val idx for ours' frames) ===")
    b0 = int(base_val[0])
    for s in sorted(SNAPS):
        of = int(ours_val[s])
        print(f"ours val idx {s:2d} = frame {of:3d} (u12 t={t12(of):.0f}s) "
              f"-> baseline --snapshot-index {of - b0}")

    # -------- per-frame metric variability from the archived JSONs --------
    print("\n=== Per-frame metrics from operator-matrix outputs (as archived) ===")
    for op in OPS:
        row = {}
        p = f"{N18}/Evaluation/ops_{op}_K8.json"
        if os.path.exists(p):
            d = json.load(open(p))
            per = {e["snapshot"]: e["aggregate"] for e in d["snapshots"]}
            rel = np.array([per[s]["rel_l2_mean"] for s in SNAPS])
            row["ours"] = rel
        for tag, rd in [("lfm", LFM), ("sen", SEN)]:
            vals = []
            for s in SNAPS:
                q = f"{rd}/Evaluation/ops_{op}_snap{s}.json"
                if os.path.exists(q):
                    vals.append(json.load(open(q))["aggregate"]["rel_l2_mean"])
            if vals:
                row[tag] = np.array(vals)
        print(f"[{op}]")
        xs = np.array(SNAPS, dtype=float)
        for tag, rel in row.items():
            sl = np.polyfit(xs, rel, 1)[0]
            r = np.corrcoef(xs, rel)[0, 1]
            print(f"  {tag:5s} relL2(mean): mean={rel.mean():.4f} std={rel.std():.4f} "
                  f"min={rel.min():.4f} max={rel.max():.4f} "
                  f"rel.spread={(rel.max()-rel.min())/rel.mean()*100:.1f}% "
                  f"slope/idx={sl:+.4f} (corr {r:+.2f})")

    # -------- content statistics: are the two frame sets materially different? ----
    print("\n=== Frame content statistics (subsampled every 64th point) ===")
    fields = ["CH4", "CO", "T", "U_1", "p"]
    frames = sorted(set(ours_scored) | set(base_scored))
    stats = {}
    with h5py.File(H5, "r") as f:
        for fr in frames:
            x = f["fields"][0, fr, ::64, 0, 0, :]  # [~57k, 5]
            stats[fr] = (x.mean(axis=0), x.std(axis=0))
    hdr = "  ".join(f"{n:>12s}" for n in fields)
    print(f"{'frame':>5s} {'set':>5s} {'t[s]':>5s}  std of: {hdr}")
    for fr in frames:
        tag = "ours" if fr in ours_scored else "base"
        line = "  ".join(f"{v:12.4g}" for v in stats[fr][1])
        print(f"{fr:5d} {tag:>5s} {t12(fr):5.0f}  std of: {line}")
    b_std = np.stack([stats[fr][1] for fr in sorted(base_scored)])
    o_std = np.stack([stats[fr][1] for fr in sorted(ours_scored)])
    print("\nmean per-field std, baseline-scored frames:",
          " ".join(f"{v:.4g}" for v in b_std.mean(0)))
    print("mean per-field std, ours-scored frames:    ",
          " ".join(f"{v:.4g}" for v in o_std.mean(0)))
    print("ratio (ours/base):                         ",
          " ".join(f"{v:.3f}" for v in o_std.mean(0) / b_std.mean(0)))


if __name__ == "__main__":
    sys.exit(main())
