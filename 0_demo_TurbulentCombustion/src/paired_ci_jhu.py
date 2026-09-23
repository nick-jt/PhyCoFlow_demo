#!/usr/bin/env python3
"""Paired per-snapshot CIs for the 3D (JHU) fleet.

The JHU Evaluation directories hold many arms and the payloads carry no model
field, so the canonical row per method cannot be guessed from filenames. We
instead IDENTIFY each canonical JSON by matching its summary against the
published fleet table (FLEET_SUMMARY_TABLE_2026-08-30.md, recoverable from git
at 464c72e^), which is the same provenance the paper's tab:jhu cites. A row is
accepted only when aggregate rel-L2 AND CRPS both match to 5e-3; anything
ambiguous or unmatched is reported, never silently substituted.

Then, exactly as for the 2D fleets (src/paired_ci_2d.py): all methods are
scored on the same 50 snapshots under the same seeded draws, so differences are
paired. Percentile bootstrap (20k) + Wilcoxon; a gap counts as resolved only if
both agree.

Usage:  python paired_ci_jhu.py [--out FILE.json]
"""
import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "Save_TrainedModel" / "JHU"
B = 20000
RNG = np.random.default_rng(0)

# EXPLICIT mapping method -> canonical eval JSON, each with the value published
# in the fleet table. The path is asserted, not searched: we verify that the
# named file reproduces the published aggregate (within TOL) and refuse it
# otherwise, so a wrong file cannot silently enter the statistics.
CANONICAL = [
    ("DMF-Gen (N29, NFE4)",
     "pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100/Evaluation/canonical_all50_nfe4_K8.json",
     0.593, 0.291),
    ("Latent FM",
     "baseline_latent_fm/Baseline_latent_fm_Stage2_DemoN24_20260828_164541/Evaluation/lfm_canonical_best_K8_nfe4.json",
     0.469, 0.244),
    ("FNO3D (NFE4)",
     "baseline_fno/fno3d_matched_DemoN90_20260828_181849/Evaluation/last_nfe4_K8_all50.json",
     0.586, 0.269),
]
# Present in the fleet table but NOT usable here: no per-snapshot payload was
# transferred (SiT-point, Gen4Turb, DeepONet) or the payload uses a different
# schema without a snapshots[] list (Senseiver: snapshot_ids / n_snapshots).
UNAVAILABLE = {
    "SiT-point (N=32)": "no eval JSON with a snapshots[] list under baseline_sit/",
    "Senseiver": "iclr_protocol_eval_best.json uses a different schema (snapshot_ids)",
    "Gen4Turb (32-step)": "eval lived under the untransferred baselines/ tree",
    "DeepONet": "no canonical per-snapshot JSON in the transferred tree",
}
TOL = 5e-3


def candidates():
    out = []
    for p in glob.glob(str(S / "*" / "**" / "Evaluation" / "*.json"), recursive=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        sn = d.get("snapshots")
        if not (isinstance(sn, list) and len(sn) == 50):
            continue
        # Most JHU payloads carry no top-level summary, so derive the aggregate
        # the same way the fleet table does: mean over the 50 snapshots.
        try:
            rl2 = float(np.mean([x["aggregate"]["rel_l2_mean"] for x in sn]))
            crp = float(np.mean([x["aggregate"]["crps"] for x in sn]))
        except (KeyError, TypeError):
            continue
        out.append((p, d, rl2, crp))
    return out


def identify():
    chosen, unmatched = {}, []
    for label, rel, rl2, crps in CANONICAL:
        f = S / rel
        if not f.exists():
            unmatched.append((label, f"missing file {rel}"))
            print(f"  MISSING  {label:<22} {rel}")
            continue
        d = json.load(open(f))
        sn = d["snapshots"]
        a = float(np.mean([x["aggregate"]["rel_l2_mean"] for x in sn]))
        c = float(np.mean([x["aggregate"]["crps"] for x in sn]))
        ok = abs(a - rl2) <= TOL and abs(c - crps) <= TOL
        print(f"  {'VERIFIED' if ok else 'MISMATCH'} {label:<22} n={len(sn)} "
              f"agg={a:.4f}/{c:.4f} vs published {rl2:.3f}/{crps:.3f}  {Path(rel).name}")
        if ok:
            chosen[label] = (str(f), d)
        else:
            unmatched.append((label, f"published {rl2}/{crps} but file gives {a:.4f}/{c:.4f}"))
    for label, why in UNAVAILABLE.items():
        unmatched.append((label, why))
        print(f"  SKIPPED  {label:<22} {why}")
    return chosen, unmatched


def boot_ci(diff, alpha=0.05):
    idx = RNG.integers(0, diff.size, size=(B, diff.size))
    m = diff[idx].mean(axis=1)
    lo, hi = np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "Paper/pof2026/paired_ci_jhu.json"))
    a = ap.parse_args()
    chosen, unmatched = identify()
    if len(chosen) < 2:
        raise SystemExit("\nnot enough identified rows for a paired comparison")

    series = {}
    for label, (p, d) in chosen.items():
        series[label] = {int(s["snapshot"]): (s["aggregate"]["rel_l2_mean"],
                                              s["aggregate"]["crps"]) for s in d["snapshots"]}
    common = sorted(set.intersection(*(set(v) for v in series.values())))
    print(f"\n{len(series)} identified methods on {len(common)} shared snapshots")
    order = sorted(series, key=lambda m: np.mean([series[m][s][0] for s in common]))
    for m in order:
        r = np.array([series[m][s][0] for s in common])
        print(f"  {m:<22} relL2 {r.mean():.4f} +/- {r.std(ddof=1)/np.sqrt(r.size):.4f}")

    res = {"n_snapshots": len(common), "ranking": order, "pairs": {},
           "unidentified": unmatched}
    print("\npaired differences (row - col), 95% bootstrap CI, rel-L2:")
    for x, y in combinations(order, 2):
        d = np.array([series[x][s][0] - series[y][s][0] for s in common])
        lo, hi = boot_ci(d)
        try:
            pv = float(wilcoxon(d).pvalue)
        except ValueError:
            pv = float("nan")
        resolved = ((lo > 0) or (hi < 0)) and (pv == pv) and pv < 0.05
        res["pairs"][f"{x} - {y}"] = {"mean_diff": float(d.mean()), "ci95": [lo, hi],
                                      "wilcoxon_p": pv, "resolved": bool(resolved)}
        print(f"  {x:<22} - {y:<22} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"p={pv:.1e} {'resolved' if resolved else 'TIE'}")
    lead = order[0]
    tie = [lead] + [m for m in order[1:]
                    if not res["pairs"].get(f"{lead} - {m}", {}).get("resolved", True)]
    res["leader"], res["tie_group"] = lead, tie
    print(f"\n  leader: {lead}; tied with it: {tie[1:] or 'none (lead is resolved)'}")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
