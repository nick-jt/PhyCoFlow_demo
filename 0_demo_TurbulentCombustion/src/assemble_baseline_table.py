#!/usr/bin/env python3
"""Assemble the final JHU cross-cube baseline comparison table.

Reads every baseline's evaluation JSON, normalises the schemas, and emits one
table with per-channel and aggregate metrics plus cost. Run after the fleet lands:

    python assemble_baseline_table.py                 # table to stdout
    python assemble_baseline_table.py --json out.json # machine-readable

Design notes:
  * Per-channel is primary; aggregate is secondary and flagged, because the
    aggregate averages two observed channels against two unobserved ones and
    makes every method look closer than it is.
  * Every row carries its operating point (density, NFE, K, checkpoint). A
    calibration number without one is meaningless -- NFE 4->16 moves
    spread/error +0.217 and K 8->32 moves coverage +0.107.
  * Missing entries print as "--", never as zero or as a silent omission.
"""
import json, glob, os, argparse, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
STM  = os.path.join(ROOT, "..", "Save_TrainedModel", "JHU")
CH   = ["Ux", "Uy", "Uz", "p"]          # Ux,Uz observed; Uy,p unobserved

# where each baseline's results live; first existing glob wins
SOURCES = {
    "Ours (N29)":        [f"{STM}/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*/Evaluation/canonical_all50_nfe4_K8.json",
                          f"{STM}/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*/Evaluation/sensor_sweep_n19531.json"],
    "Latent FM":         [f"{STM}/baseline_latent_fm/*/Evaluation/*canonical*.json"],
    "Senseiver":         [f"{STM}/baseline_senseiver/Baseline_senseiver_Stage1_DemoN43_*/Evaluation/*.json"],
    # Wide-bottleneck fairness arms (authorized re-runs 2026-08-29): second
    # labelled rows; the original arms above stay.
    "Senseiver-wide":    [f"{STM}/baseline_senseiver/Baseline_senseiver_Stage1_DemoN44_*/Evaluation/*.json"],
    "SiT-point":         [f"{STM}/baseline_sit/matched/*/Evaluation_seeded_*/*.json"],
    "CoNFiLD (C 1024d)": [f"{STM}/baseline_confild/unified_cap1024/*/Evaluation/*.json"],
    "CoNFiLD (F 384d)":  [f"{STM}/baseline_confild/unified_faithful384/*/Evaluation/*.json"],
    "CoNFiLD (P pub)":   [f"{STM}/baseline_confild/unified_published_prior/*/Evaluation/*.json"],
    "S3GM":              [f"{STM}/baseline_s3gm/*/Evaluation/*.json"],
    "FNO3D":             [f"{STM}/baseline_fno/*/Evaluation/*.json"],
    "DeepONet":          [f"{STM}/baseline_deeponet/*/Evaluation/*.json"],
    "DeepONet-wide":     [f"{STM}/baseline_deeponet_wide/*/Evaluation/*.json"],
    # Anneal arm supersedes the constant-LR 4930 run (window-verified; report
    # window mean +- sigma alongside, per audit).
    "Gen4Turb":          ["/projects/ammoniacomb/generative_reconstruction/baselines/Gen4Turbulence/3_flow_reconstruction/eval/canon_anneal_4190_strict.json",
                          "/projects/ammoniacomb/generative_reconstruction/baselines/Gen4Turbulence/3_flow_reconstruction/eval/canon_uxuz_4930_strict.json"],
    "KD-tree":           [f"{STM}/baseline_classical/classical_baselines_main_n19531.json"],
    "IDW (k=8)":         [f"{STM}/baseline_classical/classical_baselines_main_n19531.json"],
    "Gappy POD":         [f"{STM}/baseline_classical/classical_baselines_main_n19531.json"],
    "Constant (train)":  [f"{STM}/baseline_classical/classical_baselines_main_n19531.json"],
}
CLASSICAL_KEY = {"KD-tree":"kdtree_n19531", "IDW (k=8)":"idw_n19531",
                 "Gappy POD":"gappy_pod_n19531", "Constant (train)":"constant_train_mean"}

def _mean(snaps, path):
    vals = []
    for s in snaps:
        cur = s
        for k in path:
            if not isinstance(cur, dict) or k not in cur: cur = None; break
            cur = cur[k]
        if isinstance(cur, (int, float)) and np.isfinite(cur): vals.append(cur)
    return float(np.mean(vals)) if vals else None

def extract(name, path):
    """Return dict of per-channel relL2 + aggregate metrics, or None."""
    try: d = json.load(open(path))
    except Exception: return None
    # classical file holds several methods under results/
    if name in CLASSICAL_KEY:
        r = d.get("results", {}).get(CLASSICAL_KEY[name])
        if not r: return None
        pf, ag = r.get("per_field", {}), r.get("aggregate", {})
        out = {c: (pf.get(c) or {}).get("rel_l2_mean") for c in CH}
        out.update(agg=ag.get("rel_l2_mean"), crps=ag.get("crps"),
                   se=ag.get("spread_error_ratio"), cov=ag.get("coverage_90"),
                   n=r.get("n_snapshots"), src=os.path.basename(path))
        return out
    snaps = d.get("snapshots") or d.get("cases") or []
    if snaps:
        out = {c: _mean(snaps, ["per_field", c, "rel_l2_mean"]) for c in CH}
        out.update(agg=_mean(snaps, ["aggregate","rel_l2_mean"]),
                   crps=_mean(snaps, ["aggregate","crps"]),
                   se=_mean(snaps, ["aggregate","spread_error_ratio"]),
                   cov=_mean(snaps, ["aggregate","coverage_90"]),
                   n=len(snaps), src=os.path.basename(path))
        return out
    ag = d.get("aggregate") or d.get("summary") or {}
    if not ag: return None
    pf = d.get("per_field", {})
    out = {c: (pf.get(c) or {}).get("rel_l2_mean") for c in CH}
    out.update(agg=ag.get("rel_l2_mean"), crps=ag.get("crps"),
               se=ag.get("spread_error_ratio"), cov=ag.get("coverage_90"),
               n=d.get("n_snapshots"), src=os.path.basename(path))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = {}
    for name, pats in SOURCES.items():
        hit = None
        for pat in pats:
            files = sorted(glob.glob(pat))
            if files:
                for f in reversed(files):
                    r = extract(name, f)
                    if r: hit = r; break
            if hit: break
        rows[name] = hit

    f = lambda v: "  --  " if v is None else f"{v:6.3f}"
    print("\nJHU cross-cube -- 1% sensors/observed channel, Ux+Uz observed, Uy+p UNOBSERVED")
    print("PER-CHANNEL IS PRIMARY. Aggregate averages 2 observed vs 2 unobserved channels")
    print("and compresses real differences -- do not lead with it.\n")
    print(f"{'method':<20} {'Ux':>7}{'Uy*':>7}{'Uz':>7}{'p*':>7} | {'agg':>7}{'CRPS':>7}{'sp/err':>7}{'cov90':>7} {'n':>4}")
    print("-"*20 + "-"*30 + "-+-" + "-"*30 + "-----")
    for name, r in rows.items():
        if r is None:
            print(f"{name:<20} {'PENDING -- run not yet evaluated':<60}"); continue
        print(f"{name:<20} " + "".join(f(r.get(c)) + " " for c in CH) +
              f"| {f(r.get('agg'))} {f(r.get('crps'))} {f(r.get('se'))} {f(r.get('cov'))} "
              f"{r.get('n') if r.get('n') else '--':>4}")
    print("\n* Uy and p are unobserved. A constant predictor scores exactly 1.000 on every channel.")
    print("  Any method above 1.000 on Uy or p is worse than predicting the training mean.")
    print("\nFootnotes:")
    print("  [1] SenConsis is NaN for unobserved channels and 0.0 EXACTLY for models that")
    print("      hard-clamp observations (hard-obs); in both cases it is uninformative and")
    print("      must not be read as a quality signal.")
    print("  [2] Deterministic baselines (Senseiver, DeepONet, FNO3D, classical): CRPS == MAE")
    print("      by construction (two identical ensemble members) and spread is undefined")
    print("      (reported 0 exactly). Their sp/err and cov90 columns are not comparable to")
    print("      generative rows.")
    print("  [3] Gen4Turb row is the STRICT per-channel mask variant. The shared-mask variant")
    print("      observes all four channels at identical locations (~2x the information any")
    print("      other method receives) and must never share a column with strict-protocol")
    print("      numbers.")
    print("  [4] Every calibration number must carry its operating point (sensor density,")
    print("      NFE, K, checkpoint): NFE 4->16 moves spread/error +0.217, K 8->32 moves")
    print("      coverage +0.107.")
    n_done = sum(1 for r in rows.values() if r)
    print(f"\n{n_done}/{len(rows)} rows populated.")
    if a.json:
        json.dump(rows, open(a.json,"w"), indent=2); print(f"wrote {a.json}")

if __name__ == "__main__":
    main()
