#!/usr/bin/env python3
"""Collect every 2D fleet-eval JSON into one table (the paper's tab:kolm /
tab:cyl source). Reads Save_TrainedModel/<ds>/<family>/<run>/Evaluation/
<prefix>_<model>_K*.json plus the classical baselines_*.json, and prints
per-method aggregate rel-L2 / CRPS / dispersion / cost, then per-field.
Usage: python summarize_2d_fleet.py [kolmogorov2d|cylinder2d|both]
"""
import json, sys, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STM = ROOT / "Save_TrainedModel"
ORDER = ["dmfgen", "sit", "latentfm", "senseiver", "mlprbf", "geofno", "s3gm"]
NICE = {"dmfgen": "DMF-Gen", "sit": "SiT", "latentfm": "latent-FM",
        "senseiver": "Senseiver", "mlprbf": "MLP-RBF", "geofno": "GeoFNO", "s3gm": "S3GM"}


def load(ds):
    """model -> canonical payload, by payload identity (fleet_select.py)."""
    from fleet_select import load_canonical_fleet
    return load_canonical_fleet(ds, STM)


def fmt(v, n=3):
    return "  n/a " if v is None else f"{v:.{n}f}"


def main(ds):
    print(f"\n########## {ds} ##########")
    runs = load(ds)
    if not runs:
        print("  (no fleet JSONs)"); return
    fields = None
    print(f"{'method':<11} {'relL2':>7} {'CRPS':>7} {'sprd/err':>8} {'cov90':>6} {'K':>3} {'NFE':>4} "
          f"{'infer s':>8} {'GB':>5}  gpu")
    rows = []
    for m in ORDER:
        if m not in runs: continue
        p, d = runs[m]
        s = d["summary"]["aggregate"]; fields = fields or d["field_names"]
        rows.append((m, d, s))
        print(f"{NICE[m]:<11} {fmt(s['rel_l2_mean'],4):>7} {fmt(s['crps'],4):>7} "
              f"{fmt(s.get('spread_error_ratio')):>8} {fmt(s.get('coverage_90'),3):>6} "
              f"{d['K']:>3} {d['nfe']:>4} {d['inference_seconds_per_field_mean']:>8.3f} "
              f"{d['inference_peak_gpu_gb']:>5.2f}  {d.get('gpu','?')[:22]}")
    # classical
    for tag in sorted(glob.glob(str(STM / ds / "baseline_classical" / "classical_baselines_*.json"))):
        d = json.load(open(tag)).get("results", {})
        name = Path(tag).stem.replace("classical_baselines_", "")
        for k, v in d.items():
            if not isinstance(v, dict) or "aggregate" not in v: continue
            a = v["aggregate"]
            print(f"{'[cls] '+k:<11} {fmt(a['rel_l2_mean'],4):>7} {fmt(a['crps'],4):>7} "
                  f"{'':>8} {'':>6} {'':>3} {'':>4} {'':>8} {'':>5}  {name}")
        # all main-protocol files (cylinder ships pod80 + pod20)
    if fields and len(fields) > 1:
        print(f"\nper-field rel-L2 ({', '.join(fields)}):")
        for m, d, s in rows:
            pf = d["summary"]["per_field"]
            print(f"  {NICE[m]:<11} " + "  ".join(f"{f}={pf[f]['rel_l2_mean']:.4f}" for f in fields))
        print(f"  observed={d['observed_fields']}  unobserved={d['unobserved_fields']}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for ds in (["kolmogorov2d", "cylinder2d"] if which == "both" else [which]):
        main(ds)
