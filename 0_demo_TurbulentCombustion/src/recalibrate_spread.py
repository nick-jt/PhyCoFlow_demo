"""Route-1 post-hoc spread recalibration from existing per-snapshot summary JSONs.

Fits a scalar spread multiplier per (density, channel) on the TUNE split
(odd snapshot indices) and evaluates it frozen on the TEST split (even
indices).  Works on any ensemble_eval.py / eval_latentfm_ensemble.py payload
(``snapshots[].per_field[ch]`` with ``spread`` and ``rmse``), so the identical
wrapper can be applied to the generative baselines -- no new sampling needed.

Because a multiplier alpha rescales the stated ensemble spread linearly, the
fit that achieves spread/error = 1 on TUNE is alpha = RMS(err)/RMS(spread)
on that split.  What CAN be reported from summaries: TEST spread/error before
and after.  What CANNOT: coverage after rescaling (needs per-point ensembles;
see dump_calib_points.py + conformal_recalib.py, route 2).

Usage (login-node OK: JSON only):
    python recalibrate_spread.py \
        --json "$RD/Evaluation/calib_sweep_*.json" [more globs / files ...] \
        --channels Ux Uz --out recalib_scalar.json

TUNE/TEST discipline per PLAN_IMPROVE_2026-08-30: TUNE = odd snapshot
indices, TEST = even.  The reported (primary) number is always TEST.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def _split(snaps: List[dict]) -> Dict[str, List[dict]]:
    tune = [s for s in snaps if int(s["snapshot"]) % 2 == 1]
    test = [s for s in snaps if int(s["snapshot"]) % 2 == 0]
    return {"tune": tune, "test": test}


def _rms(snaps: List[dict], ch: str, key: str) -> float:
    # RMS over snapshots so the alpha that equalizes spread/error is exact
    # under the ratio-of-RMS definition used in the summaries.
    v = np.array([s["per_field"][ch][key] for s in snaps], dtype=np.float64)
    return float(np.sqrt(np.mean(v ** 2)))


def fit_file(path: str, channels: List[str]) -> dict:
    d = json.load(open(path))
    snaps = d["snapshots"]
    parts = _split(snaps)
    if not parts["tune"] or not parts["test"]:
        raise ValueError(f"{path}: TUNE/TEST split empty "
                         f"(n={len(snaps)}; need both parities present)")
    n_obs = d.get("n_obs")
    row: dict = {
        "file": os.path.basename(path),
        "n_obs": n_obs[0] if isinstance(n_obs, list) else n_obs,
        "K": d.get("K"),
        "nfe": d.get("n_steps"),
        "ckpt": d.get("ckpt"),
        "n_tune": len(parts["tune"]),
        "n_test": len(parts["test"]),
        "channels": {},
    }
    for ch in channels:
        sp_tu, er_tu = _rms(parts["tune"], ch, "spread"), _rms(parts["tune"], ch, "rmse")
        sp_te, er_te = _rms(parts["test"], ch, "spread"), _rms(parts["test"], ch, "rmse")
        if sp_tu <= 0:
            raise ValueError(f"{path}:{ch}: non-positive TUNE spread")
        alpha = er_tu / sp_tu
        row["channels"][ch] = {
            "alpha": alpha,
            "sp_err_tune_before": sp_tu / er_tu,
            "sp_err_test_before": sp_te / er_te,
            "sp_err_test_after": alpha * sp_te / er_te,
        }
    # pooled multiplier over the requested channels (one knob per density)
    sp_tu = np.sqrt(np.mean([_rms(parts["tune"], c, "spread") ** 2 for c in channels]))
    er_tu = np.sqrt(np.mean([_rms(parts["tune"], c, "rmse") ** 2 for c in channels]))
    sp_te = np.sqrt(np.mean([_rms(parts["test"], c, "spread") ** 2 for c in channels]))
    er_te = np.sqrt(np.mean([_rms(parts["test"], c, "rmse") ** 2 for c in channels]))
    row["pooled"] = {
        "alpha": er_tu / sp_tu,
        "sp_err_test_before": sp_te / er_te,
        "sp_err_test_after": (er_tu / sp_tu) * sp_te / er_te,
    }
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", nargs="+", required=True,
                   help="payload files or globs (calib_sweep_*.json etc.)")
    p.add_argument("--channels", nargs="+", default=["Ux", "Uz"])
    p.add_argument("--out", default=None, help="output JSON path")
    args = p.parse_args()

    files: List[str] = []
    for g in args.json:
        files.extend(sorted(glob.glob(g)) if any(c in g for c in "*?[") else [g])
    if not files:
        raise SystemExit("no input files matched")

    rows = []
    for f in files:
        try:
            rows.append(fit_file(f, args.channels))
        except Exception as exc:
            print(f"[skip] {f}: {exc}")
    rows.sort(key=lambda r: (r["K"] or 0, r["nfe"] or 0, r["n_obs"] or 0))

    hdr = (f"{'K':>3} {'NFE':>4} {'n_obs':>7} {'ckpt':>10} "
           f"{'alpha(pool)':>11} {'s/e test pre':>12} {'s/e test post':>13}")
    print(hdr)
    for r in rows:
        po = r["pooled"]
        print(f"{r['K'] or 0:3d} {r['nfe'] or 0:4d} {r['n_obs'] or 0:7d} "
              f"{str(r['ckpt']):>10} {po['alpha']:11.3f} "
              f"{po['sp_err_test_before']:12.3f} {po['sp_err_test_after']:13.3f}")
        for ch, c in r["channels"].items():
            print(f"      {ch:>4}: alpha={c['alpha']:.3f} "
                  f"s/e test {c['sp_err_test_before']:.3f} -> {c['sp_err_test_after']:.3f}")
    print("\nNOTE: TEST-split coverage after rescaling is NOT derivable from "
          "summary JSONs; use dump_calib_points.py + conformal_recalib.py.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"channels": args.channels, "split": "tune=odd/test=even",
                       "rows": rows}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
