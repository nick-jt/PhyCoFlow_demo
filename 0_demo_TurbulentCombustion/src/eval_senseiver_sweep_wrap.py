"""Wrapper around the canonical eval_senseiver_iclr.py driver.

Adds two things WITHOUT touching the canonical driver (write-guarded repo):

1. If SEN_LOCAL_IDW=1, imports sen_sweep_fixes so the "Senseiver+local"
   (IDW residual) patch is active -- checkpoints trained with the patch
   carry a `local_gate` parameter and will not load without it.
2. Records the per-snapshot headline metrics (the canonical JSON stores only
   means) and writes a companion *_splits.json with TUNE (odd val index) /
   TEST (even val index) aggregates, per the fleet tuning-hygiene rule:
   TEST-even is the primary reported number, full-50 secondary.

The canonical driver's own JSON is byte-identical to an unwrapped run for
the non-local arms: ensemble_metrics is wrapped observationally only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SRC = ("/home/ntricard/generative_reconstruction/temp/"
       "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (SRC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import sen_sweep_fixes  # noqa: F401  (env-gated; inert unless SEN_* set)
if os.environ.get("SEN_LOCAL_XATTN", "0") == "1":
    import sen_local_xattn  # noqa: F401  (secondary variant, env-gated)
import eval_senseiver_iclr as EV

_calls = []
_orig_em = EV.ensemble_metrics


def _em(pred, true, names):
    r = _orig_em(pred, true, names)
    _calls.append(r)
    return r


EV.ensemble_metrics = _em


def _get_arg(flag, default=None):
    argv = sys.argv
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return default


def main():
    EV.main()

    out = _get_arg("--out")
    if out is None:
        out = str(Path(_get_arg("--run-dir")) / "Evaluation" / "iclr_protocol_eval.json")
    with open(out, "r", encoding="utf-8") as h:
        results = json.load(h)

    snap_ids = results["snapshot_ids"]
    n = len(snap_ids)
    skip_floor = "--skip-nn-floor" in sys.argv
    stride = 1 if skip_floor else 2
    # headline loop call order: per snapshot -> model em [, floor em]; the
    # sensor sweep's calls come after index n*stride and are ignored here.
    model_calls = _calls[0:n * stride:stride]
    assert len(model_calls) == n, (len(model_calls), n)

    per_snap = []
    for sid, m in zip(snap_ids, model_calls):
        per_snap.append({
            "snapshot_id": int(sid),
            "split": "TUNE(odd)" if sid % 2 == 1 else "TEST(even)",
            "aggregate": m["aggregate"],
            "per_field": m["per_field"],
        })

    def _agg(rows):
        import numpy as np
        keys = ["rel_l2_mean", "rel_l2_single", "crps", "rmse"]
        fields = list(rows[0]["per_field"].keys()) if rows else []
        return {
            "n_snapshots": len(rows),
            "aggregate": {k: float(np.nanmean([r["aggregate"][k] for r in rows]))
                          for k in keys},
            "per_field": {f: {k: float(np.nanmean([r["per_field"][f][k] for r in rows]))
                              for k in keys} for f in fields},
        }

    even = [r for r in per_snap if r["snapshot_id"] % 2 == 0]
    odd = [r for r in per_snap if r["snapshot_id"] % 2 == 1]
    splits = {
        "source_eval_json": str(out),
        "policy": "TUNE = cube-3 odd val indices (selection); "
                  "TEST = even indices (primary reported); full set secondary",
        "TEST_even": _agg(even) if even else None,
        "TUNE_odd": _agg(odd) if odd else None,
        "full": _agg(per_snap),
        "per_snapshot": per_snap,
    }
    sp = str(out).replace(".json", "_splits.json")
    with open(sp, "w", encoding="utf-8") as h:
        json.dump(splits, h, indent=2)
    print(f"[eval-wrap] wrote {sp}", flush=True)
    if even:
        print(f"[eval-wrap] TEST(even) n={len(even)} "
              f"rel_l2_mean={splits['TEST_even']['aggregate']['rel_l2_mean']:.5f}",
              flush=True)
    if odd:
        print(f"[eval-wrap] TUNE(odd) n={len(odd)} "
              f"rel_l2_mean={splits['TUNE_odd']['aggregate']['rel_l2_mean']:.5f}",
              flush=True)


if __name__ == "__main__":
    main()
