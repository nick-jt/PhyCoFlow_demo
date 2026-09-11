"""Pick each method's CANONICAL 2D fleet row by what the payload says it is.

The read-side twin of artifact_guard.py. Every script that turned a directory of
eval JSONs into paper numbers chose files by filename glob, last-file-wins. On
2026-09-10 that silently ingested operator runs (kolm_fleet_occl0.25_*) into
tab:kolm -- DMF-Gen read 0.660 (occluded) instead of 0.487 -- and only SiT and
S3GM survived, because their clean files happened to sort last. Density-override
files (kolm_fleet_ovr_n65_*) were being ingested the same way.

A row here is canonical only if its PAYLOAD says so: clean protocol stamp,
canonical sensor count, canonical K (8 generative / 1 deterministic), and the
canonical frame count. Filenames are not consulted for meaning. If more than one
file still qualifies for a model, that is an error, not a tie to break by sort
order.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

CANON = {
    "kolmogorov2d": {"prefix": "kolm_fleet", "n_obs": (655,), "n_frames": 50},
    "cylinder2d":   {"prefix": "cyl_fleet", "n_obs": (238,), "n_frames": 50},
}
PROTOCOL = "kolm2d_matched_v1"          # the 2D evaluator's clean stamp
DETERMINISTIC = {"senseiver", "mlp_rbf", "mlprbf", "geofno"}


def _n_obs(v):
    if v is None:
        return None
    return tuple(sorted({int(x) for x in (v if isinstance(v, (list, tuple)) else [v])}))


def why_not_canonical(d: dict, ds: str) -> str | None:
    """None if canonical, else the reason it is not."""
    c = CANON[base_key(ds)]
    if d.get("protocol") != PROTOCOL:
        return f"protocol={d.get('protocol')}"
    if _n_obs(d.get("n_obs")) != c["n_obs"]:
        return f"n_obs={d.get('n_obs')}"
    m = str(d.get("model", ""))
    want_k = 1 if m in DETERMINISTIC else 8
    if int(d.get("K", -1)) != want_k:
        return f"K={d.get('K')} (canonical {want_k})"
    if int(d.get("n_frames", -1)) != c["n_frames"]:
        return f"n_frames={d.get('n_frames')}"
    return None


def base_key(ds: str) -> str:
    """kolmogorov2d_seed7 -> kolmogorov2d: a seed replicate is canonical on the
    same identity as its base fleet. Other suffixes (e.g. _fullbudget) are NOT
    replicates and have no canonical entry."""
    return ds.split("_seed")[0]


def load_canonical_fleet(ds: str, root: Path, verbose: bool = False) -> dict:
    """{model: payload} for the canonical rows of one 2D dataset key."""
    c = CANON[base_key(ds)]
    found: dict[str, list] = {}
    for p in sorted(glob.glob(str(root / ds / "*" / "*" / "Evaluation" / f"{c['prefix']}_*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or "summary" not in d:
            continue
        reason = why_not_canonical(d, base_key(ds))
        if reason:
            if verbose:
                print(f"  [skip] {Path(p).name}: {reason}")
            continue
        found.setdefault(str(d["model"]).replace("_", ""), []).append((p, d))
    dup = {m: [Path(p).name for p, _ in v] for m, v in found.items() if len(v) > 1}
    if dup:
        raise SystemExit(f"[fleet_select] more than one canonical row for {dup} -- "
                         "refusing to pick by sort order")
    return {m: v[0][1] for m, v in found.items()}
