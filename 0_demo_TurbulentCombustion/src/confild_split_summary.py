"""Aggregate per-snapshot CoNFiLD eval JSONs into ALL / TEST(even) / TUNE(odd)
split means. Usage:

    python confild_split_summary.py --eval-dir <.../Evaluation> --tag <tag>
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


def split_means(eval_dir, tag):
    files = sorted(glob.glob(str(Path(eval_dir) / f"crps_{tag}_snap*.json")))
    rows = {}
    for f in files:
        m = json.load(open(f))
        snap = int(m["snapshot"]) if "snapshot" in m else \
            int(re.search(r"snap(\d+)\.json$", f).group(1))
        rows[snap] = m
    out = {}
    for split, keep in (("all", lambda s: True),
                        ("test_even", lambda s: s % 2 == 0),
                        ("tune_odd", lambda s: s % 2 == 1)):
        sel = {s: m for s, m in rows.items() if keep(s)}
        if not sel:
            continue
        fields = sorted(next(iter(sel.values()))["per_field"])
        entry = {"n": len(sel), "snaps": sorted(sel)}
        for key in ("rel_l2_mean", "rel_l2_single", "crps", "spread_error_ratio"):
            entry[f"agg_{key}"] = float(np.mean(
                [m["aggregate"][key] for m in sel.values()]))
            entry[f"per_field_{key}"] = {
                f: float(np.mean([m["per_field"][f][key] for m in sel.values()]))
                for f in fields}
        out[split] = entry
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()
    out = split_means(args.eval_dir, args.tag)
    out["tag"] = args.tag
    out["label"] = args.label
    path = Path(args.eval_dir) / f"split_summary_{args.tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    for split in ("test_even", "tune_odd", "all"):
        if split not in out:
            continue
        e = out[split]
        pf = e["per_field_rel_l2_mean"]
        print(f"[split] tag={args.tag} split={split} n={e['n']} "
              f"agg={e['agg_rel_l2_mean']:.4f} "
              + " ".join(f"{f}={v:.4f}" for f, v in pf.items()), flush=True)
    print(f"[split] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
