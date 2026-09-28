"""Pick the best setting among tagged TUNE-split eval summaries.

Scans crps_<prefix>*_summary.json files in an Evaluation dir, ranks by the mean
aggregate rel_l2_mean over the scored (TUNE) snapshots, writes
best_<name>.json with the winning tag and its stamped settings.

    python confild_pick_best.py --eval-dir ... --prefix tuneA_scale --name scale_P
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(args.eval_dir)
                                  / f"crps_{args.prefix}*_summary.json"))):
        s = json.load(open(f))
        aggs = [v["rel_l2_mean"] for v in s["per_snapshot"].values()]
        rows.append({
            "tag": s.get("tag", Path(f).stem),
            "agg_rel_l2_mean": float(np.mean(aggs)),
            "n": len(aggs),
            "dps_scale": s.get("dps_scale"),
            "post_sensor_steps": s.get("post_sensor_steps"),
            "post_sensor_lr": s.get("post_sensor_lr"),
            "stage2_ckpt": s.get("stage2_ckpt"),
            "window1": s.get("window1"),
        })
    if not rows:
        raise SystemExit(f"no crps_{args.prefix}*_summary.json in {args.eval_dir}")
    rows.sort(key=lambda r: r["agg_rel_l2_mean"])
    best = rows[0]
    out = {"best": best, "ranking": rows}
    path = Path(args.eval_dir) / f"best_{args.name}.json"
    json.dump(out, open(path, "w"), indent=1)
    for r in rows:
        print(f"[pick:{args.name}] {r['tag']}: agg={r['agg_rel_l2_mean']:.4f} "
              f"(scale={r['dps_scale']} post={r['post_sensor_steps']}"
              f"@{r['post_sensor_lr']} ckpt={r['stage2_ckpt']})", flush=True)
    print(f"[pick:{args.name}] BEST -> {best['tag']} ({path})", flush=True)


if __name__ == "__main__":
    main()
