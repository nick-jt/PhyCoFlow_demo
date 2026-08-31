"""Driver: TUNE-split stage-2 checkpoint selection, then the final eval.

1. For every retained ckpt_step*.pt in --s2dir, run the extended canonical eval
   on the TUNE snapshots (K --k-tune, DPS scale from --scale-from) and rank by
   aggregate rel_l2_mean.
2. Run the final full-50 K=8 eval with the selected checkpoint, the tuned DPS
   scale and (optionally, --corr-from) the tuned sensor-consistency correction.
3. Emit the ALL/TEST/TUNE split summary.

Idempotent: existing summaries are reused, so a resubmitted job continues.
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def run(cmd):
    print("[driver] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def tune_agg(summary_path):
    s = json.load(open(summary_path))
    return float(np.mean([v["rel_l2_mean"] for v in s["per_snapshot"].values()]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--s2dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tune-snaps", type=int, nargs="+",
                   default=[1, 7, 13, 19, 25, 31])
    p.add_argument("--k-tune", type=int, default=2)
    p.add_argument("--scale-from", default="",
                   help="best_*.json with best.dps_scale; empty = 1.0")
    p.add_argument("--corr-from", default="",
                   help="best_*.json with best.post_sensor_steps/lr; empty = off")
    p.add_argument("--final-K", type=int, default=8)
    p.add_argument("--final-n", type=int, default=50)
    p.add_argument("--final-tag", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()

    out = Path(args.out_dir)
    (out / "Evaluation").mkdir(parents=True, exist_ok=True)
    eval_dir = out / "Evaluation"

    scale = 1.0
    if args.scale_from and Path(args.scale_from).exists():
        scale = float(json.load(open(args.scale_from))["best"]["dps_scale"])
        print(f"[driver] tuned DPS scale from {args.scale_from}: {scale}", flush=True)
    corr_steps, corr_lr = 0, 5.0e-3
    if args.corr_from and Path(args.corr_from).exists():
        best = json.load(open(args.corr_from))["best"]
        corr_steps = int(best.get("post_sensor_steps") or 0)
        corr_lr = float(best.get("post_sensor_lr") or 5.0e-3)
        print(f"[driver] tuned correction from {args.corr_from}: "
              f"steps={corr_steps} lr={corr_lr}", flush=True)

    ckpts = sorted(glob.glob(str(Path(args.s2dir) / "ckpt_step*.pt")))
    if not ckpts:
        raise SystemExit(f"no ckpt_step*.pt in {args.s2dir}")
    print(f"[driver] {len(ckpts)} retained checkpoints", flush=True)

    ranking = []
    for ck in ckpts:
        step = Path(ck).stem.replace("ckpt_", "")
        tag = f"sel_{step}"
        summary = eval_dir / f"crps_{tag}_summary.json"
        if not summary.exists():
            run([sys.executable, HERE / "confild_eval_unified2.py",
                 "--stage1-ckpt", args.stage1_ckpt, "--stage2-ckpt", ck,
                 "--out-dir", out, "--tag", tag,
                 "--snaps", *args.tune_snaps, "--K", args.k_tune,
                 "--dps-scale", scale, "--no-figs"])
        ranking.append({"ckpt": ck, "tag": tag, "agg": tune_agg(summary)})
        print(f"[driver] {step}: TUNE agg={ranking[-1]['agg']:.4f}", flush=True)

    ranking.sort(key=lambda r: r["agg"])
    best_ck = ranking[0]["ckpt"]
    json.dump({"ranking": ranking, "selected": best_ck, "dps_scale": scale,
               "corr_steps": corr_steps, "corr_lr": corr_lr,
               "tune_snaps": args.tune_snaps, "k_tune": args.k_tune},
              open(eval_dir / f"ckpt_selection_{args.final_tag}.json", "w"), indent=1)
    print(f"[driver] SELECTED {best_ck} (TUNE agg {ranking[0]['agg']:.4f})",
          flush=True)

    final_cmd = [sys.executable, HERE / "confild_eval_unified2.py",
                 "--stage1-ckpt", args.stage1_ckpt, "--stage2-ckpt", best_ck,
                 "--out-dir", out, "--tag", args.final_tag,
                 "--n-snapshots", args.final_n, "--K", args.final_K,
                 "--dps-scale", scale]
    if corr_steps > 0:
        final_cmd += ["--post-sensor-steps", corr_steps,
                      "--post-sensor-lr", corr_lr]
    if not (eval_dir / f"crps_{args.final_tag}_summary.json").exists():
        run(final_cmd)
    run([sys.executable, HERE / "confild_split_summary.py",
         "--eval-dir", eval_dir, "--tag", args.final_tag, "--label", args.label])


if __name__ == "__main__":
    main()
