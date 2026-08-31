"""Task 1: regenerate the corrupted-scale periodic figures with stable guidance.

Calls the FIXED visualize_s3gm3d (commit 2e6481b: monitor arm alpha_case=0.05 /
beta=0.004, i.e. applied alpha_dps=0.5 at the 1% protocol) on best.pt for four
TUNE-split (odd) snapshots. Output goes to a NEW subdirectory
Evaluation/stable_guidance_figs/ -- the divergence-documentation originals in
Evaluation/epoch_* are left untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3gm_improve_common import RUN, load_run  # noqa: E402  (chdirs to src, sets sys.path)

import s3gm3d as S  # noqa: E402
from ensemble_eval import require_compute_node  # noqa: E402

SNAPS = [1, 13, 25, 37]     # odd = TUNE split; TEST (even) untouched
N_STEPS = 200               # the run's reported sampling_N
K = 2


def main():
    require_compute_node()
    cfg, adapter, bundle, val_set, ck = load_run(ckpt="best")
    epoch = int(ck.get("epoch", 0))
    out_root = RUN / "Evaluation" / "stable_guidance_figs"
    out_root.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    with adapter.evaluation_weights(bundle):
        bundle.model.eval()
        for snap in SNAPS:
            d = out_root / f"snap_{snap:04d}"
            d.mkdir(parents=True, exist_ok=True)
            print(f"[figs] snapshot {snap} -> {d}", flush=True)
            m = S.visualize_s3gm3d(bundle, val_set, d, epoch,
                                   snapshot_index=snap, n_steps=N_STEPS, K=K)
            all_metrics[str(snap)] = m
            print(f"[figs] snap={snap} metrics={json.dumps(m)}", flush=True)

    with open(out_root / "stable_guidance_figs_summary.json", "w") as fh:
        json.dump({
            "checkpoint": "best.pt", "epoch": epoch,
            "guidance_arm": "monitor_jhu_tuned (alpha_case=0.05 -> alpha_dps=0.5, beta=0.004)",
            "n_steps": N_STEPS, "K": K, "snapshots": SNAPS,
            "note": "Regenerated with stable guidance; originals under "
                    "Evaluation/epoch_* are divergence documentation only.",
            "metrics": all_metrics,
        }, fh, indent=2)
    print("[figs] done", flush=True)


if __name__ == "__main__":
    main()
