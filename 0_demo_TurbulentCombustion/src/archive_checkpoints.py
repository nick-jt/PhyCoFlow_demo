"""Snapshot `last.pt` N times over the tail of a training run.

Runs as a separate CPU job so it needs no restart of, and no coupling to, the
trainer.  `last.pt` is rewritten every `eval_every` epochs, so sampling it on a
timer yields genuinely distinct checkpoints -- enough to measure checkpoint
noise (the spread of the metric across nearby checkpoints) at the end of a run.

Each copy is validated by loading it back; a torn read (the trainer writing
while we copy) is retried rather than silently archived.
"""
from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True, help="literal path or glob; newest match wins")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--interval-s", type=float, default=351.0)
ap.add_argument("--src", default="last.pt")
ap.add_argument("--retries", type=int, default=5)
a = ap.parse_args()

matches = sorted(Path().glob(a.run_dir)) if any(c in a.run_dir for c in "*?[") \
          else [Path(a.run_dir)]
matches = [m for m in matches if m.is_dir()]
if not matches:
    raise SystemExit(f"no run dir matching {a.run_dir}")
run_dir = matches[-1]
src = run_dir / a.src
out = run_dir / "archive"
out.mkdir(parents=True, exist_ok=True)
print(f"[archive] run_dir={run_dir}", flush=True)

manifest = []
for i in range(a.n):
    if i:
        time.sleep(a.interval_s)
    ok = False
    for attempt in range(a.retries):
        if not src.exists():
            time.sleep(20); continue
        dst = out / f"ckpt_{i:02d}.pt"
        try:
            shutil.copy2(src, dst)
            ck = torch.load(dst, map_location="cpu", weights_only=False)
            ep = int(ck.get("epoch", -1))
            final = out / f"ckpt_{i:02d}_epoch{ep:06d}.pt"
            dst.rename(final)
            manifest.append({"index": i, "epoch": ep, "file": final.name,
                             "val_loss": float(ck.get("val_loss", float("nan"))),
                             "t": time.time()})
            print(f"[archive] {i+1}/{a.n} epoch={ep} -> {final.name}", flush=True)
            ok = True
            break
        except Exception as exc:
            print(f"[archive] attempt {attempt+1} failed ({exc}); retrying", flush=True)
            dst.unlink(missing_ok=True)
            time.sleep(20)
    if not ok:
        print(f"[archive] {i+1}/{a.n} FAILED", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"[archive] done: {len(manifest)}/{a.n} checkpoints", flush=True)
