"""Archive `last.pt` at a fixed epoch grid over the tail of a training run.

The trainer only ever writes `last.pt` / `best.pt` (overwritten), so no epoch
window exists anywhere in the project. This runs alongside the trainer as a
plain polling process -- it never touches the trainer -- and copies `last.pt`
whenever its recorded epoch crosses the next target. Copies are verified by
loading them back, so a torn read is retried rather than archived.
"""
from __future__ import annotations
import argparse, csv, glob, json, shutil, time
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True, help="glob for the run dir; newest match wins")
ap.add_argument("--start", type=int, default=5700)
ap.add_argument("--stop", type=int, default=6000)
ap.add_argument("--every", type=int, default=30)
ap.add_argument("--poll-s", type=float, default=20.0)
ap.add_argument("--max-wait-h", type=float, default=30.0)
a = ap.parse_args()

targets = list(range(a.start, a.stop + 1, a.every))
print(f"[arch] targets={targets}", flush=True)

t_start = time.time()
run_dir = None
while run_dir is None:
    m = sorted(Path(p) for p in glob.glob(a.glob) if Path(p).is_dir())
    if m:
        run_dir = m[-1]
    else:
        time.sleep(30)
    if time.time() - t_start > a.max_wait_h * 3600:
        raise SystemExit("[arch] run dir never appeared")
print(f"[arch] run_dir={run_dir}", flush=True)
out = run_dir / "archive"; out.mkdir(parents=True, exist_ok=True)
src = run_dir / "last.pt"

def csv_epoch():
    c = sorted(run_dir.glob("Loss_*/losses.csv"))
    if not c:
        return -1
    try:
        rows = list(csv.DictReader(open(c[-1])))
        return int(rows[-1]["epoch"]) if rows else -1
    except Exception:
        return -1

manifest = []
ti = 0
while ti < len(targets):
    if time.time() - t_start > a.max_wait_h * 3600:
        print("[arch] wall limit reached; stopping", flush=True); break
    ep_csv = csv_epoch()
    if ep_csv < targets[ti] - a.every:
        # far from the window: poll cheaply on the csv only
        time.sleep(max(a.poll_s, 60.0)); continue
    if not src.exists():
        time.sleep(a.poll_s); continue
    tmp = out / "_tmp.pt"
    try:
        shutil.copy2(src, tmp)
        ck = torch.load(tmp, map_location="cpu", weights_only=False)
        ep = int(ck.get("epoch", -1))
    except Exception as exc:
        print(f"[arch] torn read ({exc}); retry", flush=True)
        tmp.unlink(missing_ok=True); time.sleep(a.poll_s); continue
    if ep >= targets[ti]:
        dst = out / f"ckpt_epoch{ep:06d}.pt"
        if dst.exists():
            tmp.unlink(missing_ok=True)
        else:
            tmp.rename(dst)
            manifest.append({"target": targets[ti], "epoch": ep, "file": dst.name,
                             "val_loss": float(ck.get("val_loss", float("nan")) or float("nan")),
                             "t": time.time()})
            json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
            print(f"[arch] archived epoch {ep} (target {targets[ti]}) -> {dst.name}", flush=True)
        while ti < len(targets) and targets[ti] <= ep:
            ti += 1
    else:
        tmp.unlink(missing_ok=True)
        time.sleep(a.poll_s)
print(f"[arch] done, {len(manifest)} checkpoints archived", flush=True)
