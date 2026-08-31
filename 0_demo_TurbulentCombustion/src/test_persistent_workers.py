"""Verify JHU_PERSISTENT_WORKERS=1 (a) actually saves the respawn cost and
(b) does NOT freeze the octahedral augmentation across epochs."""
import os, sys, time, hashlib
import torch
from model_baseline import (build_dataset, build_dataloader, load_yaml,
                            validate_and_normalize_config, ensure_absolute)

cfg = validate_and_normalize_config(load_yaml(ensure_absolute(sys.argv[1])))
ds = build_dataset(cfg, split="train", stats_path=None)


def run(tag, n_epochs=3, shuffle=False):
    dl = build_dataloader(ds, batch_size=24, num_workers=4, shuffle=shuffle)
    hashes, totals = [], []
    for ep in range(n_epochs):
        t0 = time.perf_counter(); first = None; h = None
        for i, b in enumerate(dl):
            if first is None:
                first = time.perf_counter() - t0
                # fingerprint of the FIRST batch: with shuffle=False the same
                # snapshots come back, so any change is the augmentation.
                h = hashlib.sha1(b["fields"].numpy().tobytes()).hexdigest()[:16]
        tot = time.perf_counter() - t0
        totals.append(tot); hashes.append(h)
        print(f"[pw] {tag} ep{ep}: total={tot:.2f}s batch1={first:.2f}s hash={h}", flush=True)
    return totals, hashes


os.environ["JHU_PERSISTENT_WORKERS"] = "0"
t_off, h_off = run("OFF")
os.environ["JHU_PERSISTENT_WORKERS"] = "1"
t_on, h_on = run("ON ")

print(f"[pw] mean epoch OFF={sum(t_off)/len(t_off):.2f}s  ON={sum(t_on)/len(t_on):.2f}s "
      f"saving={sum(t_off)/len(t_off)-sum(t_on)/len(t_on):.2f}s/epoch", flush=True)
print(f"[pw] augmentation varies across epochs?  OFF={len(set(h_off))==len(h_off)}  "
      f"ON={len(set(h_on))==len(h_on)}   (must be True for both)", flush=True)
assert len(set(h_on)) == len(h_on), "AUGMENTATION FROZEN with persistent_workers -- do not use"
print("[pw] PASS", flush=True)
