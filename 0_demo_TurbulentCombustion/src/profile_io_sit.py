"""Separate worker-respawn cost from Lustre read cost, and measure whether
node-local staging would help. Standalone; touches no shared module."""
import os, shutil, time, sys
import torch
from model_baseline import (build_dataset, build_dataloader, load_yaml,
                            validate_and_normalize_config, ensure_absolute)

CFG = sys.argv[1]
cfg = validate_and_normalize_config(load_yaml(ensure_absolute(CFG)))
src_path = cfg["shared"]["paths"]["data_path"]
print(f"[io] source {src_path}", flush=True)
print(f"[io] size   {os.path.getsize(src_path)/2**30:.2f} GiB", flush=True)


def sweep(ds, nw, label, n_epochs=3):
    dl = build_dataloader(ds, batch_size=24, num_workers=nw, shuffle=True)
    for ep in range(n_epochs):
        t0 = time.perf_counter(); first = None; nb = 0
        for b in dl:
            if first is None:
                first = time.perf_counter() - t0
            nb += 1
            _ = b["fields"].shape
        tot = time.perf_counter() - t0
        rest = (tot - first) / max(nb - 1, 1)
        print(f"[io] {label} nw={nw} ep{ep}: total={tot:.2f}s "
              f"batch1={first:.2f}s rest={rest:.2f}s batches={nb}", flush=True)


ds = build_dataset(cfg, split="train", stats_path=None)
print(f"[io] train snapshots={len(ds)}", flush=True)

# 1) as training runs it: 4 workers, respawned each epoch
sweep(ds, 4, "LUSTRE")
# 2) no workers at all -> isolates the spawn cost
sweep(ds, 0, "LUSTRE", n_epochs=2)

# 3) node-local staging
dst_dir = f"/tmp/{os.environ.get('USER','u')}/jhu_{os.environ.get('SLURM_JOB_ID','x')}"
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, os.path.basename(src_path))
t0 = time.perf_counter()
shutil.copy2(src_path, dst)
copy_s = time.perf_counter() - t0
print(f"[io] STAGE copy {copy_s:.1f}s "
      f"({os.path.getsize(src_path)/2**30/copy_s:.2f} GiB/s) -> {dst}", flush=True)

cfg2 = validate_and_normalize_config(load_yaml(ensure_absolute(CFG)))
cfg2["shared"]["paths"]["data_path"] = dst
ds2 = build_dataset(cfg2, split="train", stats_path=None)
sweep(ds2, 4, "TMPFS ")
sweep(ds2, 0, "TMPFS ", n_epochs=2)
shutil.rmtree(dst_dir, ignore_errors=True)
print("[io] done", flush=True)
