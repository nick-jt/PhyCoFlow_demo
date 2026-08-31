"""A/B/C the REAL SiT training epoch under three dataloader configs.
Builds the model once; only the loader changes. Reports epoch wall-clock."""
import os, sys, time, statistics as st
import torch
from model_baseline import (build_dataset, build_dataloader, get_baseline_adapter,
                            load_yaml, validate_and_normalize_config, infer_device,
                            ensure_absolute, run_epoch_sit)

cfg = validate_and_normalize_config(load_yaml(ensure_absolute(sys.argv[1])))
device = infer_device(None, cfg["shared"]["device_ids"])
train_set = build_dataset(cfg, split="train", stats_path=None)
val_set = build_dataset(cfg, split="val", stats_path=None)
adapter = get_baseline_adapter("sit")
import pathlib, tempfile
rd = pathlib.Path(tempfile.mkdtemp())
bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=rd,
                                    train_set=train_set, val_set=val_set)
print(f"[abc] params {sum(p.numel() for p in bundle.model.parameters()):,}", flush=True)

N_WARM, N_MEAS = 2, 8
for label, nw, persist in [("nw=4            ", 4, "0"),
                           ("nw=4 persistent ", 4, "1"),
                           ("nw=0            ", 0, "0")]:
    os.environ["JHU_PERSISTENT_WORKERS"] = persist
    dl = build_dataloader(train_set, batch_size=24, num_workers=nw, shuffle=True)
    times = []
    for ep in range(N_WARM + N_MEAS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_epoch_sit(bundle, dl, training=True, epoch=ep)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if ep >= N_WARM:
            times.append(dt)
    print(f"[abc] {label}: median {st.median(times):.2f}s  mean {sum(times)/len(times):.2f}s  "
          f"min {min(times):.2f}  max {max(times):.2f}", flush=True)
    del dl
print("[abc] done", flush=True)
