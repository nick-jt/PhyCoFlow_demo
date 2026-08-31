"""Shared setup for the S3GM sampler-upgrade jobs (figures / tuning / final eval).

Loads the completed run's config + checkpoint through the same adapter path as
eval_s3gm3d.py, including the staged-data-path fallback. No repo files edited.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = "/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src"
RUN = Path("/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/Save_TrainedModel/JHU/baseline_s3gm/matched/Baseline_s3gm_Stage1_DemoN94_20260828_224632")

if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.chdir(SRC)

import torch  # noqa: E402
import model_baseline as MB  # noqa: E402
import s3gm3d as S  # noqa: E402


def load_run(ckpt="best", device_str="cuda:0"):
    cfg = MB.validate_and_normalize_config(MB.load_yaml(RUN / "run_config.yaml"))

    _paths = cfg["shared"]["paths"]
    _dp = Path(_paths["data_path"])
    if not _dp.exists():
        _shared = _paths.get("data_path_shared")
        if _shared and Path(_shared).exists():
            print(f"[setup] staged data_path {_dp} absent; using shared {_shared}", flush=True)
            _paths["data_path"] = _shared
        else:
            raise FileNotFoundError(f"data_path {_dp} missing and no shared fallback")

    device = torch.device(device_str)
    print(f"[setup] device={device} "
          f"({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})",
          flush=True)

    stats_path = RUN / "dataset_stats.pt"
    val_set = MB.build_dataset(cfg, split="val", stats_path=stats_path)
    train_set = MB.build_dataset(cfg, split="train", stats_path=stats_path)
    print(f"[setup] val snapshots={len(val_set)} points={val_set.num_points} "
          f"fields={list(val_set.field_names)}", flush=True)

    adapter = S.S3GM3DAdapter()
    bundle = adapter.build_for_training(cfg, device, RUN, train_set, val_set)
    ck_path = Path(ckpt) if os.path.sep in ckpt else RUN / f"{ckpt}.pt"
    ck = MB.safe_torch_load(ck_path, map_location="cpu")
    adapter.load_checkpoint(bundle, ck)
    print(f"[setup] checkpoint={ck_path} epoch={ck.get('epoch')} "
          f"val_loss={ck.get('val_loss')}", flush=True)
    return cfg, adapter, bundle, val_set, ck
