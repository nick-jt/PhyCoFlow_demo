"""CPU preflight for the SHIFT-WING baseline wiring (no GPU, no SLURM).

Builds all three point-native baselines from their wing configs, reports the
achieved parameter count and step budget, then pushes one real surface-pool
observation tuple through each forward. Catches config/shape breaks in
seconds instead of after a queue wait. Run from `src/`:

    python wing_baseline_preflight.py
"""
import copy
import math
from pathlib import Path

import torch

import model_baseline as MB
from helpers_baseline import collate_snapshots

ROOT = Path(__file__).resolve().parents[1]
CFGS = {
    "senseiver": "config_baseline_wing_senseiver.yaml",
    "mlp_rbf": "config_baseline_wing_mlprbf.yaml",
    "sit": "config_baseline_SiT_wing.yaml",
}

ds = None
for name, fn in CFGS.items():
    cfg = MB.validate_and_normalize_config(MB.load_yaml(ROOT / "Save_config" / fn))
    stage = MB.resolve_stage_config(cfg)
    assert MB.surface_pool_enabled(cfg), name
    if ds is None:
        ds = MB.build_dataset(cfg, split="val", stats_path=Path("/dev/null"))
        print(f"[data] val={len(ds)} num_points={ds.num_points} "
              f"num_fields={ds.num_fields} n_obs_field_types={ds.n_obs_field_types} "
              f"grid_shape={ds.grid_shape}")
    adapter = MB.get_baseline_adapter(name)
    bundle = adapter.build_for_training(cfg=cfg, device=torch.device("cpu"),
                                        run_dir=Path("/tmp"), train_set=ds, val_set=ds)
    n = sum(p.numel() for p in bundle.model.parameters() if p.requires_grad)
    print(f"[{name}] trainable_params={n:,}  "
          f"epochs={stage['training']['epochs']} batch={stage['training']['batch_size']} "
          f"steps/epoch={-(-600 // int(stage['training']['batch_size']))} "
          f"total_steps={4000 * (-(-600 // int(stage['training']['batch_size'])))}")

    # One real forward on 2 cases through the surface-pool conditioning.
    batch = collate_snapshots([ds[0], ds[1]])
    assert "obs_pool_coords" in batch and "param_values" in batch, "collate dropped pool keys"
    oc, ov, om, oi, ofid = MB.build_wing_condition(
        batch, torch.device("cpu"),
        n_obs_min=cfg["shared"]["conditioning"]["n_obs_min_list"],
        n_obs_max=cfg["shared"]["conditioning"]["n_obs_max_list"])
    print(f"  [cond] {MB.describe_wing_condition(om[:1], ofid[:1])}")
    q = batch["coords"][:, :256]
    with torch.no_grad():
        if name == "sit":
            val, sup = MB.wing_fill_nodes(q, oc, ov, om, ofid, n_cond_ch=9)
            print(f"  [fill] value={tuple(val.shape)} support={tuple(sup.shape)} "
                  f"param_support={float(sup[0, :, 7].min()):.1f}..{float(sup[0, :, 7].max()):.1f}")
            out = bundle.model(torch.randn(2, 256, 4), torch.rand(2),
                               coords=q, obs_value_nodes=val, obs_mask_nodes=sup)
        else:
            out = bundle.model(q, oc, ov, om, ofid)
    assert torch.isfinite(out).all(), f"{name}: non-finite forward"
    print(f"  [fwd] out={tuple(out.shape)} finite=True")

    # One real optimizer step through the adapter's own run_epoch, on a
    # shrunken copy of the config (batch 1, tiny query/token budget, tiny
    # sensor budget) so the exact training code path is exercised on CPU
    # before any GPU time is spent. This is where a wrong channel count or a
    # scatter-vs-fill mistake actually shows up.
    if name == "sit" and torch.cuda.is_available():
        # SiTAdapter.run_epoch calls torch.cuda.reset_peak_memory_stats(
        # bundle.device) whenever CUDA exists, which rejects a cpu device.
        # Pre-existing, unrelated to the wing wiring; the GPU smoke covers it.
        print("  [step] skipped on CPU: SiTAdapter.run_epoch requires a CUDA "
              "bundle.device when CUDA is present (covered by the GPU smoke)")
        continue

    small = copy.deepcopy(cfg)
    small["shared"]["conditioning"]["n_obs_min_list"] = [8, 4, 4, 4]
    small["shared"]["conditioning"]["n_obs_max_list"] = [16, 8, 8, 8]
    st = small[f"{name}_params"]["training"]
    st["n_query_points"] = 256
    if name == "sit":
        small["sit_params"]["architecture"]["node_subsample"] = 256
    bundle2 = adapter.build_for_training(cfg=small, device=torch.device("cpu"),
                                         run_dir=Path("/tmp"), train_set=ds,
                                         val_set=ds)
    loader = [collate_snapshots([ds[0]])]
    before = [p.detach().clone() for p in bundle2.model.parameters()]
    out_epoch = adapter.run_epoch(bundle2, loader, training=True, epoch=1)
    loss = out_epoch[0] if isinstance(out_epoch, tuple) else out_epoch
    moved = sum(int(not torch.equal(a, b))
                for a, b in zip(before, bundle2.model.parameters()))
    assert math.isfinite(loss), f"{name}: non-finite training loss {loss}"
    assert moved > 0, f"{name}: optimizer step changed no parameters"
    print(f"  [step] loss={loss:.6e} params_updated={moved}")

print("ALL OK")
