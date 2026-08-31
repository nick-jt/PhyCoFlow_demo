"""CoNFiLD Stage A failure isolation, parts (i) and (ii).

(i)  CODEC CEILING -- optimize a latent against the FULL held-out field through
     the frozen stage-1 decoder. This is the best any conditional sampler could
     possibly do with this codec, per arm.
(ii) SENSOR-ONLY -- optimize a latent against ONLY the 39,062 canonical sensor
     values (19,531 per observed channel, fields Ux and Uz), no diffusion
     prior. The gap (i)->(ii) is the pure information limit of the sensors
     under this codec; the gap (ii)->DPS is prior/guidance failure.

Snapshots are TEST-split fields (even cube-3 val indices) -- these numbers are
diagnostics, not tuned settings, so touching TEST is per protocol (same fields
the canonical eval already scores).

Sensor layouts are the canonical draw: torch.manual_seed(seed*777+snap) then
helpers.build_sparse_condition on a compute node, with the snap-29 fingerprint
gate exercised explicitly.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = ("/home/ntricard/generative_reconstruction/temp/"
       "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from confild_eval_unified import load_stage1  # noqa: E402  (inserts CoNFiLD root)
from helpers import TurbulentCombustionH5Dataset, build_sparse_condition  # noqa: E402
from ensemble_eval import check_canonical_fingerprint, require_compute_node  # noqa: E402

ROOT = ("/home/ntricard/generative_reconstruction/temp/"
        "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion")
ARMS = {
    "P": f"{ROOT}/Save_TrainedModel/JHU/baseline_confild/unified_published_prior/"
         "Baseline_confild_Stage1_DemoN23_20260828_182524/best.pt",
    "C": f"{ROOT}/Save_TrainedModel/JHU/baseline_confild/unified_cap1024/"
         "Baseline_confild_Stage1_DemoN23_20260828_182524/best.pt",
    "F": f"{ROOT}/Save_TrainedModel/JHU/baseline_confild/unified_faithful384/"
         "Baseline_confild_Stage1_DemoN23_20260828_182531/best.pt",
}


def decode_full(decoder, z, coords_full, chunk=262144):
    pieces = []
    with torch.no_grad():
        for s in range(0, coords_full.shape[0], chunk):
            c = coords_full[s:s + chunk].unsqueeze(0)
            pieces.append(decoder(c, z.view(1, 1, -1))[0])
    return torch.cat(pieces, 0)  # [N, F] in pm1 units


def per_channel_rel_l2(pred_pm1, truth_z, stats):
    """z-scored per-channel rel L2, the units of the canonical table."""
    pred_z = stats.benchmark_normalize(stats.denormalize(pred_pm1))
    num = torch.linalg.vector_norm(pred_z - truth_z, dim=0)
    den = torch.linalg.vector_norm(truth_z, dim=0).clamp_min(1e-12)
    return (num / den).tolist()


def fit_latent_adam(decoder, latent_dim, device, loss_fn, steps, lr, seed,
                    record_at, record_fn):
    """Optimize a single latent with the decoder frozen; record at milestones."""
    z = torch.nn.Parameter(torch.zeros(1, 1, latent_dim, device=device))
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 0.03)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    records = {}
    for step in range(1, steps + 1):
        loss = loss_fn(z, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step in record_at:
            records[step] = record_fn(z.detach())
            records[step]["fit_loss"] = float(loss.detach())
    return z.detach(), records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["P", "C", "F"])
    p.add_argument("--snaps", type=int, nargs="+", default=[0, 12, 24, 36, 48])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--record-at", type=int, nargs="+",
                   default=[400, 1000, 2000, 3000])
    p.add_argument("--lr", type=float, default=1.0e-2)
    p.add_argument("--points-per-step", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0, 2])
    p.add_argument("--n-obs", type=int, nargs="+", default=[19531, 19531])
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default="/projects/ammoniacomb/generative_reconstruction/"
                   "jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5")
    args = p.parse_args()

    require_compute_node()
    device = torch.device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds_kwargs = dict(train_ratio=0.75, seed=42, field_names=("Ux", "Uy", "Uz", "p"),
                     time_stride=1, stats_path=str(out.parent / "dataset_stats.pt"))
    TurbulentCombustionH5Dataset(args.data, split="train", **ds_kwargs)
    dataset = TurbulentCombustionH5Dataset(args.data, split="val", **ds_kwargs)
    field_names = list(dataset.field_names)

    # Canonical sensor draws (+ snap-29 gate exercised even if 29 is not scored).
    sensors = {}
    for snap in sorted(set(args.snaps) | {29}):
        item = dataset[int(snap)]
        coords = item["coords"].unsqueeze(0).to(device)
        fields = item["fields"].unsqueeze(0).to(device)
        torch.manual_seed(args.seed * 777 + int(snap))
        _oc, _ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=args.cond_fields, n_obs_min=args.n_obs, n_obs_max=args.n_obs,
        )
        valid = om[0] > 0
        print(f"[seedcheck] snap={int(snap)} sensors={int(om[0].sum())} "
              f"idx_sum={int(oi[0][valid].sum())}", flush=True)
        check_canonical_fingerprint(int(snap), int(om[0].sum()),
                                    int(oi[0][valid].sum()), args.seed,
                                    args.cond_fields, args.n_obs)
        sensors[int(snap)] = (oi[0, valid].long(), ofid[0, valid].long())

    coords_full = dataset[0]["coords"].to(device) * 2.0 - 1.0
    results = {}
    for arm in args.arms:
        decoder, stats, s1 = load_stage1(ARMS[arm], device)
        latent_dim = int(s1["architecture"]["latent_dim"])
        results[arm] = {"stage1_ckpt": ARMS[arm], "epoch": int(s1["epoch"]),
                        "latent_dim": latent_dim, "snaps": {}}
        for snap in args.snaps:
            truth_z = dataset[int(snap)]["fields"].to(device)          # [N, F]
            raw = truth_z * stats.benchmark_std + stats.benchmark_mean
            truth_pm1 = stats.normalize(raw)                            # [N, F]

            def record(z):
                pred = decode_full(decoder, z, coords_full)
                rel = per_channel_rel_l2(pred, truth_z, stats)
                return {"rel_l2": dict(zip(field_names, rel)),
                        "rel_l2_agg": float(np.mean(rel))}

            # ---- (i) full-field fit: the codec ceiling ------------------
            n_points = truth_pm1.shape[0]

            def loss_full(z, gen):
                ids = torch.randint(n_points, (args.points_per_step,),
                                    generator=gen).to(device)
                pred = decoder(coords_full[ids].unsqueeze(0), z)
                return F.mse_loss(pred, truth_pm1[ids].unsqueeze(0))

            t0 = time.time()
            _, rec_full = fit_latent_adam(
                decoder, latent_dim, device, loss_full, args.steps, args.lr,
                seed=9 + int(snap), record_at=set(args.record_at) | {args.steps}, record_fn=record)

            # ---- (ii) sensor-only fit: the information limit ------------
            s_idx, s_fld = sensors[int(snap)]
            s_coords = coords_full[s_idx]                               # [M, 3]
            y = truth_pm1[s_idx].gather(1, s_fld.unsqueeze(-1)).squeeze(-1)  # [M]

            def loss_sensor(z, gen):
                pred = decoder(s_coords.unsqueeze(0), z)[0]             # [M, F]
                pred_at = pred.gather(1, s_fld.unsqueeze(-1)).squeeze(-1)
                return F.mse_loss(pred_at, y)

            def record_sensor(z):
                entry = record(z)
                with torch.no_grad():
                    pred = decoder(s_coords.unsqueeze(0), z.view(1, 1, -1))[0]
                    pred_at = pred.gather(1, s_fld.unsqueeze(-1)).squeeze(-1)
                    entry["sensor_rel_l2_pm1"] = float(
                        torch.linalg.vector_norm(pred_at - y)
                        / torch.linalg.vector_norm(y).clamp_min(1e-12))
                return entry

            _, rec_sensor = fit_latent_adam(
                decoder, latent_dim, device, loss_sensor, args.steps, args.lr,
                seed=17 + int(snap), record_at=set(args.record_at) | {args.steps},
                record_fn=record_sensor)

            results[arm]["snaps"][int(snap)] = {
                "codec_full_fit": {str(k): v for k, v in rec_full.items()},
                "sensor_only_fit": {str(k): v for k, v in rec_sensor.items()},
                "seconds": time.time() - t0,
            }
            last = str(args.steps)
            cf = results[arm]["snaps"][int(snap)]["codec_full_fit"][last]
            so = results[arm]["snaps"][int(snap)]["sensor_only_fit"][last]
            print(f"[stageA] arm={arm} snap={snap} "
                  f"codec_agg={cf['rel_l2_agg']:.4f} "
                  f"sensor_agg={so['rel_l2_agg']:.4f} "
                  f"codec={ {k: round(v,3) for k,v in cf['rel_l2'].items()} } "
                  f"sensor={ {k: round(v,3) for k,v in so['rel_l2'].items()} } "
                  f"({results[arm]['snaps'][int(snap)]['seconds']:.0f}s)", flush=True)
            json.dump(results, open(out, "w"), indent=1)

        # Arm-level means at the final record step.
        last = str(args.steps)
        for key in ("codec_full_fit", "sensor_only_fit"):
            per = {f: float(np.mean([results[arm]["snaps"][s][key][last]["rel_l2"][f]
                                     for s in results[arm]["snaps"]]))
                   for f in field_names}
            agg = float(np.mean(list(per.values())))
            results[arm][f"{key}_mean"] = {"rel_l2": per, "rel_l2_agg": agg}
            print(f"[stageA] ARM {arm} MEAN {key}: agg={agg:.4f} "
                  f"{ {k: round(v,3) for k,v in per.items()} }", flush=True)
        json.dump(results, open(out, "w"), indent=1)

    print("[stageA] done", flush=True)


if __name__ == "__main__":
    main()
