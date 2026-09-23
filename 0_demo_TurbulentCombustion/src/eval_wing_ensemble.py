#!/usr/bin/env python3
"""Matched-protocol evaluation for the SHIFT-WING fleet (ours + baselines).

One driver for our DMF-Gen point-cloud FFM run and the three point-native
learned baselines (Senseiver, MLP-RBF, SiT point-token), so every row in the
wing table is produced by the same code path.

PROTOCOL -- what is held identical across all four rows
-------------------------------------------------------
* SENSORS.  ``evaluate_wing.build_surface_obs`` is imported and called
  VERBATIM: 512 taps from pool channel 0 plus 128 sensors from each of
  channels 1-3, then the two exact Mach/alpha parameter tokens appended last;
  896 surface sensors + 2 tokens = 898 observations. Its draw is
  ``torch.Generator().manual_seed(seed)`` on the CPU, so unlike the JHU
  protocol the layout is NOT GPU-SKU dependent and is bit-identical on any
  node.
* CASES AND SEEDS.  The first ``--n-cases`` (default 8) validation cases in
  split order, with per-case sensor seed ``seed * 100 + i`` at ``--seed 0``.
  These are exactly the case indices and seeds ``evaluate_wing.main`` used for
  the reference DMF-Gen numbers, so the baselines answer the same 8 inverse
  problems from the same 8 sensor layouts.
* METRICS.  ``ensemble_eval.ensemble_metrics``, unmodified, on the normalized
  fields; reported per field (Ux, Uy, Uz, Cp) and as the aggregate mean over
  fields, then averaged over cases.
* ENSEMBLE NOISE.  Per-case base ``seed * 31 + i`` (the constant
  ``evaluate_wing.main`` passes to ``sample_ensemble``), then
  ``torch.manual_seed(base * 10_000 + k)`` before sample k -- the same
  convention ``ensemble_eval.sample_ensemble`` applies internally, so the
  generative baselines and our model draw their K members the same way.

DETERMINISTIC ROWS (senseiver, mlp_rbf)
---------------------------------------
One forward per case, chunked over query points. Scored the way the JHU and
2D canonical evals score point predictors: the prediction is tiled into TWO
IDENTICAL ensemble members, so the fair CRPS estimator's pair term vanishes
exactly and CRPS == MAE while spread == 0. The dispersion fields (spread,
spread_error_ratio, coverage_50/90, rank_hist) are then set to null, because
they carry no information for a point predictor, and K is reported as 1.

SiT (point-token)
-----------------
``model_baseline.sit_conditional_sample_points_chunked`` with
``surface_pool=True``, i.e. the same nearest-sensor fill over 9 field-id
channels that training used, with Mach/alpha broadcast globally. The field is
reconstructed in independent ``node_subsample``-sized chunks of a random
permutation: pointwise metrics are well defined, single-sample spatial
coherence is limited to the chunk, and that is a documented property of this
baseline rather than a protocol choice.

COST
----
Per-case sampling wall-clock (CUDA-synchronized) and peak GPU memory are
recorded and summarized in every JSON. For DMF-Gen the K members come out of
one ``sample_ensemble`` call, so the reported per-field second count is
total / K; for SiT it is the k == 0 draw; for the deterministic rows it is the
single forward.

Touches no shared module.

    python eval_wing_ensemble.py --model senseiver --run-dir <run> --ckpt best
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

DISPERSION_KEYS = ("spread", "spread_error_ratio", "coverage_50", "coverage_90")
DET_MODELS = {"senseiver", "mlp_rbf"}
PROTOCOL = "shiftwing_surface_to_volume_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Matched SHIFT-WING fleet evaluation")
    p.add_argument("--model", required=True,
                   choices=["dmfgen", "senseiver", "mlp_rbf", "mlprbf", "sit"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--K", type=int, default=4,
                   help="Posterior members for generative rows. The reference "
                        "DMF-Gen wing numbers are K=4.")
    p.add_argument("--nfe", type=int, default=None,
                   help="ODE steps. Default: 4 for dmfgen (the reference "
                        "setting); the run config's sampling_N for sit; 1 for "
                        "the deterministic rows.")
    p.add_argument("--n-cases", type=int, default=8)
    p.add_argument("--n-taps", type=int, default=512)
    p.add_argument("--n-shear", type=int, default=128)
    p.add_argument("--noise-sigma", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=131_072,
                   help="Query points per forward (deterministic rows) / per "
                        "ODE step (dmfgen).")
    p.add_argument("--expect-val-len", type=int, default=73,
                   help="Abort if the val split length differs (protocol guard).")
    p.add_argument("--out-prefix", default="wing_fleet")
    p.add_argument("--out", default=None,
                   help="Explicit JSON path; default is "
                        "<run-dir>/Evaluation/<prefix>_<model>_*.json")
    p.add_argument("--dry-run", action="store_true",
                   help="CPU-safe: resolve config + dataset + case list, print "
                        "the plan, build no model, touch no GPU.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric bookkeeping
# ---------------------------------------------------------------------------

def num_points(dataset) -> int:
    """ShiftWingBaselineDataset exposes num_points; the raw ShiftWingDataset
    our model uses does not, so fall back to the volume node count."""
    n = getattr(dataset, "num_points", None)
    if n is not None:
        return int(n)
    from dataset_shiftwing import N_VOL
    return int(N_VOL)


def case_name(dataset, i: int) -> str:
    files = getattr(dataset, "files", None)
    if files is None:
        files = dataset.inner.files
    return files[i].name


def null_dispersion(m: dict) -> dict:
    """Deterministic row: dispersion metrics carry no information -- null them."""
    for d in [m["aggregate"], *m["per_field"].values()]:
        for k in DISPERSION_KEYS:
            d[k] = None
    m["rank_hist"] = None
    m["deterministic"] = True
    return m


def mean_std(per_case: list[dict], path: list[str]):
    vals = []
    for m in per_case:
        d = m
        for k in path:
            d = d[k]
        if d is None:            # nulled dispersion field
            return None, None
        vals.append(float(d))
    return float(np.mean(vals)), float(np.std(vals))


def summarize(per_case: list[dict], field_names: list[str]) -> dict:
    agg_keys = list(per_case[0]["aggregate"].keys())
    fld_keys = list(per_case[0]["per_field"][field_names[0]].keys())
    return {
        "aggregate": {k: mean_std(per_case, ["aggregate", k])[0] for k in agg_keys},
        "aggregate_std": {k: mean_std(per_case, ["aggregate", k])[1] for k in agg_keys},
        "per_field": {f: {k: mean_std(per_case, ["per_field", f, k])[0]
                          for k in fld_keys} for f in field_names},
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dmfgen(run_dir: Path, ckpt: str, device: str):
    """Mirror of evaluate_wing.main's loader, so the reference row reproduces."""
    from dataset_shiftwing import ShiftWingDataset
    from evaluate_ffm import _build_model, _normalize_eval_config

    cfg = _normalize_eval_config(json.load(open(run_dir / "args.json")))
    dataset = ShiftWingDataset(cfg["processed_root"], split="val")
    cfg.setdefault("n_obs_field_types", dataset.n_obs_field_types)
    model = _build_model(cfg, dataset)
    ckpt_name = ckpt if ckpt.endswith(".pt") else f"{ckpt}.pt"
    state = torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if state.get("ema") is not None:
        sd = model.state_dict()
        for k, v in state["ema"]["shadow"].items():
            sd[k].copy_(v.to(sd[k].dtype))
    model.to(device).eval()
    print(f"[eval] dmfgen {run_dir.name}/{ckpt_name} epoch={state.get('epoch')}",
          flush=True)
    return model, dataset, cfg


def load_baseline(run_dir: Path, ckpt: str, model_name: str, split: str):
    import model_baseline as MB

    cfg = MB.validate_and_normalize_config(MB.load_yaml(run_dir / "run_config.yaml"))
    cfg["baseline_model"] = model_name
    cfg["training_stage"] = 1
    if not MB.surface_pool_enabled(cfg):
        raise SystemExit(
            "[guard] this run's config does not set shared.conditioning."
            "source: surface_pool -- it was trained on VOLUME sensors and is "
            "not comparable on the surface-to-volume protocol.")
    checkpoint = MB.safe_torch_load(run_dir / f"{ckpt}.pt", map_location="cpu")
    device = MB.infer_device(None, cfg["shared"]["device_ids"])
    dataset = MB.build_dataset(cfg, split=split, stats_path=run_dir / "dataset_stats.pt")
    adapter = MB.get_baseline_adapter(model_name)
    bundle = adapter.build_for_training(cfg=cfg, device=device, run_dir=run_dir,
                                        train_set=dataset, val_set=dataset)
    adapter.load_checkpoint(bundle, checkpoint)
    print(f"[eval] {model_name} {run_dir.name}/{ckpt}.pt "
          f"(epoch {checkpoint.get('epoch')})", flush=True)
    return MB, adapter, bundle, dataset, cfg, device


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.model == "mlprbf":
        args.model = "mlp_rbf"
    deterministic = args.model in DET_MODELS
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "Evaluation"

    # ---------------- dry run -------------------------------------------
    if args.dry_run:
        if args.model == "dmfgen":
            from dataset_shiftwing import ShiftWingDataset
            from evaluate_ffm import _normalize_eval_config
            cfg = _normalize_eval_config(json.load(open(run_dir / "args.json")))
            dataset = ShiftWingDataset(cfg["processed_root"], split=args.split)
            data_path = cfg["processed_root"]
            ckpt_file = run_dir / (args.ckpt if args.ckpt.endswith(".pt")
                                   else f"{args.ckpt}.pt")
            nfe = args.nfe if args.nfe is not None else 4
        else:
            import model_baseline as MB
            cfg = MB.validate_and_normalize_config(
                MB.load_yaml(run_dir / "run_config.yaml"))
            cfg["training_stage"] = 1
            dataset = MB.build_dataset(cfg, split=args.split,
                                       stats_path=run_dir / "dataset_stats.pt")
            data_path = cfg["shared"]["data"]["processed_root"]
            ckpt_file = run_dir / f"{args.ckpt}.pt"
            nfe = (1 if deterministic else
                   (args.nfe if args.nfe is not None
                    else int(MB.resolve_stage_config(cfg)["sampling"]["sampling_N"])))
        print(f"[dry-run] model={args.model} run={run_dir}")
        print(f"[dry-run] ckpt={ckpt_file} exists={ckpt_file.is_file()}")
        print(f"[dry-run] data={data_path} split={args.split} len={len(dataset)} "
              f"(expected {args.expect_val_len})")
        print(f"[dry-run] fields={list(dataset.field_names)} "
              f"points={num_points(dataset)}")
        print(f"[dry-run] cases={list(range(min(args.n_cases, len(dataset))))} "
              f"sensor_seeds={[args.seed * 100 + i for i in range(min(args.n_cases, len(dataset)))]}")
        print(f"[dry-run] K={1 if deterministic else args.K} nfe={nfe} "
              f"n_taps={args.n_taps} n_shear={args.n_shear} "
              f"observations={args.n_taps + 3 * args.n_shear + 2}")
        if len(dataset) != args.expect_val_len:
            raise SystemExit(f"[dry-run] VAL LENGTH MISMATCH: {len(dataset)} "
                             f"!= {args.expect_val_len}")
        print("[dry-run] OK")
        return

    # ---------------- real run ------------------------------------------
    # The canonical sensor draw is CPU-seeded, so layouts reproduce anywhere;
    # only the COST fields need real hardware. Guard so a login-node run cannot
    # quietly publish timings. (--dry-run returns above and is unaffected.)
    if not os.environ.get("SLURM_JOB_ID") and os.environ.get("ALLOW_LOGIN_EVAL") != "1":
        raise SystemExit(
            "[nodecheck] SLURM_JOB_ID unset: refusing to publish cost fields "
            "measured off a compute node. Submit via sbatch, or set "
            "ALLOW_LOGIN_EVAL=1 for debugging (sensor layouts are unaffected: "
            "build_surface_obs seeds a CPU generator).")

    from ensemble_eval import ensemble_metrics
    from evaluate_wing import build_surface_obs   # VERBATIM canonical draw

    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    MB = adapter = bundle = None
    if args.model == "dmfgen":
        device = "cuda:0"
        model, dataset, cfg = load_dmfgen(run_dir, args.ckpt, device)
        data_path = cfg["processed_root"]
        nfe = args.nfe if args.nfe is not None else 4
        solver = "euler"
    else:
        MB, adapter, bundle, dataset, cfg, device = load_baseline(
            run_dir, args.ckpt, args.model, args.split)
        data_path = cfg["shared"]["data"]["processed_root"]
        if deterministic:
            nfe, solver = 1, "none"
        else:
            sampling = MB.resolve_stage_config(cfg)["sampling"]
            nfe = int(args.nfe if args.nfe is not None else sampling["sampling_N"])
            solver = str(sampling["ode_solver"])

    n_total = len(dataset)
    if n_total != args.expect_val_len:
        raise SystemExit(f"[guard] {args.split} split has {n_total} cases, "
                         f"expected {args.expect_val_len}.")
    field_names = list(dataset.field_names)
    n_cases = min(args.n_cases, n_total)
    n_fields = int(dataset.num_fields)
    K_eff = 1 if deterministic else args.K

    print(f"[protocol] model={args.model} split={args.split} n={n_total} "
          f"cases={list(range(n_cases))} data={data_path}", flush=True)
    print(f"[protocol] K={K_eff} nfe={nfe} solver={solver} seed={args.seed} "
          f"n_taps={args.n_taps} n_shear={args.n_shear} gpu={gpu_name}", flush=True)

    seeding_note = (
        "sensors: evaluate_wing.build_surface_obs called verbatim with "
        f"n_taps={args.n_taps}, n_shear={args.n_shear}, seed=seed*100+case "
        "(a CPU torch.Generator, so the layout is GPU-SKU independent). "
        "ensemble noise: base=seed*31+case, torch.manual_seed(base*10000+k) "
        "before member k, matching ensemble_eval.sample_ensemble's internal "
        "convention. Cases are the first n_cases validation cases in split "
        "order -- the same ones evaluate_wing.main scored for our model.")

    per_case: list[dict] = []
    timings: list[float] = []
    mems: list[float] = []

    def score_one(i: int) -> dict:
        item = dataset[i]
        coords = item["coords"][None].to(device)
        true = item["fields"].numpy()
        obs = build_surface_obs(item, args.n_taps, args.n_shear, device,
                                seed=args.seed * 100 + i,
                                noise_sigma=args.noise_sigma)
        n_obs_tokens = int(obs["coords"].shape[1])
        expected = args.n_taps + 3 * args.n_shear + item["param_coords"].shape[0]
        if n_obs_tokens != expected:
            raise SystemExit(f"[guard] case {i}: {n_obs_tokens} observations, "
                             f"expected {expected}")
        ids = obs["field_ids"][0].tolist()
        print(f"[seedcheck] case={i} seed={args.seed * 100 + i} "
              f"obs={n_obs_tokens} param_tokens={ids[-2:]} "
              f"coord_sum={float(obs['coords'].sum()):.4f}", flush=True)

        base = args.seed * 31 + i
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        if args.model == "dmfgen":
            from ensemble_eval import sample_ensemble
            t0 = time.perf_counter()
            ens = sample_ensemble(model, coords, obs, K=args.K, n_steps=nfe,
                                  chunk=args.chunk, clamp_hard=False, seed=base)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) / args.K)
            ens = ens.numpy()
        elif deterministic:
            # One forward, chunked over query points (400k nodes will not fit
            # a single decoder pass), then tiled into two identical members.
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                n_q = coords.shape[1]
                pred = torch.empty(1, n_q, n_fields, device=coords.device,
                                   dtype=coords.dtype)
                for s in range(0, n_q, args.chunk):
                    e = min(s + args.chunk, n_q)
                    pred[:, s:e] = bundle.model(
                        coords[:, s:e], obs["coords"], obs["values"],
                        obs["mask"], obs["field_ids"])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - t0)
            ens = np.repeat(pred[0].detach().float().cpu().numpy()[None], 2, axis=0)
        else:  # sit, point-token
            comp = bundle.components
            members = []
            with torch.no_grad():
                for k in range(K_eff):
                    torch.manual_seed(base * 10_000 + k)
                    if k == 0 and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if k == 0:
                        t0 = time.perf_counter()
                    r = MB.sit_conditional_sample_points_chunked(
                        net=bundle.model, transport=comp["transport"],
                        coords=coords, obs_coords=obs["coords"],
                        obs_values=obs["values"], obs_mask=obs["mask"],
                        obs_field_ids=obs["field_ids"], n_fields=n_fields,
                        device=device, n_steps=nfe, sampler_type=solver,
                        chunk=int(comp["node_subsample"]),
                        sigma=float(comp.get("cond_fill_sigma", 0.05)),
                        cond_value_channels=int(comp["cond_value_channels"]),
                        surface_pool=True)
                    if k == 0:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        timings.append(time.perf_counter() - t0)
                    members.append(r[0].detach().float().cpu().numpy())
            ens = np.stack(members, axis=0)

        mems.append(torch.cuda.max_memory_allocated() / 1024 ** 3
                    if torch.cuda.is_available() else 0.0)
        m = ensemble_metrics(ens, true, field_names)
        if deterministic:
            m = null_dispersion(m)
        m["case_index"] = i
        m["case"] = case_name(dataset, i)
        m["sensor_seed"] = args.seed * 100 + i
        m["observations"] = n_obs_tokens
        m["K"] = K_eff
        m["nfe"] = nfe
        m["sample_seconds"] = timings[-1]
        m["peak_gpu_gb"] = mems[-1]
        agg = m["aggregate"]
        print(f"[case {i}] {m['case']} " + " ".join(
            f"{k}={v:.5f}" for k, v in agg.items() if v is not None), flush=True)
        return m

    if args.model == "dmfgen":
        for i in range(n_cases):
            per_case.append(score_one(i))
    else:
        with adapter.evaluation_weights(bundle):
            bundle.model.eval()
            for i in range(n_cases):
                per_case.append(score_one(i))

    payload = {
        "protocol": PROTOCOL,
        "model": args.model,
        "run_dir": str(run_dir),
        "ckpt": args.ckpt,
        "split": args.split,
        "val_len": n_total,
        "n_cases": n_cases,
        "case_indices": list(range(n_cases)),
        "sensor_seeds": [args.seed * 100 + i for i in range(n_cases)],
        "K": K_eff,
        "deterministic": deterministic,
        **({"note_K": "deterministic model; metrics computed with two "
                      "identical members so the fair CRPS estimator is well "
                      "defined: CRPS == MAE exactly, rel_l2_single == "
                      "rel_l2_mean; dispersion fields (spread, "
                      "spread_error_ratio, coverage, rank_hist) are null"}
           if deterministic else {}),
        "nfe": nfe,
        "ode_solver": solver,
        "seed": args.seed,
        "n_taps": args.n_taps,
        "n_shear": args.n_shear,
        "observations_per_case": args.n_taps + 3 * args.n_shear + 2,
        "noise_sigma": args.noise_sigma,
        "data_path": str(data_path),
        "num_points": num_points(dataset),
        "field_names": field_names,
        "gpu": gpu_name,
        "seeding": seeding_note,
        "conditioning_note": (
            "all rows condition ONLY on the body-surface observation pool "
            "(Cp + 3 wall-shear channels) plus 2 exact flow-parameter tokens; "
            "no interior sample is ever provided to any model"),
        "inference_seconds_per_field_mean": float(np.mean(timings)),
        "inference_seconds_per_field_std": float(np.std(timings)),
        "inference_peak_gpu_gb": float(np.max(mems)) if mems else None,
        "timing_note": ("dmfgen: total sample_ensemble wall-clock / K; sit: "
                        "the k==0 draw; deterministic rows: the single "
                        "chunked forward. CUDA-synchronized."),
        "summary": summarize(per_case, field_names),
        "cases": per_case,
    }

    tag = {"mlp_rbf": "mlprbf"}.get(args.model, args.model)
    name = (f"{args.out_prefix}_{tag}_K1.json" if deterministic
            else f"{args.out_prefix}_{tag}_K{K_eff}_nfe{nfe}.json")
    out_path = Path(args.out) if args.out else out_dir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))

    s = payload["summary"]["aggregate"]
    pf = payload["summary"]["per_field"]
    disp = ("deterministic (dispersion null)" if deterministic else
            f"spread/err={s['spread_error_ratio']:.3f} "
            f"cov90={s['coverage_90']:.3f}")
    print(f"\n[RESULT] {args.model} relL2={s['rel_l2_mean']:.5f} "
          f"crps={s['crps']:.5f} {disp}", flush=True)
    print("[RESULT] per-field relL2 " + " ".join(
        f"{f}={pf[f]['rel_l2_mean']:.4f}" for f in field_names), flush=True)
    print(f"[RESULT] cost infer={payload['inference_seconds_per_field_mean']:.3f}s "
          f"peak={payload['inference_peak_gpu_gb']:.2f}GB", flush=True)
    print(f"[out] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
