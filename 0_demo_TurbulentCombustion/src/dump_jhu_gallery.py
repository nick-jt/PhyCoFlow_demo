#!/usr/bin/env python3
"""Field dumps for the JHU all-baselines gallery, on the EXACT recorded sensors.

The existing JHU gallery reads DMF-Gen and latent-FM fields from
Paper/iclr2027/figures/{qual_jhu,spectra_fields}.npz, which were never
transferred to this host, so it cannot be regenerated here. This script
rebuilds those panels -- and adds FNO3D and the two classical floors -- on the
SAME held-out snapshot and the SAME sensors as the surviving jhu_sit.npz /
jhu_senseiver.npz dumps (qual protocol: absolute frame 153, sensor seed 103,
U_x and U_z observed at 19,531 points each, U_y and p unobserved).

The sensors are INJECTED from jhu_sit.npz rather than redrawn. CUDA randperm is
not portable across GPU SKUs at every size (the 23,800-point cylinder mesh
draws differently on GH200 than on the H100 the fleet used), so a fresh draw is
only a diagnostic here: it is printed, never trusted. Injection makes every
column's conditioning identical by construction.

Refuses to write a panel whose truth does not match the reference dump's --
a wrong frame or split would otherwise produce a plausible-looking field.

Outputs (Save_TrainedModel_pof/field_dumps/): jhu_dmfgen.npz, jhu_fno3d.npz,
jhu_classical.npz -- all in PHYSICAL units, same schema as jhu_sit.npz.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

WT = Path(__file__).resolve().parents[1]
FD = WT / "Save_TrainedModel_pof" / "field_dumps"
REF = FD / "jhu_sit.npz"
ABS_FRAME, SENSOR_SEED, COND, N_OBS = 153, 103, [0, 2], 19531
SJHU = WT / "Save_TrainedModel" / "JHU"
RUNS = {
    "dmfgen": SJHU / "pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100",
    "fno3d": SJHU / "baseline_fno/fno3d_matched_DemoN90_20260828_181849",
}


def stats(ds):
    m, s = getattr(ds, "mean"), getattr(ds, "std")
    m = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
    s = s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
    return m.astype(np.float64).ravel(), s.astype(np.float64).ravel()


def framework_dump(method, ref, device):
    from ensemble_eval import load_run, sample_ensemble
    from helpers import build_sparse_condition
    model, dataset, _ = load_run(str(RUNS[method]), "best.pt", device)
    idx_abs = np.asarray(dataset.indices)
    hit = np.where(idx_abs == ABS_FRAME)[0]
    if hit.size != 1:
        raise SystemExit(f"[{method}] absolute frame {ABS_FRAME} not in this "
                         f"run's split (indices {idx_abs[:6]}...)")
    snap = int(hit[0])
    item = dataset[snap]
    coords = item["coords"][None].to(device)
    fields = item["fields"][None].to(device)
    mean, std = stats(dataset)
    truth_p = item["fields"].numpy().astype(np.float64) * std + mean
    ok = np.allclose(truth_p, ref["truth"].astype(np.float64), rtol=1e-3, atol=1e-3)
    print(f"[{method}] val snap {snap} = abs {ABS_FRAME}; truth matches "
          f"reference: {ok}", flush=True)
    if not ok:
        raise SystemExit(f"[{method}] truth mismatch -- refusing to dump")

    # diagnostic only: does this GPU reproduce the recorded draw?
    torch.manual_seed(SENSOR_SEED)
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords, fields_full=fields, cond_fields=COND,
        n_obs_min=[N_OBS], n_obs_max=[N_OBS])
    ridx = np.asarray(ref["sensor_indices"]).astype(np.int64)
    rfid = np.asarray(ref["sensor_field_ids"]).astype(np.int64)
    fresh = int(oi[om.bool()].sum())
    print(f"[{method}] fresh draw idx_sum={fresh} vs recorded {int(ridx.sum())} "
          f"-> {'reproduces' if fresh == int(ridx.sum()) else 'DIFFERS (injecting recorded)'}",
          flush=True)
    if oi.shape[1] != ridx.size:
        raise SystemExit(f"[{method}] sensor count {oi.shape[1]} != recorded {ridx.size}")
    # inject the recorded sensors into the draw's own tensors (keeps dtypes,
    # shapes and padding convention exactly as the model expects)
    it = torch.from_numpy(ridx).to(device)
    ft = torch.from_numpy(rfid).to(device)
    oi[0] = it.to(oi.dtype)
    ofid[0] = ft.to(ofid.dtype)
    om[0] = 1
    oc[0] = coords[0, it]
    ov[0, :, 0] = fields[0, it, ft]
    obs = {"coords": oc, "values": ov, "mask": om, "indices": oi, "field_ids": ofid}
    t0 = time.perf_counter()
    # FNO3D is a spectral operator on the whole cube and refuses query chunks,
    # so it gets the full grid in one pass; point models keep the qual chunking.
    chunk = coords.shape[1] if method == "fno3d" else 262144
    ens = sample_ensemble(model, coords, obs, K=1, n_steps=4, chunk=chunk,
                          clamp_hard=False, seed=5 + 3).numpy()   # qual convention
    print(f"[{method}] sampled in {time.perf_counter()-t0:.1f}s", flush=True)
    pred_p = ens[0].astype(np.float64) * std + mean
    return pred_p, truth_p, mean, std, {"run_dir": str(RUNS[method]), "val_snapshot": snap,
                                         "fresh_draw_idx_sum": fresh, "nfe": 4, "K": 1,
                                         "sample_seed": 8}


def classical_dump(ref):
    sys.path.insert(0, str(WT / "src" / "figs_pof"))
    from make_recon_galleries import sensors_by_field, load_frames_std, JHU_H5
    from baseline_classical_jhu import GappyPOD, kd_predict, obs_columns
    G = 125
    mean = ref["norm_mean"].astype(np.float64); std = ref["norm_std"].astype(np.float64)
    truth_p = ref["truth"].astype(np.float64); truth_z = (truth_p - mean) / std
    sens = sensors_by_field(ref["sensor_indices"], ref["sensor_field_ids"], truth_z)
    craw = ref["coords_raw"].astype(np.float64)
    lo = craw.min(0); dx = (craw.max(0) - lo) / (G - 1)
    box = np.ascontiguousarray((craw - lo) / (G * dx))
    t0 = time.time(); idw_z = kd_predict(box, sens, 4, "idw", 8, None)
    print(f"[classical] IDW k=8 {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    X = load_frames_std(JHU_H5, range(150), mean.astype(np.float32),
                        std.astype(np.float32), G ** 3, 4).reshape(150, -1)
    mu = X.mean(axis=0); X -= mu
    pod = GappyPOD(X, mu, rank=80); del X
    cols, vals = obs_columns(sens, 4)
    pod_z = pod.reconstruct(cols, vals).reshape(G ** 3, 4)
    print(f"[classical] gappy POD r={pod.r} {time.time()-t0:.0f}s", flush=True)
    return {"pred_idw": idw_z * std + mean, "pred_gappy_pod_r80": pod_z * std + mean}


def save(name, ref, meta, **arrays):
    out = FD / name
    if out.exists():
        print(f"[save] {name} exists, not overwriting"); return
    np.savez_compressed(out, coords=ref["coords"], coords_raw=ref["coords_raw"],
                        sensor_indices=ref["sensor_indices"],
                        sensor_field_ids=ref["sensor_field_ids"], names=ref["names"],
                        norm_mean=ref["norm_mean"], norm_std=ref["norm_std"],
                        meta=np.array(json.dumps(meta, indent=1)),
                        **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
    print(f"[save] wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["dmfgen", "fno3d", "classical"])
    a = ap.parse_args()
    ref = dict(np.load(REF, allow_pickle=False))
    base = {"protocol": "jhu_qual_gallery", "absolute_frame": ABS_FRAME,
            "sensor_source": "INJECTED from jhu_sit.npz (qual draw, seed 103)",
            "sensor_idx_sum": int(np.asarray(ref["sensor_indices"]).sum()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    fail = 0
    for m in a.methods:
        try:
            if m == "classical":
                save("jhu_classical.npz", ref, {**base, "method": "classical"},
                     truth=ref["truth"], **classical_dump(ref))
            else:
                pred, truth, mean, std, extra = framework_dump(m, ref, "cuda:0")
                save(f"jhu_{m}.npz", ref, {**base, "method": m, **extra},
                     truth=truth, pred_sample=pred, pred_mean=pred)
        except SystemExit as e:
            print(f"[{m}] ABORTED: {e}", flush=True); fail = 1
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[{m}] FAILED: {e}"); fail = 1
    sys.exit(fail)


if __name__ == "__main__":
    main()
