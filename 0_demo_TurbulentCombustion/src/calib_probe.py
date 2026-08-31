"""H2 probe + fingerprint + structure of the ensemble spread.

Measures (a) the canonical sensor fingerprint, (b) the pointwise std of the
RFF-GP source prior at the evaluation query points, (c) the radial spectrum
and correlation length of the ensemble deviations vs the prior deviations,
which discriminates "uncontracted prior" (H2) from a learned conditional
width with data-like structure (H3).
"""
import json, time, sys
import numpy as np
import torch

from ensemble_eval import load_run, sample_ensemble
from helpers import build_sparse_condition

RD = ("/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/"
      "0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/"
      "iclr_jhu_xcube_spec02_DemoN29_20260822_140100")
DEV = "cuda:0"
SEED = 0
out = {}

model, dataset, cfg = load_run(RD, "best.pt", DEV)
print(f"[probe] dataset len={len(dataset)} num_points={dataset.num_points} "
      f"fields={dataset.field_names}", flush=True)

rng = np.random.default_rng(SEED)
snap_ids = rng.choice(len(dataset), size=50, replace=False)
print(f"[probe] snap_ids[:8]={snap_ids[:8].tolist()}  contains29={29 in snap_ids.tolist()}", flush=True)

# ---------------------------------------------------------------- fingerprint
item = dataset[29]
coords = item["coords"].unsqueeze(0).to(DEV)
fields = item["fields"].unsqueeze(0).to(DEV)
torch.manual_seed(SEED * 777 + 29)
oc, ov, om, oi, ofid = build_sparse_condition(
    coords_full=coords, fields_full=fields, cond_fields=[0, 2],
    n_obs_min=[19531, 19531], n_obs_max=[19531, 19531])
fp = {"snap": 29, "sensors": int(om.sum().item()),
      "idx_sum": int(oi[om > 0].sum().item())}
print(f"[probe] FINGERPRINT snap=29 sensors={fp['sensors']} idx_sum={fp['idx_sum']}", flush=True)
print(f"[probe] expected           sensors=39062 idx_sum=37987162596", flush=True)
fp["match"] = (fp["sensors"] == 39062 and fp["idx_sum"] == 37987162596)
out["fingerprint"] = fp

# ------------------------------------------------------- prior pointwise std
sub = torch.from_numpy(np.sort(rng.choice(coords.shape[1], size=400_000, replace=False))).to(DEV)
cq = coords[:, sub].contiguous()
P = 64
acc = torch.zeros(P, cq.shape[1], model.model.n_fields)
for k in range(P):
    torch.manual_seed(90_000 + k)
    acc[k] = model.sample_source(cq)[0].cpu()
pa = acc.numpy()
prior = {
    "n_draws": P,
    "pointwise_std_per_channel": [float(pa[:, :, j].std(axis=0, ddof=1).mean())
                                  for j in range(pa.shape[2])],
    "spatial_std_per_channel": [float(pa[0, :, j].std()) for j in range(pa.shape[2])],
    "lengthscale": cfg.get("rff_lengthscale"), "n_features": cfg.get("rff_features"),
}
print(f"[probe] PRIOR pointwise std per channel = {prior['pointwise_std_per_channel']}", flush=True)
out["prior"] = prior

# data scale for reference (normalized units)
fq = fields[0, sub].cpu().numpy()
out["data_std_per_channel"] = [float(fq[:, j].std()) for j in range(fq.shape[1])]
print(f"[probe] data std per channel = {out['data_std_per_channel']}", flush=True)

# ----------------------------------------------------------- grid reordering
c = dataset.coords_raw.numpy()
order = np.lexsort((c[:, 2], c[:, 1], c[:, 0]))
side = int(round(dataset.num_points ** (1 / 3)))
assert side ** 3 == dataset.num_points, side
print(f"[probe] grid side={side}", flush=True)


def radial_spec(vol):
    """Radially binned power spectrum of a [S,S,S] field, mean-removed."""
    S = vol.shape[0]
    v = vol - vol.mean()
    F = np.fft.rfftn(v)
    P = (F.real ** 2 + F.imag ** 2) / (S ** 3)
    kx = np.fft.fftfreq(S) * S
    kz = np.fft.rfftfreq(S) * S
    K = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    kb = np.minimum(K.astype(int), S // 2)
    n = np.bincount(kb.ravel(), minlength=S // 2 + 1)
    s = np.bincount(kb.ravel(), weights=P.ravel(), minlength=S // 2 + 1)
    return s / np.maximum(n, 1)


def corr_len(vol):
    """1/e correlation length in normalized coords (box side = 1)."""
    S = vol.shape[0]
    v = vol - vol.mean()
    F = np.fft.fftn(v)
    ac = np.fft.ifftn(np.abs(F) ** 2).real
    ac = ac / ac.flat[0]
    prof = 0.5 * (ac[:, 0, 0] + ac[0, :, 0])
    idx = np.where(prof < np.exp(-1.0))[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    p0, p1 = prof[i - 1], prof[i]
    f = (p0 - np.exp(-1.0)) / (p0 - p1 + 1e-12)
    return float((i - 1 + f) / S)


def analyse(dev_stack, tag):
    """dev_stack [K, N, C] deviations in dataset point order."""
    res = {}
    for j, nm in enumerate(dataset.field_names):
        specs, cls = [], []
        for k in range(dev_stack.shape[0]):
            vol = dev_stack[k, order, j].reshape(side, side, side)
            specs.append(radial_spec(vol)); cls.append(corr_len(vol))
        res[nm] = {"spec": np.mean(specs, axis=0).tolist(),
                   "corr_len": float(np.mean(cls)),
                   "std": float(dev_stack[:, :, j].std())}
        print(f"[probe] {tag} {nm}: dev std={res[nm]['std']:.4f} "
              f"corr_len={res[nm]['corr_len']:.4f}", flush=True)
    return res


# prior deviations (same K as ensembles)
pf = np.empty((8, dataset.num_points, model.model.n_fields), dtype=np.float32)
for k in range(8):
    torch.manual_seed(70_000 + k)
    pf[k] = model.sample_source(coords)[0].cpu().numpy()
out["prior_structure"] = analyse(pf - pf.mean(axis=0, keepdims=True), "PRIOR")
del pf

# --------------------------------------------- ensemble structure + timings
cases = [(19531, 4), (19531, 16), (390625, 4)]
snaps = [int(snap_ids[0]), int(snap_ids[1])]
out["ensembles"] = {}
for nobs, nfe in cases:
    per = []
    for si, snap in enumerate(snaps):
        it = dataset[snap]
        co = it["coords"].unsqueeze(0).to(DEV)
        fl = it["fields"].unsqueeze(0).to(DEV)
        torch.manual_seed(SEED * 777 + snap)
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=co, fields_full=fl, cond_fields=[0, 2],
            n_obs_min=[nobs, nobs], n_obs_max=[nobs, nobs])
        obs = {"coords": oc, "values": ov, "mask": om, "indices": oi, "field_ids": ofid}
        t0 = time.time()
        ens = sample_ensemble(model, co, obs, K=8, n_steps=nfe, chunk=262_144,
                              clamp_hard=False, seed=SEED * 131 + si).numpy()
        dt = time.time() - t0
        tru = fl[0].cpu().numpy()
        mean = ens.mean(axis=0)
        rec = {"snap": snap, "seconds_K8": round(dt, 1)}
        for j, nm in enumerate(dataset.field_names):
            rec[nm] = {"spread": float(np.sqrt((ens[:, :, j].std(axis=0, ddof=1) ** 2).mean())),
                       "rmse": float(np.sqrt(((mean[:, j] - tru[:, j]) ** 2).mean()))}
        print(f"[probe] n_obs={nobs} nfe={nfe} snap={snap} {dt:.0f}s "
              f"Ux spread={rec['Ux']['spread']:.4f} rmse={rec['Ux']['rmse']:.4f}", flush=True)
        if si == 0:
            rec["structure"] = analyse(ens - mean[None], f"ENS n{nobs} nfe{nfe}")
        per.append(rec)
    out["ensembles"][f"n{nobs}_nfe{nfe}"] = per

with open(f"{RD}/Evaluation/calib_probe.json", "w") as f:
    json.dump(out, f, indent=2)
print("[probe] wrote calib_probe.json", flush=True)
