"""Radial spectrum of the TRUE fields, for comparison with the prior and with
the ensemble-deviation spectra measured in calib_probe.py. CPU only."""
import json, os
import numpy as np, torch
from helpers import TurbulentCombustionH5Dataset

RD = ("/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/"
      "0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/"
      "iclr_jhu_xcube_spec02_DemoN29_20260822_140100")
cfg = json.load(open(f"{RD}/args.json"))
ds = TurbulentCombustionH5Dataset(cfg["data"], split="val", train_ratio=cfg["train_ratio"],
                                  field_names=cfg["field_names"], seed=cfg["seed"],
                                  time_stride=1, stats_path=f"{RD}/dataset_stats.pt")
c = ds.coords_raw.numpy()
order = np.lexsort((c[:, 2], c[:, 1], c[:, 0]))
side = int(round(ds.num_points ** (1/3)))

def radial_spec(vol):
    S = vol.shape[0]; v = vol - vol.mean()
    F = np.fft.rfftn(v); P = (F.real**2 + F.imag**2)/(S**3)
    kx = np.fft.fftfreq(S)*S; kz = np.fft.rfftfreq(S)*S
    K = np.sqrt(kx[:,None,None]**2 + kx[None,:,None]**2 + kz[None,None,:]**2)
    kb = np.minimum(K.astype(int), S//2)
    n = np.bincount(kb.ravel(), minlength=S//2+1)
    s = np.bincount(kb.ravel(), weights=P.ravel(), minlength=S//2+1)
    return s/np.maximum(n,1)

def corr_len(vol):
    S = vol.shape[0]; v = vol - vol.mean()
    F = np.fft.fftn(v); ac = np.fft.ifftn(np.abs(F)**2).real; ac /= ac.flat[0]
    prof = 0.5*(ac[:,0,0] + ac[0,:,0])
    idx = np.where(prof < np.exp(-1.0))[0]
    if len(idx)==0: return float('nan')
    i = idx[0]; p0,p1 = prof[i-1], prof[i]
    f = (p0-np.exp(-1.0))/(p0-p1+1e-12)
    return float((i-1+f)/S)

out = {}
rng = np.random.default_rng(0)
snaps = [int(s) for s in rng.choice(len(ds), size=50, replace=False)[:2]]
for snap in snaps:
    fl = ds[snap]["fields"].numpy()
    r = {}
    for j, nm in enumerate(ds.field_names):
        vol = fl[order, j].reshape(side, side, side)
        r[nm] = {"spec": radial_spec(vol).tolist(), "corr_len": corr_len(vol),
                 "std": float(fl[:, j].std())}
        print(f"[truth] snap={snap} {nm}: std={r[nm]['std']:.4f} corr_len={r[nm]['corr_len']:.4f}")
    out[str(snap)] = r
json.dump(out, open(f"{RD}/Evaluation/calib_truth_spec.json", "w"))
print("wrote calib_truth_spec.json")
