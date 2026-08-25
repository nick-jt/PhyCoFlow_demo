"""Spectral verdict on the binned spectral loss: N22 vs N15, judged on the
metric the loss actually optimizes (radial shell energy ratios on held-out
cube-3 snapshots), averaged over several snapshots and both at NFE=2 and 16."""
import numpy as np, torch
from ensemble_eval import load_run, sample_ensemble
from helpers import build_sparse_condition

GRID = 125
def shell(g):
    f = np.abs(np.fft.fftn(g))**2
    k = np.fft.fftfreq(GRID)*GRID
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    kb = np.round(np.sqrt(KX**2+KY**2+KZ**2)).astype(int)
    return np.bincount(kb.ravel(), weights=f.ravel())

device = torch.device('cuda:0')
RUNS = [
 ("N15 (no spectral)", "../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_aug_DemoN15_20260818_083446"),
 ("N29 (+spectral 0.02)", "../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_20260822_140100"),
]
import glob
# resolve actual N22 dir name
RUNS[1] = ("N29 (+spectral 0.02)", sorted(glob.glob(
    "../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*"))[-1])

for tag, rd in RUNS:
    model, dataset, _ = load_run(rd, "best.pt", str(device))
    for ns in (2, 16):
        stats = {"relL2": [], "inertial": [], "dissip": []}
        for si in (0, 3, 12, 23):
            item = dataset[si]
            coords = item['coords'][None].to(device)
            fields = item['fields'][None].to(device)
            true = item['fields'].numpy()
            torch.manual_seed(100 + si)
            oc, ov, om, oi, ofid = build_sparse_condition(
                coords_full=coords, fields_full=fields, cond_fields=[0, 2],
                n_obs_min=[19531], n_obs_max=[19531])
            obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
                   "field_ids": ofid}
            ens = sample_ensemble(model, coords, obs, K=1, n_steps=ns,
                                  chunk=262144, clamp_hard=False, seed=5+si)
            e = ens[0].numpy()
            for j in (0, 1, 2):
                sp = shell(e[:, j].reshape(GRID, GRID, GRID))
                st = shell(true[:, j].reshape(GRID, GRID, GRID))
                stats["relL2"].append(np.linalg.norm(e[:, j]-true[:, j])
                                      / np.linalg.norm(true[:, j]))
                stats["inertial"].append(sp[8:32].sum()/st[8:32].sum())
                stats["dissip"].append(sp[32:63].sum()/st[32:63].sum())
        print(f"{tag} NFE={ns:2d}: relL2 {np.mean(stats['relL2']):.4f} | "
              f"inertial-band ratio {np.mean(stats['inertial']):.3f} | "
              f"dissipation-band {np.mean(stats['dissip']):.3f}", flush=True)
