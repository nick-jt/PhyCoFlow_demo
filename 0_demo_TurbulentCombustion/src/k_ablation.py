"""Accuracy cost of smaller inference K on the K=64-trained DemoN4 model."""
import numpy as np, torch
from ensemble_eval import load_run, sample_ensemble
from helpers import build_sparse_condition
device = torch.device('cuda:0')
model, dataset, _ = load_run('../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_main_DemoN4_20260813_122006', 'last.pt')
item = dataset[0]
coords = item['coords'][None].to(device); fields = item['fields'][None].to(device)
true = item['fields'].numpy()
torch.manual_seed(1)
oc,ov,om,oi,ofid = build_sparse_condition(coords_full=coords, fields_full=fields,
    cond_fields=[0,2], n_obs_min=[19531], n_obs_max=[19531])
obs={"coords":oc,"values":ov,"mask":om,"indices":oi,"field_ids":ofid}
for K in [64, 32, 16]:
    model.model.gather_topk = K
    ens = sample_ensemble(model, coords, obs, K=2, n_steps=16, chunk=262144, clamp_hard=False, seed=5)
    m = ens.numpy().mean(0)
    rel = [float(np.linalg.norm(m[:,j]-true[:,j])/np.linalg.norm(true[:,j])) for j in range(4)]
    print(f"K={K}: relL2 per field = {[round(r,3) for r in rel]} mean={np.mean(rel):.4f}", flush=True)
