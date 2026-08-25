"""Correctness + profiling for the valid-key cross-attention rewrite."""
import time, torch
from collections import defaultdict
from ensemble_eval import load_run
from helpers import build_sparse_condition
from train_pointcloud_ffm import sample_query_subset

RUN = ("../Save_TrainedModel/JHU/pointcloud_ffm/"
       "iclr_jhu_xcube_aug_DemoN15_20260818_083446")
device = torch.device("cuda:0")
model, dataset, cfg = load_run(RUN, "best.pt", str(device))
net = model.model

# ---- correctness: new path vs original masked attention -------------------
B, M = 6, 19531
items = [dataset[i] for i in range(B)]
coords = torch.stack([it["coords"] for it in items]).to(device)
fields = torch.stack([it["fields"] for it in items]).to(device)
torch.manual_seed(0)
oc, ov, om, oi, ofid = build_sparse_condition(
    coords_full=coords, fields_full=fields, cond_fields=[0, 2],
    n_obs_min=[1953], n_obs_max=[M])
# scatter some invalid entries mid-buffer to mimic measurement ops
om2 = om.clone(); om2[:, ::7] = 0.0

net.eval()
with torch.no_grad():
    tokens = net._build_sensor_tokens(obs_coords=oc, obs_values=ov,
                                      obs_field_ids=ofid, obs_mask=om2)
    lat = net.latents.unsqueeze(0).expand(B, -1, -1)
    ref = net.input_cross_attn(q=lat, kv=tokens,
                               kv_padding_mask=~om2.bool())
    new = net._cross_attn_valid(net.input_cross_attn, lat, tokens, om2)
d = (ref - new).abs().max().item()
print(f"correctness (scattered invalid): max|masked - valid-key| = {d:.3e} "
      f"({'PASS' if d < 2e-5 else 'FAIL'})")

# ---- full training-step timing: before impossible now, so measure new ----
net.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
B, Q = 20, 39062
items = [dataset[i % len(dataset)] for i in range(B)]
coords = torch.stack([it["coords"] for it in items]).to(device)
fields = torch.stack([it["fields"] for it in items]).to(device)
times = defaultdict(float); reps, warm = 6, 3
for r in range(warm + reps):
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords, fields_full=fields, cond_fields=[0, 2],
        n_obs_min=[1953], n_obs_max=[19531])
    cq, fq, _ = sample_query_subset(coords=coords, fields=fields,
                                    n_query=Q, mode="uniform")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.autocast("cuda", torch.bfloat16, enabled=True):
        loss, _ = model.training_loss(x1=fq, coords=cq, obs_coords=oc,
                                      obs_values=ov, obs_mask=om,
                                      obs_field_ids=ofid,
                                      compute_metrics=False)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    if r >= warm:
        times["fwd"] += t1 - t0; times["bwd_opt"] += t2 - t1
        times["tot"] += t2 - t0
torch.cuda.reset_peak_memory_stats()
# one more step for peak memory under the new path
with torch.autocast("cuda", torch.bfloat16, enabled=True):
    loss, _ = model.training_loss(x1=fq, coords=cq, obs_coords=oc,
                                  obs_values=ov, obs_mask=om,
                                  obs_field_ids=ofid, compute_metrics=False)
opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"NEW training step: fwd {times['fwd']/reps:.3f} s, "
      f"bwd+opt {times['bwd_opt']/reps:.3f} s, total {times['tot']/reps:.3f} s, "
      f"peak {peak:.0f} MB")
print("OLD reference     : fwd 0.241 s, bwd+opt 0.875 s, total 1.116 s, peak 33064 MB")
