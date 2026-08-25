"""Verify the ODE-step cache changes nothing, then measure the speedup."""
import time, torch
from ensemble_eval import load_run
from helpers import build_sparse_condition

RUN = ("../Save_TrainedModel/JHU/pointcloud_ffm/"
       "iclr_jhu_xcube_aug_DemoN15_20260818_083446")
device = torch.device("cuda:0")
model, dataset, cfg = load_run(RUN, "best.pt", str(device))
model.eval()

item = dataset[0]
coords_full = item["coords"][None].to(device)
fields_full = item["fields"][None].to(device)
torch.manual_seed(0)
oc, ov, om, oi, ofid = build_sparse_condition(
    coords_full=coords_full, fields_full=fields_full,
    cond_fields=[0, 2], n_obs_min=[19531], n_obs_max=[19531])

CH = 131072
def reconstruct(n_steps, disable_cache=False):
    outs = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        for c0 in range(0, coords_full.shape[1], CH):
            cc = coords_full[:, c0:c0+CH]
            torch.manual_seed(1234 + c0)     # same prior draw both paths
            if disable_cache:
                model.model._ode_cache = "disabled"   # non-None -> never fresh
            out = model.sample(coords=cc, obs_coords=oc, obs_values=ov,
                               obs_mask=om, obs_field_ids=ofid,
                               n_steps=n_steps, obs_consistency_mode="none")
            if disable_cache:
                model.model._ode_cache = None
            outs.append(out.float().cpu())
    return torch.cat(outs, dim=1)

# correctness at NFE=4 on the first chunk only (fast)
sub = coords_full[:, :CH]
torch.manual_seed(7)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
    model.model._ode_cache = "disabled"
    torch.manual_seed(99); a = model.sample(coords=sub, obs_coords=oc, obs_values=ov,
        obs_mask=om, obs_field_ids=ofid, n_steps=4, obs_consistency_mode="none").float().cpu()
    model.model._ode_cache = None
    torch.manual_seed(99); b = model.sample(coords=sub, obs_coords=ov.new_zeros(0) if False else oc, obs_values=ov,
        obs_mask=om, obs_field_ids=ofid, n_steps=4, obs_consistency_mode="none").float().cpu()
diff = (a-b).abs().max().item()
print(f"cache correctness: max |uncached - cached| = {diff:.3e}  "
      f"({'PASS' if diff < 1e-5 else 'FAIL'})")

for ns in (2, 4, 8, 16):
    for tag, dis in (("uncached", True), ("cached", False)):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        reconstruct(ns, disable_cache=dis)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        print(f"NFE={ns:2d} {tag:9s}: {dt:6.2f} s")
