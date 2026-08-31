"""Probe: do helpers.build_sparse_condition and helpers_baseline.build_sparse_condition
produce BIT-IDENTICAL sensor layouts from the same torch seed on CUDA?

ensemble_eval.py (canonical driver) uses helpers.build_sparse_condition.
model_baseline.py (every baseline) uses helpers_baseline.build_sparse_condition.
"""
import torch
import helpers
import helpers_baseline

dev = torch.device("cuda:0")
B, N, C = 1, 1_953_125, 4
coords = torch.rand(B, N, 3, device=dev)
fields = torch.randn(B, N, C, device=dev)
COND = [0, 2]
NOBS = [19531, 19531]


def draw(fn, seed):
    torch.manual_seed(seed)
    out = fn(coords_full=coords, fields_full=fields, cond_fields=COND,
             n_obs_min=NOBS, n_obs_max=NOBS)
    return out[3]          # obs_indices


for seed in (0, 1, 7):
    a = draw(helpers.build_sparse_condition, seed)
    b = draw(helpers_baseline.build_sparse_condition, seed)
    same = torch.equal(a, b)
    overlap = len(set(a.flatten().tolist()) & set(b.flatten().tolist()))
    print(f"[probe] seed={seed} identical={same} "
          f"index_overlap={overlap}/{a.numel()} "
          f"({100.0*overlap/a.numel():.3f}%)", flush=True)

# self-consistency: same fn + same seed must reproduce
print("[probe] helpers self-repro:",
      torch.equal(draw(helpers.build_sparse_condition, 0),
                  draw(helpers.build_sparse_condition, 0)), flush=True)
print("[probe] helpers_baseline self-repro:",
      torch.equal(draw(helpers_baseline.build_sparse_condition, 0),
                  draw(helpers_baseline.build_sparse_condition, 0)), flush=True)

# Is the divergence caused by the extra CUDA randint in helpers_baseline?
torch.manual_seed(0)
r1 = torch.randperm(N, device=dev)[:5]
torch.manual_seed(0)
_ = torch.randint(low=19531, high=19532, size=(1,), device=dev)   # baseline's extra draw
r2 = torch.randperm(N, device=dev)[:5]
torch.manual_seed(0)
_ = torch.randint(low=19531, high=19532, size=(1,))               # helpers' CPU draw
r3 = torch.randperm(N, device=dev)[:5]
print(f"[probe] randperm after no draw   {r1.tolist()}", flush=True)
print(f"[probe] randperm after CUDA randint {r2.tolist()}", flush=True)
print(f"[probe] randperm after CPU randint  {r3.tolist()}", flush=True)
