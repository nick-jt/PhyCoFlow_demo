"""Compiled-mode comparison: masked vs valid-key attention.

Three variants, each with a FRESH model load and dynamo reset:
  A. masked attention (pre-fix behavior), torch.compile as in production
  B. valid-key attention as implemented, torch.compile
  C. valid-key attention with the attention helper excluded from the compiled
     graph (torch.compiler.disable) - the fallback if B recompiles per step,
     since the per-item loop has data-dependent shapes.

Per-warmup-step times are printed: recompilation shows up as repeated
multi-second steps. Steady-state = mean of the last 6 steps. Peak memory is
reset after warmup so compile-time allocations are excluded. A kernel table
from torch.profiler on one steady step is the runtime breakdown available
under fusion (boundary events cannot see inside a compiled graph).
"""
import time, torch
from ensemble_eval import load_run
from helpers import build_sparse_condition
from train_pointcloud_ffm import sample_query_subset
from torch.profiler import profile, ProfilerActivity

RUN = ("../Save_TrainedModel/JHU/pointcloud_ffm/"
       "iclr_jhu_xcube_aug_DemoN15_20260818_083446")
device = torch.device("cuda:0")
B, Q, M = 20, 39062, 19531

def masked_attn(self, attn, q, kv, obs_mask):
    return attn(q=q, kv=kv, kv_padding_mask=~obs_mask.bool())

def run_variant(tag, patch=None, disable_attn=False):
    import torch._dynamo as dynamo
    dynamo.reset()
    model, dataset, cfg = load_run(RUN, "best.pt", str(device))
    net = model.model
    if patch is not None:
        net._cross_attn_valid = patch.__get__(net, type(net))
    if disable_attn:
        net._cross_attn_valid = torch.compiler.disable(net._cross_attn_valid)
    net.forward = torch.compile(net.forward, mode="max-autotune-no-cudagraphs")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    items = [dataset[i % len(dataset)] for i in range(B)]
    coords = torch.stack([it["coords"] for it in items]).to(device)
    fields = torch.stack([it["fields"] for it in items]).to(device)

    def step():
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields, cond_fields=[0, 2],
            n_obs_min=[1953], n_obs_max=[M])
        cq, fq, _ = sample_query_subset(coords=coords, fields=fields,
                                        n_query=Q, mode="uniform")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            loss, _ = model.training_loss(
                x1=fq, coords=cq, obs_coords=oc, obs_values=ov, obs_mask=om,
                obs_field_ids=ofid, compute_metrics=False)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print(f"\n===== {tag}")
    warm = [step() for _ in range(6)]
    print("  warmup steps:", " ".join(f"{w:.2f}" for w in warm))
    torch.cuda.reset_peak_memory_stats()
    steady = [step() for _ in range(6)]
    peak = torch.cuda.max_memory_allocated() / 1024**2
    mean = sum(steady) / len(steady)
    print(f"  steady: {mean:.3f} s/step   peak {peak:.0f} MB")

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
    rows = prof.key_averages().table(sort_by="cuda_time_total",
                                     row_limit=8).split("\n")
    print("  top CUDA kernels (fwd+bwd+opt of one step):")
    for r in rows[3:11]:
        print("   ", r[:135])
    del model, net, opt
    torch.cuda.empty_cache()
    return mean, peak

a = run_variant("A: masked attention, compiled", patch=masked_attn)
b = run_variant("B: valid-key, compiled")
c = run_variant("C: valid-key, attention outside compiled graph",
                disable_attn=True)
print("\n===== summary (steady s/step, peak MB)")
for tag, (m, p) in zip("ABC", (a, b, c)):
    print(f"  {tag}: {m:.3f} s   {p:.0f} MB")
