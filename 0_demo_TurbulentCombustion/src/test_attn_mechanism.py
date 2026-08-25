"""Isolate the cross-attention backward slowdown and identify its mechanism.

Conditions on the REAL module (net.input_cross_attn) at the real shapes
(B=20, L=128 latents, M=39062 sensor keys, D=256, bf16 autocast):

  A. key_padding_mask as in training (all-False mask: every sensor valid,
     which is exactly the training case at max sensor draw)
  B. mask=None (drops masking entirely)
  C. mask with 30% padded entries (mixed-count batch case)
  D. mask=None on PRE-FILTERED keys (M reduced to the valid 70%)

If the hypothesis (mask blocks the fused kernel; math path materializes
[B,H,L,M] grads) is right: A and C are slow in backward, B and D are fast,
and the profiler shows softmax_backward_data + large bmm kernels in A but
fused attention kernels in B. If A is fast and only C is slow, the mask
content matters, not its presence. If all are equally slow, the hypothesis
is wrong and the cost is elsewhere (e.g. the MHA out-projection).
"""
import time, torch
from ensemble_eval import load_run

RUN = ("../Save_TrainedModel/JHU/pointcloud_ffm/"
       "iclr_jhu_xcube_aug_DemoN15_20260818_083446")
device = torch.device("cuda:0")
model, dataset, cfg = load_run(RUN, "best.pt", str(device))
net = model.model
attn = net.input_cross_attn
attn.train()

B, L, M, D = 20, net.latents.shape[0], 39062, net.latents.shape[1]
print(f"shapes: B={B} L={L} M={M} D={D}")

def bench(tag, mask, m_keys=M, reps=5, warm=2):
    q = net.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
    kv = torch.randn(B, m_keys, D, device=device, requires_grad=True)
    fw = bw = 0.0
    for r in range(warm + reps):
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = attn(q=q, kv=kv, kv_padding_mask=mask)
            torch.cuda.synchronize(); t1 = time.perf_counter()
        g = torch.randn_like(out)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        out.backward(g)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        kv.grad = None
        for p in attn.parameters(): p.grad = None
        if r >= warm:
            fw += t1 - t0; bw += t3 - t2
    print(f"{tag:38s} fwd {fw/reps*1e3:7.1f} ms   bwd {bw/reps*1e3:7.1f} ms   "
          f"ratio {bw/max(fw,1e-9):4.1f}x")
    return bw / reps

mask_allvalid = torch.zeros(B, M, dtype=torch.bool, device=device)
mask_30pad = torch.zeros(B, M, dtype=torch.bool, device=device)
mask_30pad[:, int(0.7 * M):] = True

bench("A: key_padding_mask (all valid)", mask_allvalid)
bench("B: mask=None", None)
bench("C: key_padding_mask (30% padded)", mask_30pad)
bench("D: mask=None, pre-filtered 70% keys", None, m_keys=int(0.7 * M))

# Kernel-level evidence for A vs B
from torch.profiler import profile, ProfilerActivity
for tag, mask in (("A(mask)", mask_allvalid), ("B(none)", None)):
    q = net.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
    kv = torch.randn(B, M, D, device=device, requires_grad=True)
    with torch.autocast("cuda", torch.bfloat16, enabled=True):
        out = attn(q=q, kv=kv, kv_padding_mask=mask)
    g = torch.randn_like(out)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out.backward(g)
    torch.cuda.synchronize()
    print(f"\n=== top backward CUDA kernels, condition {tag} ===")
    for row in prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=6).split("\n")[:12]:
        print(row[:150])
    kv.grad = None
    for p in attn.parameters(): p.grad = None
