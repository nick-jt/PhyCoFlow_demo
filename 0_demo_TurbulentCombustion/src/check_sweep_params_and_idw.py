"""Param counts for the sweep arms + CPU correctness checks for the IDW patch.

Run with the job environment python (source ~/envs/jhtdb).  CPU-only; makes
no canonical-protocol claims (those come from the compute-node evals).
"""
import os
import sys

os.environ.setdefault("SEN_LOCAL_IDW", "1")  # activate patch B for the check
os.environ.setdefault("SEN_IDW_K", "8")

SRC = ("/home/ntricard/generative_reconstruction/temp/"
       "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (SRC, HERE):
    sys.path.insert(0, p)

import numpy as np
import torch

import model_baseline as MB

TARGET = 6506253
ARGS = dict(
    n_fields=4, coord_dim=3, latent_dim=320,
    num_encoder_layers=3, num_self_attn_per_block=3,
    num_cross_attn_heads=2, num_self_attn_heads=2,
    dec_num_cross_attn_heads=1, field_embed_dim=32,
    space_bands=32, max_freq=[125, 125, 125],
    ff_mult=1, dropout=0.0, upstream_layout=True,
    enc_preproc_ch=320, dec_latent_dim=320,
    dec_preproc_ch=None, share_encoder_layers=True,
)


def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---- unpatched counts (capacity arms) -------------------------------------
base_fw = MB.Senseiver.forward
base_init = MB.Senseiver.__init__
import sen_sweep_fixes as SF  # applies patch (env set above)

# temporarily un-patch to count the capacity arms exactly as they will run
MB.Senseiver.__init__ = SF._orig_s_init
MB.Senseiver.forward = SF._orig_s_forward
for name, nl in [("DemoN43  64x320 (ref)", 64), ("Senseiver-128 (DemoN44)", 128),
                 ("Senseiver-256 (DemoN45)", 256), ("Senseiver-512 (DemoN46)", 512)]:
    n = count(MB.Senseiver(num_latents=nl, **ARGS))
    print(f"{name:26s} params={n:>10,d} delta={(n-TARGET)/TARGET*100:+.2f}% "
          f"bottleneck={nl*320:,d}")

# ---- patched count (Senseiver+local, 128x320 base) ------------------------
MB.Senseiver.__init__ = SF._s_init
MB.Senseiver.forward = SF._s_forward
m = MB.Senseiver(num_latents=128, **ARGS)
n = count(m)
print(f"{'Senseiver+local (DemoN47)':26s} params={n:>10,d} "
      f"delta={(n-TARGET)/TARGET*100:+.2f}% (gate adds "
      f"{n - 6439876:+d} vs DemoN44)")
assert n == 6439876 + 4, n
assert "local_gate" in dict(m.named_parameters())

# ---- IDW correctness vs scipy cKDTree (kd_predict numerics) ----------------
torch.manual_seed(0)
B, Q, M, C = 3, 500, 400, 4
q = torch.rand(B, Q, 3)
oc = torch.rand(B, M, 3)
ov = torch.randn(B, M, 1)
mask = (torch.rand(B, M) > 0.15).float()
fid = torch.randint(0, 2, (B, M)) * 2  # fields 0 and 2 observed
# make a handful of queries EXACT sensor hits
q[0, :5] = oc[0, 100:105]

# strict comparison against the exact kernel
_saved_mode = SF._IDW_CDIST_MODE
SF._IDW_CDIST_MODE = "donot_use_mm_for_euclid_dist"
idw = SF._idw_interp(q, oc, ov, mask, fid, n_fields=C, k=8, chunk=64)
SF._IDW_CDIST_MODE = _saved_mode

from scipy.spatial import cKDTree
ref = np.zeros((B, Q, C), dtype=np.float64)
for b in range(B):
    for f in (0, 2):
        sel = ((fid[b] == f) & mask[b].bool()).numpy()
        tree = cKDTree(oc[b].numpy()[sel])
        vals = ov[b, :, 0].numpy()[sel]
        d, nn = tree.query(q[b].numpy(), k=8, workers=-1)
        exact = d[:, 0] < 1e-11
        d = np.maximum(d, 1e-12)
        w = 1.0 / d
        ref[b, :, f] = (w * vals[nn]).sum(1) / w.sum(1)
        ref[b, exact, f] = vals[nn[exact, 0]]
err = np.abs(idw.numpy() - ref).max()
print(f"IDW (exact kernel) vs cKDTree max|diff| = {err:.3e}")
assert err < 5e-5, err
assert np.abs(idw.numpy()[..., 1]).max() == 0.0  # unobserved channels: zero
assert np.abs(idw.numpy()[..., 3]).max() == 0.0

# production (mm) mode: exact hits must still resolve exactly; elsewhere the
# ~5e-4 distance error perturbs weights by O(1%) at these test distances
idw_mm = SF._idw_interp(q, oc, ov, mask, fid, n_fields=C, k=8, chunk=64)
err_mm = np.abs(idw_mm.numpy() - ref).max()
exact_err = np.abs(idw_mm.numpy()[0, :5, [0, 2][0]] - ref[0, :5, 0]).max()
print(f"IDW (mm mode)      vs cKDTree max|diff| = {err_mm:.3e} "
      f"(exact-hit rows max|diff| = {exact_err:.3e})")
assert err_mm < 2e-2, err_mm
assert exact_err < 1e-6, exact_err
idw = idw_mm  # the arms train with mm mode; test the forward against it

# ---- gate=0 => identical to unpatched forward ------------------------------
m.eval()
with torch.no_grad():
    out_patched = m(q, oc, ov, mask, fid)
    out_base = SF._orig_s_forward(m, q, oc, ov, mask, fid)
print(f"gate=0 forward max|patched-base| = {(out_patched-out_base).abs().max():.3e}")
assert torch.equal(out_patched, out_base)

# ---- gate learns: gradient reaches the gate --------------------------------
m.train()
with torch.no_grad():
    m.local_gate += torch.tensor([0.5, 0.0, 0.5, 0.0])
out = m(q, oc, ov, mask, fid)
with torch.no_grad():
    expect = out_base + m.local_gate.view(1, 1, -1) * idw.to(out.dtype)
print(f"gate=0.5 forward max|out-(base+g*idw)| = {(out-expect).abs().max():.3e}")
assert (out - expect).abs().max() < 5e-5
loss = out.pow(2).mean()
loss.backward()
g = m.local_gate.grad
print(f"gate grad = {g.tolist()}")
assert g is not None and g[0] != 0 and g[2] != 0 and g[1] == 0 and g[3] == 0
print("ALL CHECKS PASSED")
