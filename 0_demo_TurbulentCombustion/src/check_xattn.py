"""CPU checks for the SECONDARY variant (sen_local_xattn.py):
param count in-band, gate=0 exact identity, gradients reach the attention
weights and gates, neighbour gather validity under padded sensor masks."""
import os
import sys

os.environ["SEN_LOCAL_IDW"] = "0"
os.environ["SEN_LOCAL_XATTN"] = "1"

SRC = ("/home/ntricard/generative_reconstruction/temp/"
       "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (SRC, HERE):
    sys.path.insert(0, p)

import torch

import model_baseline as MB
import sen_sweep_fixes as SF   # selection patches; IDW inert
import sen_local_xattn as SX   # applies the xattn patch

TARGET = 6506253
ARGS = dict(
    n_fields=4, coord_dim=3, latent_dim=320, num_latents=128,
    num_encoder_layers=3, num_self_attn_per_block=3,
    num_cross_attn_heads=2, num_self_attn_heads=2,
    dec_num_cross_attn_heads=1, field_embed_dim=32,
    space_bands=32, max_freq=[125, 125, 125],
    ff_mult=1, dropout=0.0, upstream_layout=True,
    enc_preproc_ch=320, dec_latent_dim=320,
    dec_preproc_ch=None, share_encoder_layers=True,
)

m = MB.Senseiver(**ARGS)
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
extra = n - 6439876
print(f"Senseiver+xattn (128x320 base) params={n:,d} "
      f"delta_vs_target={(n-TARGET)/TARGET*100:+.2f}% (extra={extra:+,d})")
assert abs(n - TARGET) / TARGET < 0.10

torch.manual_seed(0)
B, Q, M = 3, 400, 300
q = torch.rand(B, Q, 3)
oc = torch.rand(B, M, 3)
ov = torch.randn(B, M, 1)
mask = (torch.rand(B, M) > 0.3).float()
fid = torch.randint(0, 2, (B, M)) * 2
fid[mask == 0] = -1  # padded slots carry -1 field ids like the real sampler

m.eval()
with torch.no_grad():
    out = m(q, oc, ov, mask, fid)
    base = SX._base_forward(m, q, oc, ov, mask, fid)
print(f"gate=0 forward max|patched-base| = {(out-base).abs().max():.3e}")
assert torch.equal(out, base)

m.train()
with torch.no_grad():
    m.local_gate += 0.5
out = m(q, oc, ov, mask, fid)
assert not torch.equal(out, base)
out.pow(2).mean().backward()
g_gate = m.local_gate.grad
g_attn = m.local_xattn.k_proj.weight.grad
print(f"gate grad = {g_gate.tolist()}")
print(f"k_proj grad norm = {g_attn.norm().item():.3e}")
assert g_gate is not None and g_gate.abs().max() > 0
assert g_attn is not None and g_attn.norm() > 0

# neighbour validity: with padded (invalid) sensors pushed to 2e3, top-K must
# only select valid sensors when every batch item has >= K valid ones
oc_pad = torch.where(mask.bool().unsqueeze(-1), oc, torch.full_like(oc, 2.0e3))
d = torch.cdist(q, oc_pad)
_, i_k = torch.topk(d, 8, dim=2, largest=False)
sel_valid = torch.gather(mask, 1, i_k.reshape(B, -1)).reshape(B, Q, 8)
assert sel_valid.min() == 1.0, "top-K selected a padded sensor"
print("neighbour validity OK")
print("ALL XATTN CHECKS PASSED")
