"""Smoke gates for the GL_rbf_CQ adapter (CPU, tiny dims).

Gates: seeded double-build determinism; 3-D forward/backward through OUR
training_loss incl. spectral block + logit-normal t; mixed valid sensor
counts; n_obs_field_types embedding swap; ckpt round-trip via state_dict.
"""
import sys, types
import torch

sys.path.insert(0, ".")
from model_cq import build_cq_model, load_cq_config


class TinySet:
    num_fields = 4
    n_obs_field_types = None


def tiny_args(**kw):
    a = types.SimpleNamespace(
        prior="rff", rff_features=16, rff_lengthscale=0.15, sigma_min=1e-4,
        neighbor_backend="torch", gather_query_chunk_size=None,
        t_sampling="logit_normal", n_obs_field_types=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def shrink(cfg):
    cfg.update(hidden_dim=32, cond_dim=16, field_embed_dim=8, latent_dim=32,
               num_latents=8, num_heads=4, num_latent_blocks=1, ff_mult=2,
               gather_topk=4, cq_query_dim=16, cq_readout_rank=8,
               cq_readout_heads=2, cq_time_embed_dim=16,
               fourier_pe_num_bands=4)
    return cfg


import model_cq
_orig = model_cq.load_cq_config
model_cq.load_cq_config = lambda args=None: shrink(_orig(args))

# 1) seeded double-build determinism
torch.manual_seed(0)
m1, _ = build_cq_model(tiny_args(), TinySet(), "cpu")
torch.manual_seed(0)
m2, _ = build_cq_model(tiny_args(), TinySet(), "cpu")
sd1, sd2 = m1.state_dict(), m2.state_dict()
assert sd1.keys() == sd2.keys()
assert all(torch.equal(sd1[k], sd2[k]) for k in sd1), "seeded build not deterministic"
prefixes = {k.split(".")[0] for k in sd1}
assert prefixes == {"model", "prior"}, prefixes
print(f"[1] deterministic build OK ({len(sd1)} keys, prefixes {sorted(prefixes)})")

# 2) forward/backward with spectral block + logit-normal t, mixed valid counts
B, Nq, M, F_ = 2, 64, 12, 4
blk = (2, 2, 2); n_blk = 8
coords = torch.rand(B, Nq + n_blk, 3)
x1 = torch.randn(B, Nq + n_blk, F_)
oc = torch.rand(B, M, 3); ov = torch.randn(B, M, 1)
om = torch.ones(B, M); om[0, 7:] = 0        # mixed valid counts
ofid = torch.randint(0, F_, (B, M))
loss, parts = m1.training_loss(
    x1=x1, coords=coords, obs_coords=oc, obs_values=ov, obs_mask=om,
    obs_field_ids=ofid, compute_metrics=True,
    spectral_block_shape=blk, spectral_weight=0.05, spectral_bins=3)
assert "spectral" in parts and torch.isfinite(loss), parts
loss.backward()
grads = [p.grad for p in m1.parameters() if p.grad is not None]
assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
print(f"[2] fwd/bwd OK loss={float(loss):.4f} spectral={parts['spectral']:.4f} "
      f"({len(grads)} grads)")

# 3) sample() runs (legacy_full + cached_streamed parity)
with torch.no_grad():
    sA = None
    for mode in ("legacy_full", "cached_streamed"):
        torch.manual_seed(7)
        s = m1.sample(coords=coords, obs_coords=oc, obs_values=ov, obs_mask=om,
                      obs_field_ids=ofid, n_steps=2, obs_consistency_mode="none",
                      reconstruction_execution_mode=mode)
        sA = s if sA is None else sA
    diff = float((sA - s).abs().max())
assert diff < 1e-4, f"legacy vs cached parity {diff}"
print(f"[3] sample parity legacy_full vs cached_streamed: max|d|={diff:.2e}")

# 4) n_obs_field_types swap
class WingSet(TinySet):
    n_obs_field_types = 9
torch.manual_seed(0)
mw, _ = build_cq_model(tiny_args(), WingSet(), "cpu")
assert mw.model.field_embed.num_embeddings == 9
print("[4] n_obs_field_types swap OK (field_embed rows = 9)")

# 5) checkpoint round-trip
torch.manual_seed(1)
m3, _ = build_cq_model(tiny_args(), TinySet(), "cpu")
m3.load_state_dict(m1.state_dict())          # strict=True default
assert all(torch.equal(a, b) for a, b in zip(m3.state_dict().values(),
                                             m1.state_dict().values()))
print("[5] strict state-dict round-trip OK")
print("ALL CQ ADAPTER GATES PASSED")
