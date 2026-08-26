"""Combine the upstream CQ cached-K/V path with our valid-key flash attention.

The vendored `cached_kv` block projects the sensor K/V once per condition
(their win: the projection is not repeated across latent blocks) and then
calls SDPA with an explicit float attn_mask carrying -inf at padded sensors.
That mask is what forces SDPA onto the CUTLASS backward, which at our
128-query x 39k-key shape measures ~70x slower per kernel than flash.

The two ideas are orthogonal. This patch keeps the single K/V projection and
replaces the masked call with per-item boolean selection of the valid K/V
rows followed by an unmasked SDPA, which is flash-eligible. It is applied by
monkey-patching the vendored class so the vendored package stays pristine.
"""

from typing import Mapping, Optional

import torch
import torch.nn.functional as F

from phycoflow_pointcloud.models import portable_core as _pc


def _prepare_kv_valid(self, kv, kv_padding_mask=None):
    """As upstream, but keep the padding mask instead of building -inf bias."""
    kv_in = self.norm_kv(kv)
    dim = self.attn.embed_dim
    kv_proj = F.linear(
        kv_in,
        self.attn.in_proj_weight[dim:],
        None if self.attn.in_proj_bias is None else self.attn.in_proj_bias[dim:],
    )
    key, value = kv_proj.chunk(2, dim=-1)
    b, m, _ = key.shape
    h = self.attn.num_heads
    hd = dim // h
    key = key.view(b, m, h, hd).transpose(1, 2)
    value = value.view(b, m, h, hd).transpose(1, 2)
    self.kv_projection_calls += 1
    return {"key": key, "value": value, "attn_mask": None,
            "valid": None if kv_padding_mask is None else ~kv_padding_mask.bool()}


def _forward_prepared_valid(self, q, prepared_kv: Mapping[str, Optional[torch.Tensor]]):
    q_in = self.norm_q(q)
    dim = self.attn.embed_dim
    q_proj = F.linear(
        q_in,
        self.attn.in_proj_weight[:dim],
        None if self.attn.in_proj_bias is None else self.attn.in_proj_bias[:dim],
    )
    b, l, _ = q_proj.shape
    h = self.attn.num_heads
    hd = dim // h
    q_proj = q_proj.view(b, l, h, hd).transpose(1, 2)
    key, value = prepared_kv["key"], prepared_kv["value"]
    valid = prepared_kv.get("valid")
    p = self.attn.dropout if self.training else 0.0

    if valid is None or bool(valid.all()):
        attn_out = F.scaled_dot_product_attention(q_proj, key, value,
                                                  attn_mask=None, dropout_p=p)
    else:
        outs = []
        for i in range(b):
            vi = valid[i]
            if not bool(vi.any()):
                outs.append(torch.zeros_like(q_proj[i:i + 1]))
                continue
            outs.append(F.scaled_dot_product_attention(
                q_proj[i:i + 1], key[i:i + 1][:, :, vi], value[i:i + 1][:, :, vi],
                attn_mask=None, dropout_p=p))
        attn_out = torch.cat(outs, dim=0)

    attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, dim)
    attn_out = self.attn.out_proj(attn_out)
    x = q + attn_out
    return x + self.ff(self.norm_ff(x))


_ORIG = {}


def enable():
    """Swap the vendored cached-K/V attention for the valid-key version."""
    cls = _pc.CachedKVCrossAttentionBlock if hasattr(
        _pc, "CachedKVCrossAttentionBlock") else _pc.CrossAttentionBlock
    if "prepare_kv" not in _ORIG:
        _ORIG["cls"] = cls
        _ORIG["prepare_kv"] = cls.prepare_kv
        _ORIG["forward_prepared"] = cls.forward_prepared
    cls.prepare_kv = _prepare_kv_valid
    cls.forward_prepared = _forward_prepared_valid


def disable():
    cls = _ORIG.get("cls")
    if cls is not None:
        cls.prepare_kv = _ORIG["prepare_kv"]
        cls.forward_prepared = _ORIG["forward_prepared"]
