"""SECONDARY enhanced variant -- "Senseiver+xattn": local query-to-sensor
cross-attention over the top-K=8 nearest sensors (PLAN_IMPROVE_2026-08-30.md
section 1b, secondary design).  Separate module from sen_sweep_fixes.py so
the primary (IDW) arm's execution path is untouched; import ORDER:
sen_sweep_fixes first (selection patches), then this module.

Mechanism (output-level gated residual, like the primary):

    y(q) = senseiver_decoder(q) + gate[c] * head_c( Attn(q, N_8(q)) )

where N_8(q) are the 8 nearest VALID sensors (union of the observed
channels; the field embedding distinguishes them), attended with learned
weights: query stream = pos_enc(q); neighbour stream = [value | field_embed
| pos_enc(q - x_s)] (relative position, so the attention is
translation-aware).  Unlike the fixed IDW kernel of the primary, the
weighting AND the read-out are learned; gradients flow through values and
weights.  gate is a per-channel plain scalar init 0 -> exact base Senseiver
at step 0.

Env gates:
    SEN_LOCAL_XATTN=1   activate (adds ~46k params + n_fields gates;
                        checkpoints REQUIRE the patch at load time).
    SEN_XATTN_K         neighbours (default 8).
    SEN_XATTN_DIM       attention width (default 64).
    SEN_XATTN_CHUNK     query-chunk for the cdist/topk (default 2048).
Do NOT combine with SEN_LOCAL_IDW=1 (asserted below).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_baseline as MB

_XATTN = os.environ.get("SEN_LOCAL_XATTN", "0") == "1"
_K = int(os.environ.get("SEN_XATTN_K", "8"))
_DIM = int(os.environ.get("SEN_XATTN_DIM", "64"))
_CHUNK = int(os.environ.get("SEN_XATTN_CHUNK", "2048"))

print(f"[senxattn] SEN_LOCAL_XATTN={int(_XATTN)} K={_K} DIM={_DIM} "
      f"CHUNK={_CHUNK}", flush=True)

if _XATTN and os.environ.get("SEN_LOCAL_IDW", "0") == "1":
    raise SystemExit("[senxattn] SEN_LOCAL_XATTN and SEN_LOCAL_IDW are "
                     "mutually exclusive arms")

_S = MB.Senseiver
# capture whatever forward/init are live AFTER sen_sweep_fixes import; with
# SEN_LOCAL_IDW=0 these are the unpatched originals.
_base_init = _S.__init__
_base_forward = _S.forward


class _LocalXAttn(nn.Module):
    """Single-head cross-attention from a query point to its K neighbours."""

    def __init__(self, pos_dim: int, field_embed_dim: int, n_fields: int,
                 dim: int):
        super().__init__()
        kv_in = 1 + field_embed_dim + pos_dim  # value | field emb | rel pos
        self.q_proj = nn.Linear(pos_dim, dim)
        self.k_proj = nn.Linear(kv_in, dim)
        self.v_proj = nn.Linear(kv_in, dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                  nn.GELU(), nn.Linear(dim, n_fields))
        self.scale = dim ** -0.5

    def forward(self, q_pos, nb_feat):
        # q_pos [B,q,P]; nb_feat [B,q,K,kv_in]
        q = self.q_proj(q_pos).unsqueeze(2)          # [B,q,1,D]
        k = self.k_proj(nb_feat)                     # [B,q,K,D]
        v = self.v_proj(nb_feat)
        w = F.softmax((q * k).sum(-1) * self.scale, dim=-1)  # [B,q,K]
        h = (w.unsqueeze(-1) * v).sum(2)             # [B,q,D]
        return self.head(h)                          # [B,q,n_fields]


def _x_init(self, *args, **kwargs):
    _base_init(self, *args, **kwargs)
    if _XATTN:
        self.local_xattn = _LocalXAttn(self.pos_enc.out_dim,
                                       self.field_embed.embedding_dim,
                                       self.n_fields, _DIM)
        self.local_gate = nn.Parameter(torch.zeros(self.n_fields))
        self.local_xattn_k = _K
        n = sum(p.numel() for p in self.local_xattn.parameters())
        print(f"[senxattn] local cross-attn ACTIVE: K={_K} dim={_DIM} "
              f"extra_params={n + self.n_fields}", flush=True)


def _x_forward(self, query_coords, obs_coords, obs_values, obs_mask,
               obs_field_ids):
    base = _base_forward(self, query_coords, obs_coords, obs_values,
                         obs_mask, obs_field_ids)
    xa = getattr(self, "local_xattn", None)
    if xa is None:
        return base
    B, Q, D = query_coords.shape
    K = self.local_xattn_k
    res = torch.empty_like(base)
    # push invalid (padded) sensors far outside the unit box: never in top-K
    # as long as every batch item has >= K valid sensors (min draw 1953).
    oc = torch.where(obs_mask.bool().unsqueeze(-1), obs_coords,
                     torch.full_like(obs_coords, 2.0e3))
    fid = obs_field_ids.long().clamp(min=0)
    for s in range(0, Q, _CHUNK):
        q = query_coords[:, s:s + _CHUNK]                       # [B,q,D]
        with torch.no_grad():
            d = torch.cdist(q, oc)                              # [B,q,M]
            _, i_k = torch.topk(d, K, dim=2, largest=False)     # [B,q,K]
        nb_c = torch.gather(
            obs_coords.unsqueeze(1).expand(-1, q.shape[1], -1, -1), 2,
            i_k.unsqueeze(-1).expand(-1, -1, -1, D))            # [B,q,K,D]
        nb_v = torch.gather(
            obs_values[..., 0].unsqueeze(1).expand(-1, q.shape[1], -1), 2,
            i_k).unsqueeze(-1)                                  # [B,q,K,1]
        nb_f = torch.gather(
            fid.unsqueeze(1).expand(-1, q.shape[1], -1), 2, i_k)
        rel = self.pos_enc(
            (q.unsqueeze(2) - nb_c).reshape(B, -1, D)
        ).reshape(B, q.shape[1], K, -1)
        nb_feat = torch.cat([nb_v, self.field_embed(nb_f), rel], dim=-1)
        res[:, s:s + _CHUNK] = xa(self.pos_enc(q), nb_feat)
    return base + self.local_gate.view(1, 1, -1) * res


if _XATTN:
    _S.__init__ = _x_init
    _S.forward = _x_forward
