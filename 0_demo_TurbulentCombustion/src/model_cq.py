"""Adapter wiring the vendored GL_rbf_CQ portable package into our project.

The vendored package (src/phycoflow_pointcloud/, upstream v0.9.0-pre1) stays
pristine; everything fork-specific lives here:

  * ``PointCloudFFM_Ours`` adds the fork's training-loss features on top of
    the portable rectified-flow wrapper: logit-normal t sampling (read from
    the ``t_sampling`` instance attribute, set by the trainer), the binned
    spectral power loss on the structured block riding at the tail of the
    query tensor, and the ``compute_metrics`` contract our trainer expects.
    Sampling, context caching, and checkpoint key naming (``model.*`` /
    ``prior.*``) are inherited unchanged.
  * ``build_cq_model`` merges ``configs/gl_rbf_cq_core.yaml`` FIRST (the
    factory defaults differ from the validated core in nine keys - merging
    core is mandatory, per the upstream guide), applies our project-owned
    overrides from the run args, builds through the upstream factory so the
    construction order (and therefore checkpoint schema) matches theirs, and
    rewraps the result in ``PointCloudFFM_Ours``.
  * ``n_obs_field_types`` support (wing surface-sensor vocabulary) swaps the
    ``field_embed`` row count after construction. This intentionally departs
    from the upstream oracle schema - acceptable because every CQ model we
    use is trained fresh; JHU/FireBench pass None and keep the oracle shape.

Microbatched training is deliberately not used (user directive: runtime
priority; at our 39k queries it would re-traverse the condition-side backward
~19x for memory savings we do not need).
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from phycoflow_pointcloud import PointCloudFFM, build_pointcloud_model

_CORE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gl_rbf_cq_core.yaml"


class PointCloudFFM_Ours(PointCloudFFM):
    """Portable CQ wrapper + fork training-loss features (spectral, t-sampling)."""

    def training_loss(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
        compute_metrics: bool = True,
        spectral_block_shape: Optional[Sequence[int]] = None,
        spectral_weight: float = 0.0,
        spectral_bins: int = 12,
        spectral_window: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del obs_indices  # only used for reconstruction clamping
        x0 = self.sample_source(coords)

        bsz = x1.shape[0]
        if getattr(self, "t_sampling", "uniform") == "logit_normal":
            t = torch.sigmoid(torch.randn(bsz, device=x1.device, dtype=x1.dtype))
        else:
            t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)

        x_t = self.simulate(t, x0, x1)
        target = self.target_vector_field(x0, x1)
        pred = self.model(t, x_t, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        loss = F.mse_loss(pred, target)

        spec_val = None
        if spectral_weight > 0.0 and spectral_block_shape is not None:
            from spectral_loss import binned_spectral_loss
            n_blk = int(spectral_block_shape[0]) * int(spectral_block_shape[1]) \
                * int(spectral_block_shape[2])
            t_col = t.view(-1, *([1] * (x1.dim() - 1)))
            x1_hat = x_t + (1.0 - t_col) * pred
            spec = binned_spectral_loss(
                x1_hat[:, -n_blk:].float(), x1[:, -n_blk:].float(),
                block_shape=spectral_block_shape, n_bins=spectral_bins,
                sample_weight=t.detach().float(),
                window=spectral_window,
            )
            loss = loss + spectral_weight * spec
            spec_val = spec

        if not compute_metrics:
            return loss, ({"spectral": float(spec_val.detach().cpu())}
                          if spec_val is not None else {})
        metrics = {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(target.pow(2).mean().sqrt().detach().cpu()),
        }
        if spec_val is not None:
            metrics["spectral"] = float(spec_val.detach().cpu())
        return loss, metrics


# Keys the run config (args) is allowed to override on top of the core model
# defaults. Everything else model-side stays at the validated core values.
_ARG_OVERRIDES = (
    "prior", "rff_features", "rff_lengthscale", "sigma_min",
    "neighbor_backend", "gather_query_chunk_size",
)


def load_cq_config(args=None) -> dict:
    cfg = yaml.safe_load(open(_CORE_CONFIG))
    cfg["coord_dim"] = 3
    if args is not None:
        for key in _ARG_OVERRIDES:
            val = getattr(args, key, None)
            if val is not None:
                cfg[key] = val
    return cfg


def build_cq_model(args, train_set, device) -> Tuple[PointCloudFFM_Ours, dict]:
    """Build a GL_rbf_CQ model for our trainer/eval stack.

    Returns (model, resolved_config). The model exposes the same duck-typed
    surface the rest of the project relies on: .model / .prior naming (and
    hence checkpoint keys), positional 7-arg backbone forward, sample(),
    sample_source(), and our training_loss contract.
    """
    cfg = load_cq_config(args)
    built = build_pointcloud_model(cfg, n_fields=int(train_set.num_fields), device="cpu")
    backbone, prior = built.model, built.prior

    n_obs_field_types = getattr(train_set, "n_obs_field_types", None)
    if getattr(args, "n_obs_field_types", None):
        n_obs_field_types = args.n_obs_field_types
    if n_obs_field_types and int(n_obs_field_types) != backbone.field_embed.num_embeddings:
        backbone.field_embed = nn.Embedding(
            int(n_obs_field_types), backbone.field_embed.embedding_dim
        )
        backbone.n_obs_field_types = int(n_obs_field_types)

    model = PointCloudFFM_Ours(
        backbone, prior, sigma_min=float(cfg.get("sigma_min", 1.0e-4))
    ).to(device)
    if getattr(args, "t_sampling", None):
        model.t_sampling = args.t_sampling
    return model, cfg
