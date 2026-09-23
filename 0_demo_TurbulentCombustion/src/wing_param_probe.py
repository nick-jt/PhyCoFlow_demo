#!/usr/bin/env python3
"""Parameter-count probe for the SHIFT-WING baselines (CPU only, no data).

The campaign matches every baseline to our model's budget: the wing DMF-Gen
run `iclr_wing_v3_expanded_DemoN13_20260818_084059` reports

    Model parameters:  6,506,893 total  (6,506,893 trainable)

This script instantiates the three point-native baselines at candidate widths
and prints trainable parameter counts, so the achieved number that goes into
each config is measured rather than guessed. Run it from `src/`:

    python wing_param_probe.py
"""
from __future__ import annotations

import torch

import model_baseline as MB

TARGET = 6_506_893
N_FIELDS = 4
N_OBS_FIELD_TYPES = 9


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def senseiver(latent_dim: int, num_latents: int = 128, enc: int = 3,
              sa: int = 3, fe: int = 32) -> int:
    return count(MB.Senseiver(
        n_fields=N_FIELDS, coord_dim=3, num_latents=num_latents,
        latent_dim=latent_dim, num_encoder_layers=enc,
        num_self_attn_per_block=sa, num_cross_attn_heads=8,
        num_self_attn_heads=8, dec_num_cross_attn_heads=8,
        field_embed_dim=fe, space_bands=32, max_freq=64.0,
        ff_mult=4, dropout=0.0, n_obs_field_types=N_OBS_FIELD_TYPES))


def mlp_rbf(hidden: int, cond: int, fe: int = 128) -> int:
    return count(MB.DeterministicMLPRBFRegressor(MB.ConditionalPointMLPRBF(
        n_fields=N_FIELDS, coord_dim=3, hidden_dim=hidden, cond_dim=cond,
        field_embed_dim=fe, rbf_sigma=0.05, use_fourier_pe=True,
        fourier_pe_num_bands=32, fourier_pe_max_freq=64.0,
        n_obs_field_types=N_OBS_FIELD_TYPES)))


def sit(hidden: int, depth: int, heads: int = 4) -> int:
    return count(MB.SiTPhysics(
        input_size_h=0, input_size_w=0, patch_size=4,
        in_channels=N_FIELDS, cond_channels=2 * N_OBS_FIELD_TYPES + 1,
        hidden_size=hidden, depth=depth, num_heads=heads, mlp_ratio=4.0,
        tokenizer="pointnet", coord_dim=3, fourier_num_freqs=64,
        fourier_scale=10.0))


def report(name: str, rows: list[tuple[str, int]]) -> None:
    print(f"\n=== {name} (target {TARGET:,}) ===")
    for label, n in rows:
        print(f"  {label:42s} {n:>10,}  ({100.0 * n / TARGET - 100:+.1f}%)")


if __name__ == "__main__":
    report("senseiver", [
        (f"latent_dim={ld} num_latents=128 enc=3 sa=3", senseiver(ld))
        for ld in (192, 208, 224, 232, 240, 256)
    ])
    report("mlp_rbf", [
        (f"hidden={h} cond={c} fe=128", mlp_rbf(h, c))
        for h, c in ((832, 416), (864, 432), (880, 440), (896, 448))
    ])
    report("sit (pointnet, cond_channels=19)", [
        (f"hidden={h} depth={d}", sit(h, d))
        for h, d in ((256, 5), (256, 6), (224, 6), (224, 7), (224, 8),
                     (240, 6), (240, 7))
    ])
