"""CoNFiLD baseline, stage 2: unconditional diffusion over CNF latents.

Uses the openai-style GaussianDiffusion from the CoNFiLD repo verbatim
(cosine schedule, 1000 steps, epsilon prediction, MSE loss, EMA 0.9999 --
their unconditional training recipe). The only adapted component is the
denoiser network: their 2D UNet operates on [t_window x latent] latent
"images", which does not exist for single-snapshot lumped latents; a
time-embedded MLP-ResNet over the 384-d latent vector replaces it
(documented in the appendix). Latents are normalized per-dimension to
[-1, 1] (their '-11' convention), which the guided sampler's
clip_denoised=True assumes.
"""

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CONFILD_ROOT = "/projects/ammoniacomb/generative_reconstruction/baselines/CoNFiLD"
sys.path.insert(0, CONFILD_ROOT)
sys.path.insert(0, CONFILD_ROOT + "/UnconditionalDiffusionTraining_and_Generation")

from src.gaussian_diffusion import (  # noqa: E402
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType,
    get_named_beta_schedule,
)


class TimeEmbedMLP(nn.Module):
    """Epsilon-predictor over lumped latent vectors, openai (x, t) signature."""

    def __init__(self, dim=384, hidden=1024, blocks=6, t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden))
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden),
                          nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(blocks)
        ])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, dim))

    def _t_embed(self, t):
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None]
        return torch.cat([ang.cos(), ang.sin()], dim=-1)

    def forward(self, x, t):
        squeeze = x.dim() > 2
        shp = x.shape
        if squeeze:
            x = x.reshape(shp[0], -1)
        h = self.inp(x)
        te = self.t_mlp(self._t_embed(t))
        for blk in self.blocks:
            h = h + blk(h + te)
        out = self.out(h)
        return out.reshape(shp) if squeeze else out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnf-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--save-every", type=int, default=20000)
    args = p.parse_args()

    device = "cuda:0"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.cnf_ckpt, map_location="cpu")
    lat = ck["latents"].float()                       # [n_items, D]
    lo = lat.amin(dim=0)
    hi = lat.amax(dim=0)
    span = (hi - lo).clamp_min(1e-8)
    lat_n = ((lat - lo) / span * 2.0 - 1.0).to(device)
    n, dim = lat_n.shape
    print(f"[stage2] {n} latents dim {dim}; per-dim '-11' normalized", flush=True)

    diffusion = GaussianDiffusion(
        betas=get_named_beta_schedule("cosine", 1000),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
    )
    net = TimeEmbedMLP(dim=dim, hidden=args.hidden, blocks=args.blocks).to(device)
    ema_net = copy.deepcopy(net).eval()
    for q in ema_net.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    n_par = sum(q.numel() for q in net.parameters())
    print(f"[stage2] denoiser params {n_par:,}", flush=True)

    rng = np.random.default_rng(7)
    t0 = time.time()
    for step in range(args.steps):
        ids = torch.as_tensor(rng.integers(0, n, size=args.batch), device=device)
        x0 = lat_n[ids]
        t = torch.as_tensor(rng.integers(0, diffusion.num_timesteps, size=args.batch),
                            device=device)
        losses = diffusion.training_losses(net, x0, t)
        loss = losses["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for q, s in zip(ema_net.parameters(), net.parameters()):
                q.mul_(args.ema).add_(s, alpha=1.0 - args.ema)
        if step % 500 == 0:
            print(f"[stage2] step {step} loss {float(loss):.5e} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save({"net": net.state_dict(), "ema": ema_net.state_dict(),
                        "lat_lo": lo, "lat_hi": hi, "dim": dim,
                        "hidden": args.hidden, "blocks": args.blocks,
                        "step": step},
                       out / "diff_last.pt")
    print("[stage2] done", flush=True)


if __name__ == "__main__":
    main()
