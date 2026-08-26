"""CoNFiLD baseline, stage 1: Conditional Neural Field auto-decoding on JHU.

Faithful adaptation of du2024confild (repo ConditionalNeuralField, case-4 3D
recipe) to the cross-cube protocol. The decoder is their SIRENAutodecoder_film
imported verbatim from the cloned repo; per-snapshot lumped latents (dim =
hidden_size = 384, zeros-init) are auto-decoded jointly with the network,
Adam lrs from their case4.yml (nf 1e-4, latents 1e-5), fields normalized to
[-1, 1] per channel as their '-11' normalizer does.

Two deliberate adaptations, both documented in the paper appendix:
  * Coordinate subsampling per step (their trainer feeds the full field per
    item; at 1.95M points that does not fit, and the SIREN loss is pointwise,
    so a uniform subsample is an unbiased estimator of the same objective).
  * Deterministic octahedral expansion: the group element index g in [0, 48)
    is part of the item index (item = snap * 48 + g), because a fixed latent
    table cannot absorb stochastic augmentation. Every method in the paper
    sees the same 48-element octahedral family; here it also multiplies the
    stage-2 latent-diffusion training set from 150 to 7200 latents.

Stage 2 (latent diffusion) and conditional generation (sensor-guided DPS
through the frozen decoder) live in confild_stage2.py / confild_conditional.py.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CONFILD_ROOT = "/projects/ammoniacomb/generative_reconstruction/baselines/CoNFiLD"
sys.path.insert(0, CONFILD_ROOT)

from ConditionalNeuralField.cnf.nf_networks import SIRENAutodecoder_film  # noqa: E402

import helpers_baseline as HB  # noqa: E402

_PERMS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def octahedral_apply(fields: torch.Tensor, grid_shape, g: int, vel_idx=(0, 1, 2)):
    """Deterministic element g in [0, 48) of the octahedral group (perm*8+flips)."""
    gx, gy, gz = (int(s) for s in grid_shape)
    n_f = fields.shape[-1]
    perm = _PERMS[g // 8]
    fl = g % 8
    flips = [(fl >> a) & 1 for a in range(3)]
    x = fields.reshape(gx, gy, gz, n_f).permute(perm[0], perm[1], perm[2], 3)
    comp = list(range(n_f))
    vel = list(vel_idx)
    for a in range(3):
        comp[vel[a]] = vel[perm[a]]
    x = x[..., comp]
    dims = [a for a in range(3) if flips[a]]
    if dims:
        x = torch.flip(x, dims=dims)
        sign = torch.ones(n_f, dtype=x.dtype)
        for a in dims:
            sign[vel[a]] = -1.0
        x = x * sign
    return x.reshape(-1, n_f).contiguous()


def build_dataset(args, split):
    return HB.TurbulentCombustionH5Dataset(
        args.data,
        split=split,
        train_ratio=args.train_ratio,
        seed=42,
        field_names=("Ux", "Uy", "Uz", "p"),
        stats_path=str(Path(args.out_dir) / "dataset_stats.pt"),
    )


def field_minmax(dataset, device):
    """Per-field min/max over the split, streamed (their '-11' normalizer)."""
    lo = None
    hi = None
    for i in range(len(dataset)):
        f = dataset[i]["fields"]
        flo = f.amin(dim=0)
        fhi = f.amax(dim=0)
        lo = flo if lo is None else torch.minimum(lo, flo)
        hi = fhi if hi is None else torch.maximum(hi, fhi)
    return lo.to(device), hi.to(device)


def to_pm1(x, lo, hi):
    return (x - lo) / (hi - lo).clamp_min(1e-12) * 2.0 - 1.0


def from_pm1(x, lo, hi):
    return (x + 1.0) * 0.5 * (hi - lo) + lo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/projects/ammoniacomb/generative_reconstruction/"
                   "jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--hidden", type=int, default=384)      # = latent dim (their recipe)
    p.add_argument("--layers", type=int, default=15)
    p.add_argument("--n-group", type=int, default=48)      # octahedral expansion; 1 = off
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--points", type=int, default=65536)
    p.add_argument("--lr-nf", type=float, default=1e-4)
    p.add_argument("--lr-lat", type=float, default=1e-5)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--tta-every", type=int, default=20000)
    p.add_argument("--tta-steps", type=int, default=600)
    p.add_argument("--tta-lr", type=float, default=1e-2)
    p.add_argument("--resume", default="")
    args = p.parse_args()

    device = "cuda:0"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JHU_SPLIT_MODE", "block")
    os.environ.setdefault("JHU_SPLIT_GAP", "0")
    # Augmentation is handled deterministically here, never in the dataset.
    os.environ["JHU_AUGMENT"] = ""

    train_set = build_dataset(args, "train")
    val_set = build_dataset(args, "val")
    n_snap = len(train_set)
    n_items = n_snap * args.n_group
    side = round(train_set.num_points ** (1.0 / 3.0))
    grid_shape = (side, side, side)
    print(f"[confild] train snaps {n_snap} x group {args.n_group} = {n_items} items; "
          f"grid {grid_shape}; val snaps {len(val_set)}", flush=True)

    lo, hi = field_minmax(train_set, device)
    n_fields = int(train_set.num_fields)

    coords_full = train_set[0]["coords"].to(device) * 2.0 - 1.0  # [-1, 1]^3

    net = SIRENAutodecoder_film(
        in_coord_features=3, in_latent_features=args.hidden,
        out_features=n_fields, num_hidden_layers=args.layers,
        hidden_features=args.hidden,
    ).to(device)
    latents = torch.nn.Parameter(torch.zeros(n_items, args.hidden, device=device))
    n_params = sum(p_.numel() for p_ in net.parameters())
    print(f"[confild] decoder params {n_params:,}; latent table {tuple(latents.shape)}",
          flush=True)

    opt_net = torch.optim.Adam(net.parameters(), lr=args.lr_nf)
    opt_lat = torch.optim.Adam([latents], lr=args.lr_lat)

    start = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["net"])
        with torch.no_grad():
            latents.copy_(ck["latents"])
        opt_net.load_state_dict(ck["opt_net"])
        opt_lat.load_state_dict(ck["opt_lat"])
        start = int(ck["step"]) + 1
        print(f"[confild] resumed from {args.resume} at step {start}", flush=True)

    # Cache normalized snapshots on CPU once (150 x 1.95M x 4 floats ~ 4.7 GB).
    snaps = [to_pm1(train_set[i]["fields"], lo.cpu(), hi.cpu()) for i in range(n_snap)]

    rng = np.random.default_rng(1234 + start)
    t0 = time.time()
    for step in range(start, args.steps):
        ids = rng.integers(0, n_items, size=args.batch)
        pts = torch.from_numpy(
            rng.integers(0, train_set.num_points, size=(args.batch, args.points)))
        xb, yb = [], []
        for b, item in enumerate(ids):
            s, g = divmod(int(item), args.n_group)
            f = snaps[s]
            if args.n_group > 1 and g > 0:
                f = octahedral_apply(f, grid_shape, g)
            yb.append(f[pts[b]])
            xb.append(coords_full[pts[b].to(device)])
        x = torch.stack(xb)                       # [B, P, 3]
        y = torch.stack(yb).to(device)            # [B, P, F]
        z = latents[torch.as_tensor(ids, device=device)].unsqueeze(1)  # [B, 1, H]

        pred = net(x, z)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt_net.zero_grad(set_to_none=True)
        opt_lat.zero_grad(set_to_none=True)
        loss.backward()
        opt_net.step()
        opt_lat.step()

        if step % 100 == 0:
            lat_rms = float(latents.detach().pow(2).mean().sqrt())
            print(f"[confild] step {step} loss {float(loss):.6e} lat_rms {lat_rms:.4e} "
                  f"({(time.time()-t0):.1f}s)", flush=True)
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save({"net": net.state_dict(), "latents": latents.detach(),
                        "opt_net": opt_net.state_dict(), "opt_lat": opt_lat.state_dict(),
                        "step": step, "lo": lo.cpu(), "hi": hi.cpu(),
                        "hidden": args.hidden, "layers": args.layers,
                        "n_group": args.n_group, "n_snap": n_snap},
                       out / "cnf_last.pt")
        if (step + 1) % args.tta_every == 0 or step == args.steps - 1:
            rel = tta_eval(net, val_set, coords_full, lo, hi, args, device)
            print(f"[confild] TTA val relL2 (decoder bound) @step {step}: {rel:.4f}",
                  flush=True)

    print("[confild] stage 1 done", flush=True)


def tta_eval(net, val_set, coords_full, lo, hi, args, device, n_val=2):
    """Auto-decode held-out snapshots with the decoder frozen: the CNF
    reconstruction floor, the latent method's analog of the AE bound."""
    net.eval()
    rels = []
    for i in range(min(n_val, len(val_set))):
        truth = to_pm1(val_set[i]["fields"].to(device), lo, hi)
        z = torch.nn.Parameter(torch.zeros(1, 1, args.hidden, device=device))
        opt = torch.optim.Adam([z], lr=args.tta_lr)
        rng = np.random.default_rng(9)
        for _ in range(args.tta_steps):
            pts = torch.from_numpy(
                rng.integers(0, truth.shape[0], size=args.points)).to(device)
            pred = net(coords_full[pts].unsqueeze(0), z)
            loss = torch.nn.functional.mse_loss(pred, truth[pts].unsqueeze(0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = []
            for s in range(0, truth.shape[0], 262144):
                preds.append(net(coords_full[s:s+262144].unsqueeze(0), z)[0])
            pred = torch.cat(preds, dim=0)
            rel = float(torch.linalg.norm(pred - truth) / torch.linalg.norm(truth))
        rels.append(rel)
    net.train()
    return float(np.mean(rels))


if __name__ == "__main__":
    main()
