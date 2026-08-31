from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

from core import DEFAULT_CONFILD_ROOT, build_latent_windows, config_dict, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upstream CoNFiLD latent-image diffusion on JHU stage-1 codes"
    )
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--confild-root", type=Path, default=DEFAULT_CONFILD_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cube-length", type=int, default=50)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--model-image-size", type=int, default=384)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ema", type=float, default=0.9999)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-head-channels", type=int, default=64)
    parser.add_argument("--attention-resolutions", default="32,16,8")
    parser.add_argument("--channel-mult", default="1,1,2,2,4,4")
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def import_upstream_diffusion(confild_root: Path):
    root = str(confild_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from UnconditionalDiffusionTraining_and_Generation.src.script_util import (
        create_gaussian_diffusion,
        create_model,
    )

    return create_model, create_gaussian_diffusion


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    save_json(args.out_dir / "config.json", config_dict(args))
    device = torch.device(args.device)

    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu")
    stage1_config = stage1["config"]
    groups = int(stage1_config["groups"])
    n_snapshots = int(stage1["train_indices"].numel())
    windows, manifest = build_latent_windows(
        stage1["latents"].float(),
        n_snapshots=n_snapshots,
        n_groups=groups,
        cube_length=args.cube_length,
        window_length=args.window_length,
        stride=args.window_stride,
    )
    # Upstream diffusion training uses one scalar min/max over the latent array.
    latent_min = windows.amin().reshape(1)
    latent_max = windows.amax().reshape(1)
    normalized = (
        (windows - latent_min)
        / (latent_max - latent_min).clamp_min(1e-12)
        * 2.0
        - 1.0
    ).contiguous()
    save_json(args.out_dir / "window_manifest.json", {"windows": manifest})
    print(
        f"[stage2] windows={tuple(normalized.shape)} latent_range="
        f"({float(latent_min):.6g}, {float(latent_max):.6g})",
        flush=True,
    )

    create_model, create_diffusion = import_upstream_diffusion(args.confild_root)
    model = create_model(
        image_size=args.model_image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        attention_resolutions=args.attention_resolutions,
        channel_mult=args.channel_mult,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    diffusion = create_diffusion(steps=args.diffusion_steps, noise_schedule="cosine")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1

    generator = torch.Generator(device="cpu").manual_seed(args.seed + start_step)
    started = time.time()
    history = args.out_dir / "history.jsonl"
    for step in range(start_step, args.steps):
        ids = torch.randint(normalized.shape[0], (args.batch_size,), generator=generator)
        batch = normalized[ids].to(device)
        timesteps = torch.randint(
            args.diffusion_steps, (args.batch_size,), generator=generator
        ).to(device)
        loss = diffusion.training_losses(model, batch, timesteps)["loss"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                ema_parameter.mul_(args.ema).add_(parameter, alpha=1.0 - args.ema)

        if step % args.log_every == 0:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "elapsed_seconds": time.time() - started,
            }
            with history.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"[stage2] step={step} loss={record['loss']:.6g}", flush=True)
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save(
                {
                    "format_version": 1,
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "latent_min": latent_min,
                    "latent_max": latent_max,
                    "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
                    "config": config_dict(args),
                    "stage1_config": stage1_config,
                },
                args.out_dir / "latest.pt",
            )


if __name__ == "__main__":
    main()
