"""Stage-B fix 1: retrain CoNFiLD stage 2 with RETAINED periodic checkpoints.

The unified trainer writes best.pt and last.pt from the same save call, so the
completed runs kept no intermediate stage-2 states and validation-based
checkpoint selection was impossible retroactively. This retrain is the same
upstream recipe (same UNet factory, cosine schedule, AdamW, EMA) driven from a
given stage-1 checkpoint, but every --ckpt-every steps it also writes a slim
eval-ready checkpoint (EMA + latent range + architecture, no optimizer) named
ckpt_step*.pt. A TUNE-split DPS eval then selects among them.

Config comes from the stage-1 checkpoint's embedded config
(confild_params.stage2), i.e. the arm's own published-prior settings.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

SRC = ("/home/ntricard/generative_reconstruction/temp/"
       "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from confild_upstream_core import build_latent_windows  # noqa: E402
from confild_upstream_training import (  # noqa: E402
    _append_jsonl,
    _import_upstream_diffusion,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--budget-s", type=float, default=21600.0)
    p.add_argument("--ckpt-every", type=int, default=25000)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Added to the config seed (replicate control).")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("stage-2 retrain is GPU-only")
    torch.cuda.set_device(device)
    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage1 = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    if int(stage1.get("format_version", 0)) < 2 or stage1.get("training_stage") != 1:
        raise RuntimeError(f"{args.stage1_ckpt} is not a unified stage-1 checkpoint")
    cfg = stage1["config"]
    stage = cfg["confild_params"]["stage2"]
    training, architecture = stage["training"], stage["architecture"]
    groups = int(stage1["groups"])
    latent_dimension = int(stage1["latents"].shape[-1])
    if int(architecture["model_image_size"]) != latent_dimension:
        raise ValueError("model_image_size != stage-1 latent dimension")
    channel_mult = tuple(int(v.strip()) for v in str(architecture["channel_mult"]).split(","))
    downsample = 2 ** (len(channel_mult) - 1)
    if latent_dimension % downsample or int(architecture["window_length"]) % downsample:
        raise ValueError("latent/window not divisible by UNet downsampling factor")

    windows, manifest = build_latent_windows(
        stage1["latents"].float(),
        n_snapshots=int(stage1["train_indices"].numel()),
        n_groups=groups,
        cube_length=int(architecture["cube_length"]),
        window_length=int(architecture["window_length"]),
        stride=int(architecture["window_stride"]),
    )
    latent_min, latent_max = windows.amin().reshape(1), windows.amax().reshape(1)
    normalized = ((windows - latent_min) / (latent_max - latent_min).clamp_min(1e-12)
                  * 2.0 - 1.0).contiguous()
    (run_dir / "window_manifest.json").write_text(
        json.dumps({"windows": manifest}, indent=2) + "\n", encoding="utf-8")

    create_model, create_diffusion = _import_upstream_diffusion(
        cfg["confild_params"]["upstream_root"])
    model = create_model(
        image_size=int(architecture["model_image_size"]),
        num_channels=int(architecture["num_channels"]),
        num_res_blocks=int(architecture["num_res_blocks"]),
        num_heads=int(architecture["num_heads"]),
        num_head_channels=int(architecture["num_head_channels"]),
        attention_resolutions=str(architecture["attention_resolutions"]),
        channel_mult=str(architecture["channel_mult"]),
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    diffusion = create_diffusion(steps=int(architecture["diffusion_steps"]),
                                 noise_schedule="cosine")
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(training["learning_rate"]), weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(
        int(cfg["shared"]["seed"]) + int(args.seed_offset))

    total_steps = int(training["steps"])
    started = time.time()
    print(f"[s2ckpts] windows={tuple(normalized.shape)} "
          f"params={sum(q.numel() for q in model.parameters()):,} "
          f"budget_s={args.budget_s} ckpt_every={args.ckpt_every}", flush=True)

    def slim_payload(step):
        return {
            "format_version": 2, "baseline_model": "confild", "training_stage": 2,
            "epoch": step, "step": step, "ema": ema_model.state_dict(),
            "latent_min": latent_min, "latent_max": latent_max,
            "stage1_checkpoint": str(Path(args.stage1_ckpt).resolve()),
            "architecture": copy.deepcopy(architecture),
            "method": "CoNFiLD_upstream_latent_image_diffusion_ckptretain",
        }

    def full_payload(step):
        payload = slim_payload(step)
        payload["model"] = model.state_dict()
        payload["optimizer"] = optimizer.state_dict()
        payload["config"] = copy.deepcopy(cfg)
        return payload

    retained = []
    step = -1
    for step in range(total_steps):
        ids = torch.randint(normalized.shape[0], (int(training["batch_size"]),),
                            generator=generator)
        batch = normalized[ids].to(device)
        timesteps = torch.randint(int(architecture["diffusion_steps"]),
                                  (batch.shape[0],), generator=generator).to(device)
        loss = diffusion.training_losses(model, batch, timesteps)["loss"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(ema_model.parameters(),
                                                model.parameters()):
                ema_parameter.mul_(float(architecture["ema"])).add_(
                    parameter, alpha=1.0 - float(architecture["ema"]))
        if step % int(training["log_every"]) == 0:
            record = {"step": step, "loss": float(loss.detach()),
                      "elapsed_seconds": time.time() - started}
            _append_jsonl(run_dir / "history.jsonl", record)
            print(f"[s2ckpts] step={step} loss={record['loss']:.6g}", flush=True)
        if (step + 1) % int(args.ckpt_every) == 0:
            name = f"ckpt_step{step + 1:07d}.pt"
            torch.save(slim_payload(step), run_dir / name)
            retained.append(name)
            print(f"[s2ckpts] retained {name}", flush=True)
        if (step + 1) % int(training["save_every"]) == 0:
            torch.save(full_payload(step), run_dir / "last.pt")
        if args.budget_s and (time.time() - started) >= args.budget_s:
            print(f"[s2ckpts] budget reached at step {step}", flush=True)
            break

    torch.save(full_payload(step), run_dir / "last.pt")
    final_slim = f"ckpt_step{step + 1:07d}.pt"
    if final_slim not in retained:
        torch.save(slim_payload(step), run_dir / final_slim)
        retained.append(final_slim)
    json.dump({"steps_completed": step + 1, "retained": retained,
               "stage1_ckpt": args.stage1_ckpt,
               "train_seconds": time.time() - started},
              open(run_dir / "s2ckpts_summary.json", "w"), indent=1)
    print(f"[s2ckpts] DONE steps={step + 1} retained={len(retained)}", flush=True)


if __name__ == "__main__":
    main()
