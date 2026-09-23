"""Report CoNFiLD stage-2 prior (UNet) parameter count vs num_channels.

Context (2026-09-02): the prior was held at 1,441,217 parameters in EVERY sweep
arm, because it is a 1-D conv UNet whose size depends on num_channels and
channel_mult but NOT on model_image_size. So as the latent grew 1024 -> 8192,
the prior modelling it never gained capacity. That is a prime suspect for the
canonical result where strict2048 -- best codec of any arm -- produced the WORST
conditional reconstruction (agg 1.316 vs CoNFiLD C's 0.871, pressure 2.818).

This prints the menu of prior sizes so a scaled-prior run can be chosen against
measured numbers rather than a guess. Compute-only, no GPU needed:
    sbatch --partition=mit_normal --time=00:15:00 --mem=24G \
        --wrap="source ~/envs/phycoflow; python check_confild_prior_sizes.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from confild_upstream_training import _import_upstream_diffusion  # noqa: E402

CONFIG = SRC.parent / "Save_config" / "config_baseline_CoNFiLD_xcube_sweep2048.yaml"
CHANNELS = [32, 64, 96, 128, 192, 256]
BASE_PRIOR = 1_441_217          # what every arm has used so far


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    arch = cfg["confild_params"]["stage2"]["architecture"]
    create_model, _ = _import_upstream_diffusion(cfg["confild_params"]["upstream_root"])
    print(f"latent/model_image_size = {arch['model_image_size']}, "
          f"channel_mult = {arch['channel_mult']}, "
          f"num_res_blocks = {arch['num_res_blocks']}\n")
    print(f"{'num_channels':>12} {'prior params':>14} {'x base':>8}")
    for ch in CHANNELS:
        model = create_model(
            image_size=int(arch["model_image_size"]),
            num_channels=ch,
            num_res_blocks=int(arch["num_res_blocks"]),
            num_heads=int(arch["num_heads"]),
            num_head_channels=int(arch["num_head_channels"]),
            attention_resolutions=str(arch["attention_resolutions"]),
            channel_mult=str(arch["channel_mult"]),
        )
        n = sum(p.numel() for p in model.parameters())
        flag = "   <- current" if ch == int(arch["num_channels"]) else ""
        print(f"{ch:>12} {n:>14,} {n / BASE_PRIOR:>7.2f}x{flag}")
        del model
    return 0


if __name__ == "__main__":
    sys.exit(main())
