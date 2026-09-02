"""Pre-launch smoke test for the CoNFiLD stage-1 latent-dim sweep (Engaging).

Instantiates, for each sweep config, exactly what the trainer will build --
the upstream SIRENAutodecoder_film decoder (confild_upstream_training.py:455-461),
the 7200 x D latent table (:463), and the stage-2 diffusion prior via upstream
create_model (:738-746) -- and asserts parameter counts to the digit, forward
tensor shapes, and the architecture invariants (model_image_size == latent_dim,
divisibility by the UNet downsample factor, attention_ds identical across arms
so the deepest attention level is preserved at every latent size).

Run under srun/sbatch (mit_normal CPU is fine; login-node rule: torch alone
exceeds the RAM courtesy limit):
    srun --partition=mit_normal --time=00:15:00 --mem=16G --cpus-per-task=4 \
        python check_confild_sweep_params.py
Exits nonzero on any mismatch. Training must not launch until this passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from confild_upstream_core import import_upstream_decoder  # noqa: E402
from confild_upstream_training import _import_upstream_diffusion  # noqa: E402

CONFIG_DIR = SRC.parent / "Save_config"

# arm -> (config file, expected decoder params, expected latent-table values)
ARMS = {
    "sweep1024": ("config_baseline_CoNFiLD_xcube_sweep1024.yaml", 5_183_236, 7_372_800),
    "sweep2048": ("config_baseline_CoNFiLD_xcube_sweep2048.yaml", 9_377_540, 14_745_600),
    "sweep4096": ("config_baseline_CoNFiLD_xcube_sweep4096.yaml", 17_766_148, 29_491_200),
    "sweep8192": ("config_baseline_CoNFiLD_xcube_sweep8192.yaml", 34_543_364, 58_982_400),
    "strict2048": ("config_baseline_CoNFiLD_xcube_strict2048.yaml", 5_032_948, 14_745_600),
}
EXPECTED_PRIOR = 1_441_217          # identical across arms by construction
EXPECTED_ATTENTION_DS = [32, 64, 128]
COMPARISON_MODEL = 6_506_253
STRICT_TOTAL = 6_474_165            # decoder 5,032,948 + prior 1,441,217 (~6.47M)
N_ITEMS = 7200                      # int(200 * 0.75) train snapshots x 48 groups
OUT_FIELDS = 4                      # Ux, Uy, Uz, p


def closed_form_decoder(hidden: int, layers: int, latent: int, coord: int = 3, out: int = OUT_FIELDS) -> int:
    # first Linear(3,H) + L hidden Linear(H,H) + final Linear(H,out)
    # + (L+1) bias-free FiLM Linear(D,H)
    return (coord * hidden + hidden) + layers * (hidden * hidden + hidden) \
        + (hidden * out + out) + (layers + 1) * (latent * hidden)


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, expected {want}")
    return ok


def main() -> int:
    failures = 0
    rows = []
    for arm, (config_name, want_decoder, want_table) in ARMS.items():
        cfg = yaml.safe_load((CONFIG_DIR / config_name).read_text())
        arch = cfg["confild_params"]["stage1"]["architecture"]
        s2 = cfg["confild_params"]["stage2"]["architecture"]
        hidden = int(arch["hidden_features"])
        latent = int(arch["latent_dim"])
        layers = int(arch["layers"])
        print(f"\n=== {arm} (H={hidden}, D={latent}, L={layers}) ===")

        # -- invariants the trainer enforces (fail here, not 4.5h into a job) --
        ok = check("model_image_size == latent_dim", int(s2["model_image_size"]), latent)
        channel_mult = tuple(int(v) for v in str(s2["channel_mult"]).split(","))
        downsample = 2 ** (len(channel_mult) - 1)
        ok &= check(f"latent_dim % {downsample}", latent % downsample, 0)
        ok &= check(f"window_length % {downsample}", int(s2["window_length"]) % downsample, 0)
        attention_ds = [latent // int(r) for r in str(s2["attention_resolutions"]).split(",")]
        ok &= check("attention_ds (deepest level preserved)", attention_ds, EXPECTED_ATTENTION_DS)
        ok &= check("tie_latent_to_hidden", bool(arch["tie_latent_to_hidden"]), False)

        # -- decoder: exact construction from confild_upstream_training.py:455-461 --
        Decoder = import_upstream_decoder(cfg["confild_params"]["upstream_root"])
        decoder = Decoder(
            in_coord_features=3,
            in_latent_features=latent,
            out_features=OUT_FIELDS,
            num_hidden_layers=layers,
            hidden_features=hidden,
        )
        n_decoder = sum(p.numel() for p in decoder.parameters())
        ok &= check("decoder parameters", n_decoder, want_decoder)
        ok &= check("closed-form formula agrees", closed_form_decoder(hidden, layers, latent), n_decoder)

        # -- latent table (training-only): confild_upstream_training.py:463 --
        table = torch.nn.Parameter(torch.zeros(N_ITEMS, latent))
        ok &= check("latent-table values (7200 x D)", table.numel(), want_table)

        # -- decoder forward shapes, as the trainer calls it (:561-564) --
        coords = torch.rand(2, 4096, 3) * 2.0 - 1.0
        z = torch.zeros(2, 1, latent)
        with torch.no_grad():
            out = decoder(coords, z)
        ok &= check("decoder forward [2,4096,3]x[2,1,D] -> [2,4096,4]",
                    tuple(out.shape), (2, 4096, OUT_FIELDS))

        # -- diffusion prior: exact construction from confild_upstream_training.py:738-746 --
        create_model, _ = _import_upstream_diffusion(cfg["confild_params"]["upstream_root"])
        prior = create_model(
            image_size=int(s2["model_image_size"]),
            num_channels=int(s2["num_channels"]),
            num_res_blocks=int(s2["num_res_blocks"]),
            num_heads=int(s2["num_heads"]),
            num_head_channels=int(s2["num_head_channels"]),
            attention_resolutions=str(s2["attention_resolutions"]),
            channel_mult=str(s2["channel_mult"]),
        )
        n_prior = sum(p.numel() for p in prior.parameters())
        ok &= check("prior parameters (identical across arms)", n_prior, EXPECTED_PRIOR)

        # -- prior forward: 1-channel 2-D image [B,1,window,D] (dims=2, in_channels=1,
        #    learn_sigma False -> same shape out) --
        x = torch.randn(2, 1, int(s2["window_length"]), latent)
        t = torch.randint(0, int(s2["diffusion_steps"]), (2,))
        with torch.no_grad():
            y = prior(x, t)
        ok &= check("prior forward [2,1,32,D] -> same shape", tuple(y.shape), tuple(x.shape))

        total = n_decoder + n_prior
        delta = 100.0 * (total - COMPARISON_MODEL) / COMPARISON_MODEL
        rows.append((arm, hidden, latent, n_decoder, table.numel(), n_prior, total, delta))
        if not ok:
            failures += 1
        del decoder, prior, table

    # -- strict arm must land on the ~6.47M budget point --
    strict_total = next(r[6] for r in rows if r[0] == "strict2048")
    print()
    if not check("strict2048 inference total (decoder + C prior)", strict_total, STRICT_TOTAL):
        failures += 1
    tol = float(yaml.safe_load(
        (CONFIG_DIR / ARMS["strict2048"][0]).read_text()
    )["confild_params"]["parameter_budget"]["relative_tolerance"])
    within = abs(strict_total - COMPARISON_MODEL) <= tol * COMPARISON_MODEL
    print(f"  {'PASS' if within else 'FAIL'}  strict2048 within +/-{tol:.0%} of "
          f"{COMPARISON_MODEL:,}: {100.0 * (strict_total - COMPARISON_MODEL) / COMPARISON_MODEL:+.2f}%")
    failures += 0 if within else 1

    print(f"\n{'arm':<11} {'H':>4} {'D':>5} {'decoder':>12} {'latent table':>13} "
          f"{'prior':>10} {'infer total':>12} {'dvsbudget':>10}")
    for arm, h, d, dec, tab, pri, tot, delta in rows:
        print(f"{arm:<11} {h:>4} {d:>5} {dec:>12,} {tab:>13,} {pri:>10,} {tot:>12,} {delta:>+9.2f}%")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} ARM(S) FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
