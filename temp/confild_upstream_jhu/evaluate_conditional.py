from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

from core import (
    DEFAULT_CONFILD_ROOT,
    DEFAULT_DATA,
    FieldStatistics,
    PackedJHUCubes,
    config_dict,
    import_upstream_decoder,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upstream-style DPS evaluation on JHU")
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--confild-root", type=Path, default=DEFAULT_CONFILD_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--snapshots", type=int, nargs="+", default=[150, 151, 153, 162, 164, 173, 178, 186]
    )
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--condition-fields", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--observations-per-field", type=int, default=19531)
    parser.add_argument("--dps-scale", type=float, default=1.0)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--target-row", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--decode-chunk", type=int, default=262144)
    return parser.parse_args()


def import_evaluation_dependencies(confild_root: Path, repo_root: Path):
    confild = str(confild_root.resolve())
    baseline_src = str((repo_root / "0_demo_TurbulentCombustion" / "src").resolve())
    for path in (confild, baseline_src):
        if path not in sys.path:
            sys.path.insert(0, path)
    from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import (
        get_conditioning_method,
    )
    from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import (
        create_sampler,
    )
    from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_noise
    from UnconditionalDiffusionTraining_and_Generation.src.script_util import create_model
    from ensemble_eval import ensemble_metrics
    import helpers_baseline

    return (
        create_model,
        create_sampler,
        get_conditioning_method,
        get_noise,
        ensemble_metrics,
        helpers_baseline,
    )


class SensorOperator:
    def __init__(
        self,
        decoder: torch.nn.Module,
        sensor_coords: torch.Tensor,
        sensor_fields: torch.Tensor,
        latent_min: torch.Tensor,
        latent_max: torch.Tensor,
        target_row: int,
    ) -> None:
        self.decoder = decoder
        self.coords = sensor_coords
        self.fields = sensor_fields
        self.latent_min = latent_min
        self.latent_max = latent_max
        self.target_row = target_row

    def denormalize_window(self, normalized: torch.Tensor) -> torch.Tensor:
        return (normalized + 1.0) * 0.5 * (self.latent_max - self.latent_min) + self.latent_min

    def forward(self, normalized_window: torch.Tensor, **kwargs) -> torch.Tensor:
        window = self.denormalize_window(normalized_window[:, 0])
        latent = window[:, self.target_row]
        prediction = self.decoder(
            self.coords.expand(latent.shape[0], -1, -1), latent.unsqueeze(1)
        )
        fields = self.fields.view(1, -1, 1).expand(prediction.shape[0], -1, 1)
        return prediction.gather(2, fields).squeeze(-1)


def decode_target(
    decoder: torch.nn.Module,
    coords: torch.Tensor,
    latent: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    pieces = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], chunk):
            pieces.append(
                decoder(coords[start : start + chunk].unsqueeze(0), latent.unsqueeze(1))[0]
            )
    return torch.cat(pieces)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".matplotlib"))
    os.environ.setdefault("KEOPS_CACHE_FOLDER", str(args.out_dir / ".keops"))
    (args.out_dir / ".matplotlib").mkdir(exist_ok=True)
    (args.out_dir / ".keops").mkdir(exist_ok=True)
    save_json(args.out_dir / "config.json", config_dict(args))
    seed_everything(args.seed)
    device = torch.device(args.device)
    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu")
    stage2 = torch.load(args.stage2_checkpoint, map_location="cpu")
    stage1_config = stage1["config"]
    stage2_config = stage2["config"]
    dataset = PackedJHUCubes(args.data)
    statistics = FieldStatistics.from_state_dict(stage1["field_statistics"]).to(device)

    Decoder = import_upstream_decoder(args.confild_root)
    decoder = Decoder(
        in_coord_features=3,
        in_latent_features=int(stage1_config["latent_dim"]),
        out_features=dataset.num_fields,
        num_hidden_layers=int(stage1_config["layers"]),
        hidden_features=int(stage1_config["hidden"]),
    ).to(device)
    decoder.load_state_dict(stage1["decoder"])
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    (
        create_model,
        create_sampler,
        get_conditioning_method,
        get_noise,
        ensemble_metrics,
        helpers,
    ) = import_evaluation_dependencies(args.confild_root, args.repo_root)
    diffusion_model = create_model(
        image_size=int(stage2_config["model_image_size"]),
        num_channels=int(stage2_config["num_channels"]),
        num_res_blocks=int(stage2_config["num_res_blocks"]),
        num_heads=int(stage2_config["num_heads"]),
        num_head_channels=int(stage2_config["num_head_channels"]),
        attention_resolutions=stage2_config["attention_resolutions"],
        channel_mult=stage2_config["channel_mult"],
    ).to(device)
    diffusion_model.load_state_dict(stage2["ema"])
    diffusion_model.eval()
    sampler = create_sampler(
        sampler="ddpm",
        steps=args.diffusion_steps,
        noise_schedule="cosine",
        model_mean_type="epsilon",
        model_var_type="fixed_large",
        dynamic_threshold=False,
        clip_denoised=True,
        rescale_timesteps=False,
        timestep_respacing="",
    )
    noiser = get_noise(name="gaussian", sigma=0.0)
    coords = dataset.coords.to(device)
    target_row = (
        args.target_row
        if args.target_row >= 0
        else int(stage2_config["window_length"]) // 2
    )
    if target_row >= int(stage2_config["window_length"]):
        raise ValueError("target-row is outside the latent window")
    latent_min = stage2["latent_min"].to(device)
    latent_max = stage2["latent_max"].to(device)
    field_names = ["Ux", "Uy", "Uz", "p"]
    summaries = {}

    for snapshot in args.snapshots:
        if snapshot not in dataset.val_indices:
            raise ValueError(f"Snapshot {snapshot} is not held out")
        physical = dataset.read_snapshot(snapshot).to(device)
        truth_benchmark = statistics.benchmark_normalize(physical)
        torch.manual_seed(args.seed + snapshot)
        _, _, mask, indices, field_ids = helpers.build_sparse_condition(
            coords_full=coords.unsqueeze(0),
            fields_full=truth_benchmark.unsqueeze(0),
            cond_fields=args.condition_fields,
            n_obs_min=[args.observations_per_field],
            n_obs_max=[args.observations_per_field],
        )
        valid = mask[0].bool()
        sensor_indices = indices[0, valid].long()
        sensor_fields = field_ids[0, valid].long()
        sensor_coords = coords[sensor_indices].unsqueeze(0)
        np.savez_compressed(
            args.out_dir / f"sensors_snapshot_{snapshot}.npz",
            point_indices=sensor_indices.cpu().numpy(),
            field_indices=sensor_fields.cpu().numpy(),
        )
        truth_stage1 = statistics.normalize(physical)
        measurement = truth_stage1[sensor_indices, sensor_fields].unsqueeze(0)
        operator = SensorOperator(
            decoder,
            sensor_coords,
            sensor_fields,
            latent_min,
            latent_max,
            target_row,
        )
        conditioner = get_conditioning_method(
            name="ps", operator=operator, noiser=noiser, scale=args.dps_scale
        )
        sample_fn = partial(
            sampler.p_sample_loop,
            model=diffusion_model,
            measurement_cond_fn=partial(conditioner.conditioning),
            record=False,
            save_root=None,
        )

        ensemble = []
        for member in range(args.ensemble_size):
            torch.manual_seed(args.seed + 100003 * snapshot + member)
            initial = torch.randn(
                1,
                1,
                int(stage2_config["window_length"]),
                int(stage1_config["latent_dim"]),
                device=device,
            )
            normalized_window = sample_fn(x_start=initial, measurement=measurement)
            latent_window = operator.denormalize_window(normalized_window.detach()[:, 0])
            prediction_stage1 = decode_target(
                decoder, coords, latent_window[:, target_row], args.decode_chunk
            )
            prediction_physical = statistics.denormalize(prediction_stage1)
            ensemble.append(
                statistics.benchmark_normalize(prediction_physical).cpu().numpy()
            )
        metrics = ensemble_metrics(
            np.stack(ensemble), truth_benchmark.cpu().numpy(), field_names
        )
        metrics["snapshot"] = snapshot
        metrics["hard_clamped"] = False
        summaries[str(snapshot)] = metrics["aggregate"]
        save_json(args.out_dir / f"metrics_snapshot_{snapshot}.json", metrics)
        print(
            f"[conditional] snapshot={snapshot} aggregate={metrics['aggregate']}",
            flush=True,
        )

    save_json(
        args.out_dir / "summary.json",
        {"snapshots": summaries, "hard_clamped": False},
    )


if __name__ == "__main__":
    main()
