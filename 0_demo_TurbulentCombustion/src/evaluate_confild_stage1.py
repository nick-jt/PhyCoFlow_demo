from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from confild_upstream_core import (
    DEFAULT_CONFILD_ROOT,
    DEFAULT_DATA,
    FieldStatistics,
    PackedJHUCubes,
    decoder_gradient_is_clear,
    import_upstream_decoder,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only held-out CoNFiLD auto-decoding")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--confild-root", type=Path, default=DEFAULT_CONFILD_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--snapshot-indices", type=int, nargs="+", default=[150, 151, 153, 162])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--points-per-step", type=int, default=65536)
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--score-points", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-prior", type=float, default=0.0)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--decode-chunk", type=int, default=262144)
    return parser.parse_args()


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-12))


def fit_latent(
    decoder: torch.nn.Module,
    coords: torch.Tensor,
    truth_normalized: torch.Tensor,
    initial_latent: torch.Tensor,
    fit_indices: torch.Tensor,
    score_indices: torch.Tensor,
    steps: int,
    points_per_step: int,
    lr: float,
    latent_prior: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    latent = torch.nn.Parameter(initial_latent.clone())
    optimizer = torch.optim.Adam([latent], lr=lr)
    best_latent = latent.detach().clone()
    best_score = float("inf")
    for step in range(steps):
        selected = fit_indices[
            torch.randint(fit_indices.numel(), (points_per_step,), generator=generator, device=fit_indices.device)
        ]
        prediction = decoder(coords[selected].unsqueeze(0), latent.unsqueeze(1))[0]
        loss = F.mse_loss(prediction, truth_normalized[selected])
        if latent_prior:
            loss = loss + latent_prior * latent.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                score_prediction = decoder(coords[score_indices].unsqueeze(0), latent.unsqueeze(1))[0]
                score = float(F.mse_loss(score_prediction, truth_normalized[score_indices]))
            if score < best_score:
                best_score = score
                best_latent = latent.detach().clone()
    return best_latent, best_score


def decode_full(
    decoder: torch.nn.Module,
    coords: torch.Tensor,
    latent: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    pieces = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], chunk):
            pieces.append(decoder(coords[start : start + chunk].unsqueeze(0), latent.unsqueeze(1))[0])
    return torch.cat(pieces, dim=0)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    architecture = checkpoint["architecture"]
    statistics = FieldStatistics.from_state_dict(checkpoint["field_statistics"])
    train_indices = checkpoint["train_indices"]
    val_indices = checkpoint["val_indices"]
    dataset = PackedJHUCubes(
        args.data,
        train_count=int(train_indices.numel()),
        val_start=int(val_indices[0]),
        val_count=int(val_indices.numel()),
    )

    Decoder = import_upstream_decoder(args.confild_root)
    decoder = Decoder(
        in_coord_features=3,
        in_latent_features=int(architecture["latent_dim"]),
        out_features=dataset.num_fields,
        num_hidden_layers=int(architecture["layers"]),
        hidden_features=int(architecture["hidden_features"]),
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    coords = dataset.coords.to(device)
    stats_device = statistics.to(device)
    latent_table = checkpoint["latents"].to(device)
    latent_mean = latent_table.mean(dim=0, keepdim=True)
    latent_std = latent_table.std(dim=0, keepdim=True)
    results = {}

    for snapshot_index in args.snapshot_indices:
        if snapshot_index not in dataset.val_indices:
            raise ValueError(f"Snapshot {snapshot_index} is not in the held-out range")
        physical = dataset.read_snapshot(snapshot_index).to(device)
        truth_normalized = stats_device.normalize(physical)
        split_generator = torch.Generator(device=device).manual_seed(args.seed + snapshot_index)
        permutation = torch.randperm(dataset.num_points, generator=split_generator, device=device)
        n_fit = int(dataset.num_points * args.fit_fraction)
        fit_indices = permutation[:n_fit]
        score_indices = permutation[n_fit : n_fit + min(args.score_points, dataset.num_points - n_fit)]

        best_latent = None
        best_score = float("inf")
        for restart in range(args.restarts):
            if restart == 0:
                initial = latent_mean.clone()
            else:
                noise = torch.randn(latent_mean.shape, generator=split_generator, device=device)
                initial = latent_mean + noise * latent_std
            fitted, score = fit_latent(
                decoder,
                coords,
                truth_normalized,
                initial,
                fit_indices,
                score_indices,
                args.steps,
                args.points_per_step,
                args.lr,
                args.latent_prior,
                split_generator,
            )
            if score < best_score:
                best_score = score
                best_latent = fitted
        assert best_latent is not None
        prediction_normalized = decode_full(decoder, coords, best_latent, args.decode_chunk)
        prediction_physical = stats_device.denormalize(prediction_normalized)
        prediction_benchmark = stats_device.benchmark_normalize(prediction_physical)
        truth_benchmark = stats_device.benchmark_normalize(physical)
        aggregate = relative_l2(prediction_benchmark, truth_benchmark)
        per_channel = [
            relative_l2(prediction_benchmark[:, channel], truth_benchmark[:, channel])
            for channel in range(dataset.num_fields)
        ]
        results[str(snapshot_index)] = {
            "rel_l2": aggregate,
            "rel_l2_per_channel": per_channel,
            "latent_fit_score": best_score,
            "latent_rms": float(best_latent.square().mean().sqrt()),
        }
        print(f"[eval-stage1] snapshot={snapshot_index} rel_l2={aggregate:.6f} per_channel={per_channel}", flush=True)

    if not decoder_gradient_is_clear(decoder):
        raise RuntimeError("Frozen decoder accumulated gradients during held-out evaluation")
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "results": results,
        "mean_rel_l2": float(np.mean([item["rel_l2"] for item in results.values()])),
    }
    save_json(args.out_dir / "stage1_auto_decode.json", payload)


if __name__ == "__main__":
    main()
