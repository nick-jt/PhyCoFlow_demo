from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core import (
    DEFAULT_CONFILD_ROOT,
    DEFAULT_DATA,
    FieldStatistics,
    PackedJHUCubes,
    batches,
    config_dict,
    import_upstream_decoder,
    item_to_snapshot_group,
    octahedral_transform,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upstream-faithful CoNFiLD stage-1 training on packed JHU cubes"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--confild-root", type=Path, default=DEFAULT_CONFILD_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=15)
    parser.add_argument("--groups", type=int, choices=(1, 8, 48), default=48)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--points-per-item", type=int, default=65536)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--latent-lr", type=float, default=1e-5)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--diagnostic-every", type=int, default=20)
    parser.add_argument("--diagnostic-items", type=int, default=4)
    parser.add_argument("--diagnostic-points", type=int, default=65536)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--cache-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache the 150 physical training snapshots in host memory (~4.7 GiB)",
    )
    return parser.parse_args()


def load_or_compute_statistics(dataset: PackedJHUCubes, out_dir: Path) -> FieldStatistics:
    path = out_dir / "field_statistics.pt"
    if path.exists():
        return FieldStatistics.from_state_dict(torch.load(path, map_location="cpu"))
    statistics = dataset.compute_train_statistics()
    torch.save(statistics.state_dict(), path)
    return statistics


def diagnostic_reconstruction(
    model: torch.nn.Module,
    latents: torch.Tensor,
    snapshots: list[torch.Tensor] | None,
    dataset: PackedJHUCubes,
    statistics: FieldStatistics,
    groups: int,
    item_count: int,
    point_count: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    errors = []
    with torch.no_grad():
        selected = torch.linspace(0, latents.shape[0] - 1, min(item_count, latents.shape[0])).long()
        point_ids = torch.linspace(
            0, dataset.num_points - 1, min(point_count, dataset.num_points)
        ).long()
        coords = dataset.coords[point_ids].to(device).unsqueeze(0)
        stats_device = statistics.to(device)
        for item in selected.tolist():
            snapshot_index, group_index = item_to_snapshot_group(item, groups)
            physical = snapshots[snapshot_index] if snapshots is not None else dataset.read_snapshot(snapshot_index)
            if group_index:
                physical = octahedral_transform(physical, dataset.grid_shape, group_index)
            point_ids_device = point_ids.to(device)
            truth = stats_device.normalize(
                physical[point_ids].to(device), point_ids_device
            )
            prediction = model(coords, latents[item].view(1, 1, -1))[0]
            errors.append(float(torch.linalg.vector_norm(prediction - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-12)))
    model.train()
    return {"train_rel_l2": float(np.mean(errors)), "train_rel_l2_std": float(np.std(errors))}


def checkpoint_payload(
    args: argparse.Namespace,
    model: torch.nn.Module,
    latents: torch.Tensor,
    decoder_optimizer: torch.optim.Optimizer,
    latent_optimizer: torch.optim.Optimizer,
    epoch: int,
    dataset: PackedJHUCubes,
    statistics: FieldStatistics,
    best_train_rel_l2: float,
) -> dict:
    return {
        "format_version": 1,
        "epoch": epoch,
        "decoder": model.state_dict(),
        "latents": latents.detach().cpu(),
        "decoder_optimizer": decoder_optimizer.state_dict(),
        "latent_optimizer": latent_optimizer.state_dict(),
        "field_statistics": statistics.state_dict(),
        "coord_min": dataset.coord_min,
        "coord_max": dataset.coord_max,
        "grid_shape": dataset.grid_shape,
        "train_indices": torch.as_tensor(dataset.train_indices),
        "val_indices": torch.as_tensor(dataset.val_indices),
        "best_train_rel_l2": best_train_rel_l2,
        "config": config_dict(args),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.out_dir / "config.json", config_dict(args))
    seed_everything(args.seed)
    device = torch.device(args.device)

    dataset = PackedJHUCubes(args.data)
    statistics = load_or_compute_statistics(dataset, args.out_dir)
    snapshots = list(dataset.snapshots(dataset.train_indices)) if args.cache_snapshots else None
    n_items = len(dataset.train_indices) * args.groups
    print(
        f"[stage1] layout={dataset.layout!r} train={len(dataset.train_indices)} "
        f"val={len(dataset.val_indices)} groups={args.groups} items={n_items}",
        flush=True,
    )

    Decoder = import_upstream_decoder(args.confild_root)
    model = Decoder(
        in_coord_features=3,
        in_latent_features=args.latent_dim,
        out_features=dataset.num_fields,
        num_hidden_layers=args.layers,
        hidden_features=args.hidden,
    ).to(device)
    latents = torch.nn.Parameter(torch.zeros(n_items, args.latent_dim, device=device))
    decoder_optimizer = torch.optim.Adam(model.parameters(), lr=args.decoder_lr)
    latent_optimizer = torch.optim.Adam([latents], lr=args.latent_lr)
    start_epoch = 0
    best_train_rel_l2 = math.inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["decoder"])
        latents.data.copy_(checkpoint["latents"].to(device))
        decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
        latent_optimizer.load_state_dict(checkpoint["latent_optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_train_rel_l2 = float(checkpoint.get("best_train_rel_l2", math.inf))

    coords = dataset.coords.to(device)
    stats_device = statistics.to(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + start_epoch)
    history_path = args.out_dir / "history.jsonl"
    started = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        decoder_optimizer.zero_grad(set_to_none=True)
        permutation = torch.randperm(n_items, generator=generator)
        losses = []
        for item_ids_cpu in batches(permutation, args.batch_size):
            point_ids = torch.randint(
                dataset.num_points,
                (item_ids_cpu.numel(), args.points_per_item),
                generator=generator,
            )
            batch_coords = []
            batch_truth = []
            for row, item in enumerate(item_ids_cpu.tolist()):
                snapshot_index, group_index = item_to_snapshot_group(item, args.groups)
                physical = snapshots[snapshot_index] if snapshots is not None else dataset.read_snapshot(snapshot_index)
                if group_index:
                    physical = octahedral_transform(physical, dataset.grid_shape, group_index)
                ids = point_ids[row]
                batch_coords.append(coords[ids.to(device)])
                ids_device = ids.to(device)
                selected = physical[ids].to(device)
                batch_truth.append(stats_device.normalize(selected, ids_device))
            x = torch.stack(batch_coords)
            truth = torch.stack(batch_truth)
            item_ids = item_ids_cpu.to(device)
            prediction = model(x, latents[item_ids].unsqueeze(1))
            loss = F.mse_loss(prediction, truth)
            latent_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            latent_optimizer.step()
            losses.append(float(loss.detach()))

        # This is the defining upstream two-timescale update: one decoder step
        # after a complete balanced pass, while latent codes step per minibatch.
        decoder_optimizer.step()
        decoder_optimizer.zero_grad(set_to_none=True)

        record = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "latent_rms": float(latents.detach().square().mean().sqrt()),
            "decoder_updates": epoch + 1,
            "elapsed_seconds": time.time() - started,
        }
        is_diagnostic = (epoch + 1) % args.diagnostic_every == 0 or epoch == start_epoch
        if is_diagnostic:
            record.update(
                diagnostic_reconstruction(
                    model,
                    latents,
                    snapshots,
                    dataset,
                    statistics,
                    args.groups,
                    args.diagnostic_items,
                    args.diagnostic_points,
                    device,
                )
            )
            if record["train_rel_l2"] < best_train_rel_l2:
                best_train_rel_l2 = record["train_rel_l2"]
                torch.save(
                    checkpoint_payload(
                        args,
                        model,
                        latents,
                        decoder_optimizer,
                        latent_optimizer,
                        epoch,
                        dataset,
                        statistics,
                        best_train_rel_l2,
                    ),
                    args.out_dir / "best.pt",
                )
        with history_path.open("a") as handle:
            import json

            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print("[stage1] " + " ".join(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}" for key, value in record.items()), flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(
                checkpoint_payload(
                    args,
                    model,
                    latents,
                    decoder_optimizer,
                    latent_optimizer,
                    epoch,
                    dataset,
                    statistics,
                    best_train_rel_l2,
                ),
                args.out_dir / "latest.pt",
            )


if __name__ == "__main__":
    main()
