from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import h5py
import numpy as np
import torch


DEFAULT_CONFILD_ROOT = Path(
    "/projects/ammoniacomb/generative_reconstruction/baselines/CoNFiLD"
)
DEFAULT_DATA = Path(
    "/projects/ammoniacomb/generative_reconstruction/"
    "jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5"
)

PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_upstream_decoder(confild_root: str | Path = DEFAULT_CONFILD_ROOT):
    root = str(Path(confild_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ConditionalNeuralField.cnf.nf_networks import SIRENAutodecoder_film

    return SIRENAutodecoder_film


def normalize_coords(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lo = coords.amin(dim=0)
    hi = coords.amax(dim=0)
    normalized = (coords - lo) / (hi - lo).clamp_min(1e-12) * 2.0 - 1.0
    return normalized, lo, hi


@dataclass
class FieldStatistics:
    """Upstream field scaling plus benchmark-wide z-score statistics."""

    minimum: torch.Tensor  # [point, channel], reduced over training snapshots
    maximum: torch.Tensor  # [point, channel], reduced over training snapshots
    benchmark_mean: torch.Tensor  # [channel]
    benchmark_std: torch.Tensor  # [channel]

    def normalize(
        self, fields: torch.Tensor, point_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        minimum = self.minimum if point_indices is None else self.minimum[point_indices]
        maximum = self.maximum if point_indices is None else self.maximum[point_indices]
        return (fields - minimum) / (maximum - minimum).clamp_min(1e-12) * 2.0 - 1.0

    def denormalize(
        self, fields: torch.Tensor, point_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        minimum = self.minimum if point_indices is None else self.minimum[point_indices]
        maximum = self.maximum if point_indices is None else self.maximum[point_indices]
        return (fields + 1.0) * 0.5 * (maximum - minimum) + minimum

    def benchmark_normalize(self, fields: torch.Tensor) -> torch.Tensor:
        return (fields - self.benchmark_mean) / self.benchmark_std.clamp_min(1e-12)

    def cpu(self) -> "FieldStatistics":
        return FieldStatistics(
            minimum=self.minimum.cpu(),
            maximum=self.maximum.cpu(),
            benchmark_mean=self.benchmark_mean.cpu(),
            benchmark_std=self.benchmark_std.cpu(),
        )

    def to(self, device: torch.device | str) -> "FieldStatistics":
        return FieldStatistics(
            minimum=self.minimum.to(device),
            maximum=self.maximum.to(device),
            benchmark_mean=self.benchmark_mean.to(device),
            benchmark_std=self.benchmark_std.to(device),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "minimum": self.minimum.cpu(),
            "maximum": self.maximum.cpu(),
            "benchmark_mean": self.benchmark_mean.cpu(),
            "benchmark_std": self.benchmark_std.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "FieldStatistics":
        return cls(**{key: value.float() for key, value in state.items()})


class PackedJHUCubes:
    """Read the packed four-cube HDF5 without reinterpreting its leading axes."""

    def __init__(
        self,
        path: str | Path,
        train_count: int = 150,
        val_start: int = 150,
        val_count: int = 50,
    ) -> None:
        self.path = Path(path)
        self.train_indices = np.arange(train_count, dtype=np.int64)
        self.val_indices = np.arange(val_start, val_start + val_count, dtype=np.int64)
        with h5py.File(self.path, "r") as handle:
            shape = tuple(handle["fields"].shape)
            if len(shape) != 6 or shape[0] != 1 or shape[3:5] != (1, 1):
                raise ValueError(f"Unexpected packed JHU field shape: {shape}")
            self.num_snapshots = int(shape[1])
            self.num_points = int(shape[2])
            self.num_fields = int(shape[-1])
            raw_coords = torch.from_numpy(
                handle["coordinates"][:, 0, 0, :].astype(np.float32)
            )
            self.layout = str(handle.attrs.get("layout", ""))
        if self.val_indices[-1] >= self.num_snapshots:
            raise ValueError("Requested validation range exceeds the HDF5 snapshot count")
        self.coords, self.coord_min, self.coord_max = normalize_coords(raw_coords)
        side = round(self.num_points ** (1.0 / 3.0))
        if side**3 != self.num_points:
            raise ValueError(f"Expected a cubic grid, got {self.num_points} points")
        self.grid_shape = (side, side, side)

    def read_snapshot(self, index: int) -> torch.Tensor:
        with h5py.File(self.path, "r") as handle:
            array = handle["fields"][0, int(index), :, 0, 0, :].astype(np.float32)
        return torch.from_numpy(array)

    def snapshots(self, indices: Sequence[int]) -> Iterator[torch.Tensor]:
        with h5py.File(self.path, "r") as handle:
            fields = handle["fields"]
            for index in indices:
                yield torch.from_numpy(
                    fields[0, int(index), :, 0, 0, :].astype(np.float32)
                )

    def compute_train_statistics(self) -> FieldStatistics:
        minimum = None
        maximum = None
        total = torch.zeros(self.num_fields, dtype=torch.float64)
        total_sq = torch.zeros(self.num_fields, dtype=torch.float64)
        count = 0
        for fields in self.snapshots(self.train_indices):
            # Upstream dim=0 reduces [time, point, channel] over time, not points.
            minimum = fields if minimum is None else torch.minimum(minimum, fields)
            maximum = fields if maximum is None else torch.maximum(maximum, fields)
            total += fields.sum(dim=0, dtype=torch.float64)
            total_sq += fields.double().square().sum(dim=0)
            count += fields.shape[0]
        assert minimum is not None and maximum is not None
        mean = total / count
        variance = (total_sq / count - mean.square()).clamp_min(1e-12)
        return FieldStatistics(
            minimum=minimum.float(),
            maximum=maximum.float(),
            benchmark_mean=mean.float(),
            benchmark_std=variance.sqrt().float(),
        )


def octahedral_transform(
    fields: torch.Tensor,
    grid_shape: Sequence[int],
    group_index: int,
    velocity_channels: Sequence[int] = (0, 1, 2),
) -> torch.Tensor:
    """Apply one of 48 signed axis permutations to a physical vector field."""
    if not 0 <= group_index < 48:
        raise ValueError(f"group_index must be in [0, 48), got {group_index}")
    if len(grid_shape) != 3 or int(np.prod(grid_shape)) != fields.shape[0]:
        raise ValueError("grid_shape does not match flattened field size")
    permutation = PERMUTATIONS[group_index // 8]
    flip_code = group_index % 8
    flip_axes = [axis for axis in range(3) if (flip_code >> axis) & 1]
    channels = fields.shape[-1]

    transformed = fields.reshape(*grid_shape, channels).permute(
        permutation[0], permutation[1], permutation[2], 3
    )
    component_order = list(range(channels))
    velocity_channels = list(velocity_channels)
    for axis in range(3):
        component_order[velocity_channels[axis]] = velocity_channels[permutation[axis]]
    transformed = transformed[..., component_order]
    if flip_axes:
        transformed = torch.flip(transformed, dims=flip_axes)
        signs = torch.ones(channels, dtype=transformed.dtype, device=transformed.device)
        for axis in flip_axes:
            signs[velocity_channels[axis]] = -1.0
        transformed = transformed * signs
    return transformed.reshape(-1, channels).contiguous()


def item_to_snapshot_group(item_id: int, n_group: int) -> tuple[int, int]:
    if n_group not in (1, 8, 48):
        raise ValueError("n_group must be one of 1, 8, or 48")
    return divmod(int(item_id), n_group)


def batches(permutation: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, permutation.numel(), batch_size):
        yield permutation[start : start + batch_size]


def decoder_gradient_is_clear(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in model.parameters())


def build_latent_windows(
    latents: torch.Tensor,
    n_snapshots: int,
    n_groups: int,
    cube_length: int,
    window_length: int,
    stride: int = 1,
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """Arrange lumped codes as upstream [time, latent] diffusion images."""
    if latents.shape[0] != n_snapshots * n_groups:
        raise ValueError("Latent table shape does not match snapshots x groups")
    if n_snapshots % cube_length:
        raise ValueError("n_snapshots must be divisible by cube_length")
    if not 1 <= window_length <= cube_length:
        raise ValueError("window_length must be between 1 and cube_length")
    table = latents.reshape(n_snapshots, n_groups, latents.shape[-1])
    windows = []
    manifest = []
    for cube in range(n_snapshots // cube_length):
        cube_start = cube * cube_length
        for group in range(n_groups):
            for offset in range(0, cube_length - window_length + 1, stride):
                windows.append(
                    table[
                        cube_start + offset : cube_start + offset + window_length,
                        group,
                    ]
                )
                manifest.append(
                    {
                        "cube": cube,
                        "group": group,
                        "start_snapshot": cube_start + offset,
                        "window_length": window_length,
                    }
                )
    return torch.stack(windows).unsqueeze(1), manifest


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def config_dict(namespace) -> dict:
    result = vars(namespace).copy()
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
    return result
