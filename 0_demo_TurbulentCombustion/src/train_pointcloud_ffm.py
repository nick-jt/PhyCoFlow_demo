
'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training

- Model backbone can be ConditionalPointMLPRBF, ConditionalPointPerceiver
'''

import argparse
import yaml
import shutil
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datetime import datetime

from helpers import (
    MetricsLogger,
    TurbulentCombustionH5Dataset,
    validate_regular_grid_compatibility,
    create_recon_dir,
    visualize_reconstruction,
    build_sparse_condition,
)
from Model import (
    ConditionalPointFFM, 
    ConditionalPointMLPRBF, 
    ConditionalPointPerceiver,
    ConditionalPointHybridLocalGlobalRBF,
    PointCloudFFM,
    FNO,
    FNOFFM,
    )

def parse_args():

    p = argparse.ArgumentParser("Train a starter conditional point-cloud FFM on turbulent combustion HDF5 data.")

    p.add_argument("--config", type=str, 
                   default="Save_config/config_pointcloud_ffm.yaml", help="Path to YAML config")
    p.add_argument("--Demo-Num", type=int, 
                   default=0, help="Demo ID tag for saving directories")
    p.add_argument("--device-ids", type=int, nargs="+", default=[0])

    p.add_argument("--data", type=str, 
                   default="../Dataset/Merged_CH4COTU1P.h5")
    p.add_argument("--save-dir", type=str, 
                   default=f"Save_TrainedModel/ffm_tc_pointcloud")
    p.add_argument("--field-names", type=str, nargs="+", default=["CH4", "CO", "T", "U_1", "p"])
    p.add_argument("--RELOAD", action="store_true",
                   help="If set, try to reload the latest matching checkpoint and continue training.")


    # ------------------------------
    # Backbone selection
    # ------------------------------
    p.add_argument(
        "--backbone", type=str, default="mlp_rbf", choices = ["mlp_rbf", "perceiver", "fno", "GL_rbf"], 
        help="Backbone type. point-cloud MLP+RBF, point-cloud Perceiver, or grid-based FNO baseline.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    # ------------------------------
    # These are hyperparameters for mlp_rbf backbone or part of GL_rbf
    # ------------------------------
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--field-embed-dim", type=int, default=64)
    p.add_argument("--rbf-sigma", type=float, default=0.05)
    p.add_argument("--USE-FOURIER-PE", "--USE_FOURIER_PE", dest="USE_FOURIER_PE", action="store_true",
                   help="If set, feed Fourier positional coordinate features to point_encoder.")
    p.add_argument("--fourier-pe-num-bands", type=int, default=32,
                   help="Number of frequency bands for Fourier positional coordinate encoding.")
    p.add_argument("--fourier-pe-max-freq", type=float, default=64.0,
                   help="Maximum frequency scale for Fourier positional coordinate encoding.")

    # ------------------------------
    # These are hyperparameters for Perceiver backbone or part of GL_rbf
    # ------------------------------
    p.add_argument("--latent-dim", type=int, default=256, 
                   help="Token / latent width for the Perceiver backbone.",)
    p.add_argument("--num-latents", type=int, default=128, 
                   help="Number of learned latent slots in the Perceiver.",)
    p.add_argument("--num-heads", type=int, default=8, 
                   help="Number of attention heads for Perceiver attention blocks.",)
    p.add_argument("--num-latent-blocks", type=int, default=4, 
                   help="Number of latent self-attention blocks.",)
    p.add_argument("--ff-mult", type=int, default=4, 
                   help="Expansion factor for Transformer feed-forward layers.",)
    p.add_argument("--attn-dropout", type=float, default=0.0, 
                   help="Dropout used inside attention layers.",)
    p.add_argument("--mlp-dropout", type=float, default=0.0, 
                   help="Dropout used inside token projection / FFN layers.",)
    p.add_argument("--decode-chunk-size", type=int, default=4096,
                   help="Chunk size for Perceiver output decoding. Useful for full-resolution reconstruction.",)
    p.add_argument("--share-query-proj", action="store_true",
        help="If set, use the same projection for Perceiver encoder query tokens and decoder query tokens.",)

    p.add_argument("--summary-type", type=str, default='cls',
        help="Only for GL_rbf; select either cls or mean",)

    # ----------------------------------------------------------
    # Hybrid local-global gather options
    # ----------------------------------------------------------
    p.add_argument(
        "--gather-mode", type=str, default="rbf", choices=["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"],
        help="Gather mode used by ConditionalPointHybridLocalGlobalRBF. 'rbf' preserves the current full gather as default.",
    )
    p.add_argument(
        "--gather-topk", type=int, default=32, 
        help="Number of nearest refined sensor tokens used in top-k gather modes.",
    )
    p.add_argument(
        "--gather-query-chunk-size", type=int, default=None,
        help="Optional query chunk size for memory-friendly gathering. Applies to all gather modes.",
    )
    p.add_argument(
        "--learnable-rbf-sigma", action="store_true",
        help="If set, make the RBF sigma in the hybrid gather learnable.",
    )
    p.add_argument(
        "--neighbor-backend", type=str, default="torch", choices=["auto", "torch", "keops"],
        help="Neighbor / kernel backend for the hybrid gather. "
            "'auto' uses KeOps if available, otherwise falls back to pure PyTorch.",)
    p.add_argument(
        "--sensor-local-topk", type=int, default=8,
        help="Number of local sensor neighbors used by the sensor-side Point-Transformer refinement in gather_mode='topk_rbf_ptlocal'.",)
    p.add_argument(
        "--sensor-local-dropout", type=float, default=0.0,
        help="Dropout used inside the sensor-side local refinement block for gather_mode='topk_rbf_ptlocal'.",
    )

    # ----------------------------------------------------------
    # These are hyperparameters for fno backbone
    # Num_x / Num_y must be supplied for the FNO baseline.
    # ----------------------------------------------------------
    p.add_argument( "--Num-x", dest="Num_x", type=int, default=None,
        help="Number of grid points along x for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument("--Num-y", dest="Num_y", type=int, default=None,
        help="Number of grid points along y for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument( "--fno-modes-x", type=int, default=32,
        help="Number of retained Fourier modes along x for the FNO baseline.",)
    p.add_argument( "--fno-modes-y", type=int, default=8,
        help="Number of retained Fourier modes along y for the FNO baseline.",)
    p.add_argument( "--fno-hidden-channels", type=int, default=64,
        help="Hidden channel width of the neuraloperator FNO baseline.",)
    p.add_argument( "--fno-n-layers", type=int, default=4,
        help="Number of Fourier layers in the FNO baseline.",)
    p.add_argument(
        "--condition-blur",
        action="store_true",
        help="If set, Gaussian-splat sparse FNO conditioning maps before concatenation.",
    )
    p.add_argument(
        "--condition-blur-kernel",
        type=int,
        default=5,
        help="Odd Gaussian kernel size used to splat sparse FNO conditioning maps.",
    )
    p.add_argument(
        "--condition-blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used to splat sparse FNO conditioning maps.",
    )

    # ------------------------------
    # These are hyperparameters for training process
    # ------------------------------
    p.add_argument("--n-query-points", type=int, default=4096)
    p.add_argument("--query-sampling", type=str, default="uniform", choices=["uniform", "obs_mix"])
    p.add_argument("--query-sample-near-ratio", type=float, default=0.25)
    p.add_argument("--query-sample-far-ratio", type=float, default=0.25)
    p.add_argument("--query-sample-sigma-ratio", type=float, default=0.05)
    p.add_argument("--prior", type=str, default="rff", choices=["iid", "rff"])
    p.add_argument("--rff-features", type=int, default=256)
    p.add_argument("--rff-lengthscale", type=float, default=0.15)
    p.add_argument("--sigma-min", type=float, default=1e-4) # backward-compatible old args

    p.add_argument("--cond-field", type=int, default=2, help="Legacy single conditioned field.")
    p.add_argument("--n-obs-min", type=int, default=64, help="Legacy single-field minimum sensors.")
    p.add_argument("--n-obs-max", type=int, default=256, help="Legacy single-field maximum sensors.")

    # generalized args
    p.add_argument("--cond-fields", type=int, nargs="+", default=None,
                   help="Conditioned field ids, e.g. --cond-fields 0 2")
    p.add_argument("--n-obs-min-list", type=int, nargs="+", default=None,
                   help="Per-field minimum sensors. Length 1 broadcasts to all cond_fields.")
    p.add_argument("--n-obs-max-list", type=int, nargs="+", default=None,
                   help="Per-field maximum sensors. Length 1 broadcasts to all cond_fields.")

    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Visualization conditioned fields. Defaults to cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Visualization exact sensors per field. Defaults to n_obs_max_list.")
    
    # ODE solver used at generation time. For 1-RF, Euler is the main benchmark because the method is designed for coarse-step sampling.
    p.add_argument(
        "--ode-solver", type=str, default="euler",
            choices=["euler", "heun"], help="ODE solver for generation. Use Euler for the main 1-RF benchmark; Heun is optional.")
    # Reconstruction benchmark step counts. These are the NFEs to compare after moving to 1-RF.
    p.add_argument(
        "--benchmark-n-steps", type=int, nargs="+", default=[2, 4, 8, 16],
            help="Sampling step counts used for reconstruction benchmarking.")

    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--n-steps-generation", type=int, default=32)

    p.add_argument(
        "--no-scale-lr", dest="no_scale_lr", action="store_true",
        help="Deprecated compatibility flag. LR is not scaled automatically unless --scale-lr-by-world-size is set.",
    )
    p.add_argument(
        "--scale-lr-by-world-size", dest="scale_lr_by_world_size", action="store_true",
        help="Opt in to linear lr × world_size scaling when using DDP.",
    )

    return p.parse_args()

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_conditioning_args(args):
    # training
    if args.cond_fields is None:
        args.cond_fields = [args.cond_field]
    if args.n_obs_min_list is None:
        args.n_obs_min_list = [args.n_obs_min]
    if args.n_obs_max_list is None:
        args.n_obs_max_list = [args.n_obs_max]

    # visualization
    if args.vis_cond_fields is None:
        args.vis_cond_fields = list(args.cond_fields)
    if args.vis_n_obs_list is None:
        args.vis_n_obs_list = list(args.n_obs_max_list)

    return args

class IIDGaussianPrior(nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


class RFFGaussianPrior(nn.Module):
    """Scalable smooth Gaussian-field approximation via random Fourier features."""

    def __init__(self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)


def collate_snapshots(batch):
    return {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "fields": torch.stack([b["fields"] for b in batch], dim=0),
        "time_index": torch.stack([b["time_index"] for b in batch], dim=0),
        "physical_time": torch.stack([b["physical_time"] for b in batch], dim=0),
    }


def sample_query_subset(
    coords: torch.Tensor,
    fields: torch.Tensor,
    n_query: Optional[int],
    mode: str = "uniform",
    obs_coords: Optional[torch.Tensor] = None,
    obs_mask: Optional[torch.Tensor] = None,
    near_ratio: float = 0.25,
    far_ratio: float = 0.25,
    sigma_ratio: float = 0.05,
    obs_mix_chunk_size: int = 65536,
):
    if n_query is None or n_query >= coords.shape[1]:
        return coords, fields, None

    bsz, n_pts, coord_dim = coords.shape
    n_query = int(n_query)

    def take_weighted(weights: torch.Tensor, count: int, selected: torch.Tensor) -> torch.Tensor:
        count = min(int(count), int((~selected).sum().item()))
        if count <= 0:
            return torch.empty(0, device=coords.device, dtype=torch.long)

        weights = weights.to(dtype=coords.dtype).clamp_min(0.0)
        weights = weights.masked_fill(selected, 0.0)
        pieces = []

        positive = weights > 0
        if positive.any():
            n_weighted = min(count, int(positive.sum().item()))
            sampled = torch.multinomial(weights, num_samples=n_weighted, replacement=False)
            pieces.append(sampled)
            selected[sampled] = True
            count -= n_weighted

        if count > 0:
            available = (~selected).nonzero(as_tuple=False).squeeze(-1)
            fill = available[torch.randperm(available.numel(), device=coords.device)[:count]]
            pieces.append(fill)
            selected[fill] = True

        return torch.cat(pieces, dim=0) if pieces else torch.empty(0, device=coords.device, dtype=torch.long)

    all_idx = []
    for b in range(bsz):
        if mode == "obs_mix" and obs_coords is not None and obs_mask is not None:
            valid = obs_mask[b].bool()
        else:
            valid = None

        if mode != "obs_mix" or valid is None or not valid.any():
            idx = torch.randperm(n_pts, device=coords.device)[:n_query].sort().values
            all_idx.append(idx)
            continue

        # Compute min-distance from every grid point to the nearest valid
        # sensor in chunks to avoid materialising the full [N_pts × N_obs]
        # matrix, which is infeasible on 3-D grids (~1.95 M points).
        ocoords = obs_coords[b, valid]                    # [n_valid, D]
        chunk = max(1, int(obs_mix_chunk_size))
        d_min_parts: list[torch.Tensor] = []
        for start in range(0, n_pts, chunk):
            end = min(start + chunk, n_pts)
            d_min_parts.append(
                torch.cdist(coords[b, start:end], ocoords, p=2.0).amin(dim=-1)
            )
        d_min = torch.cat(d_min_parts, dim=0)             # [n_pts]
        bbox_diag = (coords[b].amax(dim=0) - coords[b].amin(dim=0)).norm().clamp_min(1e-6)
        sigma = (sigma_ratio * bbox_diag).clamp_min(1e-6)

        near_count = min(n_query, max(0, int(round(n_query * near_ratio))))
        far_count = min(n_query - near_count, max(0, int(round(n_query * far_ratio))))
        uniform_count = n_query - near_count - far_count

        selected = torch.zeros(n_pts, device=coords.device, dtype=torch.bool)
        near_weights = torch.exp(-(d_min ** 2) / (2 * sigma ** 2 + 1e-12))
        far_weights = d_min.clamp_min(0.0)

        pieces = [
            take_weighted(near_weights, near_count, selected),
            take_weighted(far_weights, far_count, selected),
            take_weighted(torch.ones(n_pts, device=coords.device, dtype=coords.dtype), uniform_count, selected),
        ]
        if int(selected.sum().item()) < n_query:
            pieces.append(
                take_weighted(
                    torch.ones(n_pts, device=coords.device, dtype=coords.dtype),
                    n_query - int(selected.sum().item()),
                    selected,
                )
            )

        idx = torch.cat([p for p in pieces if p.numel() > 0], dim=0).sort().values
        all_idx.append(idx)

    idx = torch.stack(all_idx, dim=0)
    coord_idx = idx.unsqueeze(-1).expand(-1, -1, coord_dim)
    field_idx = idx.unsqueeze(-1).expand(-1, -1, fields.shape[-1])
    return torch.gather(coords, dim=1, index=coord_idx), torch.gather(fields, dim=1, index=field_idx), idx

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: Optional[int],
    query_sampling: str = "uniform",
    query_sample_near_ratio: float = 0.25,
    query_sample_far_ratio: float = 0.25,
    query_sample_sigma_ratio: float = 0.05,
    epoch: int = 0,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """Run one training or evaluation epoch.

    Returns:
        (mean_loss, epoch_time_s, peak_gpu_mem_mb)
        epoch_time_s covers only the batch loop — logging and checkpointing
        overhead is excluded. peak_gpu_mem_mb is the CUDA peak since the last
        reset (call torch.cuda.reset_peak_memory_stats() before this function
        if you want a per-epoch peak).
    """
    training = optimizer is not None
    model.train(training)

    if training and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    total = 0.0
    count = 0

    mode_str = "Train" if training else "Eval"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False, disable=not verbose)

    t0 = time.perf_counter()
    for batch in pbar:
        coords_full = batch["coords"].to(device, non_blocking=True)
        fields_full = batch["fields"].to(device, non_blocking=True)

        # Build generalized sparse observations.
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min_list,
            n_obs_max=n_obs_max_list,
        )

        # for models that must operate on the full regular grid like FNO,
        # point subsampling will be disabled.
        _raw = getattr(model, "module", model)
        effective_n_query = None if getattr(_raw, "requires_full_grid", False) else n_query_points
        sampling_mode = query_sampling if training else "uniform"
        coords_q, fields_q, _ = sample_query_subset(
            coords=coords_full,
            fields=fields_full,
            n_query=effective_n_query,
            mode=sampling_mode,
            obs_coords=obs_coords,
            obs_mask=obs_mask,
            near_ratio=query_sample_near_ratio,
            far_ratio=query_sample_far_ratio,
            sigma_ratio=query_sample_sigma_ratio,
        )

        loss, _ = _raw.training_loss(
            x1=fields_q,
            coords=coords_q,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    epoch_time_s = time.perf_counter() - t0

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    mean_loss = total / max(count, 1)
    if dist.is_initialized() and training:
        t = torch.tensor(mean_loss, dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean_loss = t.item()
    return mean_loss, epoch_time_s, peak_mem_mb


def run_reconstruction_benchmark(
    model: nn.Module,
    dataset: Dataset,
    epoch: int,
    device: torch.device,
    recon_dir: str,
    args: argparse.Namespace,
) -> None:
    """Save reconstruction plots/metrics for one epoch at the configured NFEs."""
    recon_dir_epoch = os.path.join(recon_dir, f"Epoch_{epoch}")
    os.makedirs(recon_dir_epoch, exist_ok=True)

    step_list = args.benchmark_n_steps if args.benchmark_n_steps else [args.n_steps_generation]
    for nfe in step_list:
        recon_metrics = visualize_reconstruction(
            model=model,
            dataset=dataset,
            epoch=epoch,
            device=device,
            save_dir=recon_dir_epoch,

            cond_fields=args.vis_cond_fields,
            n_obs=args.vis_n_obs_list,
            n_steps=nfe,
            ode_solver=args.ode_solver,
            snapshot_index=0,
            file_tag=f"{args.ode_solver}_nfe{nfe}",
            save_metrics_json=True,
            field_names=args.field_names,
        )

        metric_str = ", ".join([f"{k}:{v:.4e}" for k, v in recon_metrics.items()])
        print(f"[recon] epoch={epoch:04d} solver={args.ode_solver} n_steps={nfe} | {metric_str}")


def find_latest_run_dir(demo_dir: str, save_dir: str, demo_num: int) -> Optional[Path]:
    save_root = Path(demo_dir) / Path(save_dir).parent
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    if not save_root.exists():
        return None

    candidates = [
        path for path in save_root.glob(f"{run_prefix}*")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def extract_run_timestamp(run_dir: Path, save_dir: str, demo_num: int) -> str:
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    run_name = run_dir.name
    if run_name.startswith(run_prefix):
        return run_name[len(run_prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_path(path: Path, suffix: str = "_bk") -> Path:
    candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}{suffix}{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def backup_existing_artifact(path: Path) -> None:
    if not path.exists():
        return

    target = backup_path(path)
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)

# ---------------------------------------------------------------------------
# DDP helpers — work transparently when called outside of torchrun too.
# ---------------------------------------------------------------------------

def _ddp_active() -> bool:
    return "LOCAL_RANK" in os.environ

def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1

def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0

def _is_main() -> bool:
    return _rank() == 0


def main():

    args = parse_args()

    # ── DDP / single-GPU setup ───────────────────────────────────────────────
    using_ddp = _ddp_active()
    if using_ddp:
        dist.init_process_group(backend="nccl")
        local_rank  = _local_rank()
        world_size  = _world_size()
        device      = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank  = 0
        world_size  = 1
        device      = torch.device(f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu")

    rank0 = _is_main()  # only rank-0 logs, checkpoints, and visualises
    # ────────────────────────────────────────────────────────────────────────

    script_dir = os.path.dirname(os.path.realpath(__file__))
    demo_dir = os.path.dirname(script_dir) # Go up one level to \demo

    # YAML Loading and Backup
    config_path = os.path.join(demo_dir, args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.exists(config_path):
        if rank0:
            print(f"\n[*] Starting:... I found config file at: {config_path}\n")
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Overwrite default args with YAML values
        if yaml_config is not None:
            for key, value in yaml_config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    if rank0:
                        print(f"Warning: YAML key '{key}' is not a recognized argument. Ignoring.")
        args = normalize_conditioning_args(args)

        # Backup the YAML file (rank-0 only)
        if rank0:
            backup_dir = os.path.join(demo_dir, "Save_config", "pointcloud_ffm")
            os.makedirs(backup_dir, exist_ok=True)
            backup_filename = f"config_pointcloud_ffm_DemoN{args.Demo_Num}_{timestamp}.yaml"
            shutil.copy(config_path, os.path.join(backup_dir, backup_filename))
            print(f"[*] Config backed up to: {os.path.join(backup_dir, backup_filename)}\n")
    else:
        if rank0:
            print(f"\n[Warning: !] Config file not found at {config_path}. Using default parameters.\n")
        args.Demo_Num = 0  # Force Demo_Num to 0 as default

    # Keep the configured LR by default so DDP and single-process runs are
    # directly comparable. Linear scaling is available as an explicit opt-in.
    if using_ddp and getattr(args, "scale_lr_by_world_size", False) and not getattr(args, "no_scale_lr", False):
        scaled_lr = args.lr * world_size
        if rank0:
            print(f"[DDP] world_size={world_size} — scaling lr {args.lr:.2e} → {scaled_lr:.2e} "
                  f"(requested by --scale-lr-by-world-size)\n")
        args.lr = scaled_lr
    
    # Setup the Dynamic Directories with Demo_Num
    set_seed(args.seed)

    start_epoch = 1
    best_val = float("inf")
    reload_ckpt = None
    run_timestamp = timestamp
    save_dir = Path(os.path.join(demo_dir, args.save_dir + f"_DemoN{args.Demo_Num}" + f"_{timestamp}"))

    if args.RELOAD:
        latest_run_dir = find_latest_run_dir(demo_dir=demo_dir, save_dir=args.save_dir, demo_num=args.Demo_Num)
        reload_path = None
        if latest_run_dir is not None:
            for ckpt_name in ("last.pt", "best.pt"):
                candidate = latest_run_dir / ckpt_name
                if candidate.exists():
                    reload_path = candidate
                    break

        if reload_path is not None:
            save_dir = latest_run_dir
            run_timestamp = extract_run_timestamp(latest_run_dir, args.save_dir, args.Demo_Num)
            reload_ckpt = torch.load(reload_path, map_location="cpu")
            start_epoch = int(reload_ckpt.get("epoch", 0)) + 1
            best_val = float(reload_ckpt.get("val_loss", float("inf")))

            if rank0:
                backup_existing_artifact(reload_path)
                print(f"[*] RELOAD=True, resuming from: {reload_path}")
                print(f"[*] Resume will start from epoch {start_epoch}\n")
        else:
            if rank0:
                print("[*] RELOAD=True, but no matching last.pt or best.pt was found. Training will start from scratch.\n")

    if rank0:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Wait for rank-0 to finish creating save_dir before others proceed.
    if using_ddp:
        dist.barrier()

    # Setup CSV and Recon Dirs (rank-0 only)
    csv_base_dir = os.path.join(demo_dir, f"Save_loss_csv")
    recon_base_dir = os.path.join(demo_dir, f"Save_reconstruction_files")

    if rank0:
        if args.RELOAD and reload_ckpt is not None:
            loss_dir = Path(csv_base_dir) / f"Loss_DemoN{args.Demo_Num}_{run_timestamp}"
            recon_dir_existing = Path(recon_base_dir) / "ffm_tc_pointcloud" / f"demo_N{args.Demo_Num}_{run_timestamp}"
            backup_existing_artifact(loss_dir)
            backup_existing_artifact(recon_dir_existing)

        logger = MetricsLogger(base_dir=csv_base_dir, Demo_Num=args.Demo_Num, timestamp=run_timestamp)
        recon_dir = create_recon_dir(base_dir=recon_base_dir, Demo_Num=args.Demo_Num, timestamp=run_timestamp)
        print(f"[*] Model checkpoints will save to: {save_dir}")
        print(f"[*] Logging losses to: {logger.save_dir}")
        print(f"[*] Saving recon plots to: {recon_dir}\n")
    else:
        logger = None
        recon_dir = None

    if rank0:
        print(f"Using device: {device}")
        if using_ddp:
            print(f"[DDP] Running with {world_size} GPUs (rank {_rank()})\n")
        else:
            print()

    train_set = TurbulentCombustionH5Dataset(
        args.data,
        split="train",
        train_ratio=args.train_ratio,
        field_names=args.field_names,
        seed=args.seed,
        time_stride=args.time_stride,
        stats_path=str(save_dir / "dataset_stats.pt"),
    )
    val_set = TurbulentCombustionH5Dataset(
        args.data,
        split="val",
        train_ratio=args.train_ratio,
        field_names=args.field_names,
        seed=args.seed,
        time_stride=args.time_stride,
        stats_path=str(save_dir / "dataset_stats.pt"),
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )
    if args.num_workers > 0:
        # Keep workers alive across epochs to reduce epoch-boundary stalls.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if using_ddp:
        # Each rank handles a disjoint shard of the training data.
        # Val is not sharded — all ranks run the full val set so val_loss is
        # consistent without needing an all_reduce.
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_loader  = DataLoader(train_set, sampler=train_sampler, **loader_kwargs)
    else:
        train_sampler = None
        train_loader  = DataLoader(train_set, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    prior = IIDGaussianPrior() if args.prior == "iid" else RFFGaussianPrior(
        coord_dim=3, n_features=args.rff_features, lengthscale=args.rff_lengthscale
    )

    if args.backbone == "mlp_rbf":
        backbone = ConditionalPointMLPRBF(
            n_fields=train_set.num_fields,
            coord_dim=3,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            rbf_sigma=args.rbf_sigma,
            use_fourier_pe=args.USE_FOURIER_PE,
            fourier_pe_num_bands=args.fourier_pe_num_bands,
            fourier_pe_max_freq=args.fourier_pe_max_freq,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "perceiver":
        backbone = ConditionalPointPerceiver(
            n_fields=train_set.num_fields,
            coord_dim=3,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            field_embed_dim=args.field_embed_dim,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            decode_chunk_size=args.decode_chunk_size,
            share_query_proj=args.share_query_proj,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "GL_rbf":
        backbone = ConditionalPointHybridLocalGlobalRBF(
            n_fields=train_set.num_fields,
            coord_dim=3,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            rbf_sigma=args.rbf_sigma,
            summary_type=args.summary_type,

            gather_mode=args.gather_mode,
            gather_topk=args.gather_topk,
            gather_query_chunk_size=args.gather_query_chunk_size,
            learnable_rbf_sigma=args.learnable_rbf_sigma,
            neighbor_backend=args.neighbor_backend,

            sensor_local_topk=args.sensor_local_topk,
            sensor_local_dropout=args.sensor_local_dropout,
            use_fourier_pe=args.USE_FOURIER_PE,
            fourier_pe_num_bands=args.fourier_pe_num_bands,
            fourier_pe_max_freq=args.fourier_pe_max_freq,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "fno":
        # FNO requires an explicit regular-grid interpretation of the dataset.
        try:
            validate_regular_grid_compatibility(train_set, args.Num_x, args.Num_y)
            validate_regular_grid_compatibility(val_set, args.Num_x, args.Num_y)
        except ValueError as e:
            print(f"\n[Warning: !] {e}")
            print("[Warning: !] FNO baseline cannot start because the provided Num_x / Num_y "
                  "are missing or incompatible with the dataset.\n")
            raise SystemExit(1)

        backbone = FNO(
            n_fields=train_set.num_fields,
            Num_x=args.Num_x,
            Num_y=args.Num_y,
            n_modes_x=args.fno_modes_x,
            n_modes_y=args.fno_modes_y,
            hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_n_layers,
            condition_blur=args.condition_blur,
            condition_blur_kernel=args.condition_blur_kernel,
            condition_blur_sigma=args.condition_blur_sigma,
        )
        model = FNOFFM(backbone, prior, sigma_min=args.sigma_min).to(device)

        print(f"[*] Using grid-based FNO baseline with Num_x={args.Num_x}, Num_y={args.Num_y}")
        print("[*] Note: n_query_points is ignored for FNO because it requires the full grid.\n")
    else:
        raise ValueError(
            f'Error!!! Your backbone is not supported: {args.backbone}.'
            'Please select in ["mlp_rbf", "perceiver", "fno"]'
            )
    if rank0:
        print(f'\nSelected Backbone: {args.backbone}\n')
        total_params    = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters:  {total_params:,} total  ({trainable_params:,} trainable)\n")

    # Load checkpoint weights before wrapping in DDP so all ranks start from
    # the same parameters (DDP would broadcast rank-0 anyway, but loading on
    # every rank avoids the broadcast overhead for 20M+ param models).
    if reload_ckpt is not None:
        model.load_state_dict(reload_ckpt["model"])

    # Wrap in DDP after loading weights.
    if using_ddp:
        model = DDP(model, device_ids=[local_rank])

    # raw_model: the underlying nn.Module for checkpointing and visualization.
    raw_model = model.module if using_ddp else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if reload_ckpt is not None:
        if "optimizer" in reload_ckpt:
            optimizer.load_state_dict(reload_ckpt["optimizer"])
        loaded_epoch = int(reload_ckpt.get("epoch", 0))
        if rank0:
            print(f"[*] Reloaded model state from epoch {loaded_epoch}")

        if rank0 and loaded_epoch > 0 and loaded_epoch % args.save_every == 0:
            print(f"[*] Saving reconstruction benchmark for reloaded epoch {loaded_epoch} before continuing.")
            run_reconstruction_benchmark(
                model=raw_model,
                dataset=val_set,
                epoch=loaded_epoch,
                device=device,
                recon_dir=recon_dir,
                args=args,
            )

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr_loss, tr_time, tr_mem = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cond_fields=args.cond_fields,
            n_obs_min_list=args.n_obs_min_list,
            n_obs_max_list=args.n_obs_max_list,
            n_query_points=args.n_query_points,
            query_sampling=args.query_sampling,
            query_sample_near_ratio=args.query_sample_near_ratio,
            query_sample_far_ratio=args.query_sample_far_ratio,
            query_sample_sigma_ratio=args.query_sample_sigma_ratio,
            epoch=epoch,
            verbose=rank0,
        )
        scheduler.step()

        if rank0:
            print(f"[train] epoch={epoch:04d} loss={tr_loss:.6e}  time={tr_time:.1f}s  peak_mem={tr_mem:.0f}MB")
        val_loss = None
        if epoch % args.eval_every == 0 or epoch == 1:
            with torch.no_grad():
                val_loss, _, _ = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    cond_fields=args.cond_fields,
                    n_obs_min_list=args.n_obs_min_list,
                    n_obs_max_list=args.n_obs_max_list,
                    n_query_points=args.n_query_points,
                    query_sampling=args.query_sampling,
                    query_sample_near_ratio=args.query_sample_near_ratio,
                    query_sample_far_ratio=args.query_sample_far_ratio,
                    query_sample_sigma_ratio=args.query_sample_sigma_ratio,
                    epoch=epoch,
                    verbose=rank0,
                )
            if rank0:
                print(f"[valid] epoch={epoch:04d} loss={val_loss:.6e}")

            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "mean": train_set.mean,
                "std": train_set.std,
                "field_names": train_set.field_names,
                "method": "1_rectified_flow",
                "backbone": args.backbone,
                "summary_type": args.summary_type,
                "ode_solver": args.ode_solver,
                "Num_x": args.Num_x,
                "Num_y": args.Num_y,
            }
            if rank0:
                torch.save(ckpt, save_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                if rank0:
                    torch.save(ckpt, save_dir / "best.pt")
                    print('Saving the best model...')
        
        if rank0 and epoch % args.save_every == 0:
            # Benchmark the same validation snapshot at several NFEs.
            run_reconstruction_benchmark(
                model=raw_model,
                dataset=val_set,
                epoch=epoch,
                device=device,
                recon_dir=recon_dir,
                args=args,
            )

        if rank0:
            logger.log_csv(epoch=epoch, train_loss=tr_loss, val_loss=val_loss,
                           epoch_time_s=tr_time, peak_gpu_mem_mb=tr_mem)
            if epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs:
                logger.plot_history()

    if rank0:
        print("Training complete.")
        print(f"Best validation loss: {best_val:.6e}")

    if using_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
