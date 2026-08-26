"""CoNFiLD baseline, stage 3: sensor-conditioned generation + ensemble scoring.

Guided sampling verbatim from the CoNFiLD repo's ConditionalDiffusionGeneration
package: DDPM sampler (cosine, 1000 steps, epsilon, fixed_large,
clip_denoised) with 'ps' (DPS) conditioning at their default scale 1.0. The
measurement operator decodes the candidate latent at the sensor coordinates
through the frozen stage-1 SIREN decoder, exactly their Case-4 construction.
Scored with the identical fair-CRPS/coverage estimator used for every other
method (ensemble_eval.ensemble_metrics), on the matched val snapshots and
sensor protocol (fields 0,2 at 1% each).
"""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

CONFILD_ROOT = "/projects/ammoniacomb/generative_reconstruction/baselines/CoNFiLD"
sys.path.insert(0, CONFILD_ROOT)

from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import (  # noqa: E402
    get_conditioning_method,
)
from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_noise  # noqa: E402
from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import (  # noqa: E402
    create_sampler,
)
from ConditionalNeuralField.cnf.nf_networks import SIRENAutodecoder_film  # noqa: E402

import helpers_baseline as HB  # noqa: E402
from confild_baseline import build_dataset, to_pm1, from_pm1  # noqa: E402
from confild_stage2 import TimeEmbedMLP  # noqa: E402
from ensemble_eval import ensemble_metrics  # noqa: E402


class CNFSensorOperator:
    """Their Case4-style nonlinear operator: latent -> decoded sensor values."""

    def __init__(self, decoder, sensor_coords, lat_lo, lat_hi):
        self.decoder = decoder
        self.coords = sensor_coords          # [1, M, 3] in [-1, 1]
        self.lo = lat_lo
        self.hi = lat_hi

    def _unnorm(self, z_n):
        return (z_n + 1.0) * 0.5 * (self.hi - self.lo) + self.lo

    def forward(self, z_n, **kwargs):
        z = self._unnorm(z_n.reshape(z_n.shape[0], -1))
        pred = self.decoder(self.coords.expand(z.shape[0], -1, -1), z.unsqueeze(1))
        return pred                           # [B, M, F] in [-1, 1] field units

    def project(self, data, measurement, **kwargs):
        raise NotImplementedError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnf-ckpt", required=True)
    p.add_argument("--diff-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data", default="/projects/ammoniacomb/generative_reconstruction/"
                   "jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5")
    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--snapshots", type=int, nargs="+",
                   default=[0, 1, 3, 12, 14, 23, 28, 36])
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--cond-fields", type=int, nargs="+", default=[0, 2])
    p.add_argument("--n-obs", type=int, default=19531)
    p.add_argument("--dps-scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=262144)
    args = p.parse_args()

    device = "cuda:0"
    out = Path(args.out_dir)
    (out / "Evaluation").mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.cnf_ckpt, map_location="cpu")
    dk = torch.load(args.diff_ckpt, map_location="cpu")
    lo_f, hi_f = ck["lo"].to(device), ck["hi"].to(device)

    dataset = build_dataset(args, "val")
    n_fields = int(dataset.num_fields)
    coords_full = dataset[0]["coords"].to(device) * 2.0 - 1.0

    decoder = SIRENAutodecoder_film(
        in_coord_features=3, in_latent_features=int(ck.get("latent_dim", ck["hidden"])),
        out_features=n_fields, num_hidden_layers=int(ck["layers"]),
        hidden_features=int(ck["hidden"]),
    ).to(device)
    decoder.load_state_dict(ck["net"])
    decoder.eval()
    for q in decoder.parameters():
        q.requires_grad_(False)

    denoiser = TimeEmbedMLP(dim=int(dk["dim"]), hidden=int(dk["hidden"]),
                            blocks=int(dk["blocks"])).to(device)
    denoiser.load_state_dict(dk["ema"])
    denoiser.eval()
    lat_lo = dk["lat_lo"].to(device)
    lat_hi = dk["lat_hi"].to(device)

    sampler = create_sampler(sampler="ddpm", steps=args.steps,
                             noise_schedule="cosine", model_mean_type="epsilon",
                             model_var_type="fixed_large", dynamic_threshold=False,
                             clip_denoised=True, rescale_timesteps=False,
                             timestep_respacing="")
    noiser = get_noise(sigma=0.0, name="gaussian")

    field_names = [str(n) for n in getattr(dataset, "field_names",
                                           [f"f{i}" for i in range(n_fields)])]
    summary = {}
    for snap in args.snapshots:
        item = dataset[int(snap)]
        coords = item["coords"].unsqueeze(0).to(device)
        truth = item["fields"].unsqueeze(0).to(device)
        torch.manual_seed(1000 + int(snap))
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = \
            HB.build_sparse_condition(
                coords_full=coords, fields_full=truth,
                cond_fields=args.cond_fields,
                n_obs_min=[args.n_obs], n_obs_max=[args.n_obs],
            )
        valid = obs_mask[0].bool()
        s_idx = obs_indices[0, valid].long()
        s_fld = obs_field_ids[0, valid].long()
        s_coords = (coords[0, s_idx] * 2.0 - 1.0).unsqueeze(0)
        truth_pm1 = to_pm1(truth[0], lo_f, hi_f)
        y = truth_pm1[s_idx, s_fld]                       # [M] measured, [-1,1]

        op = CNFSensorOperator(decoder, s_coords, lat_lo, lat_hi)
        base_forward = op.forward
        fsel = s_fld.view(1, -1)

        class _Op:
            def forward(self, z_n, **kw):
                pred = base_forward(z_n)                  # [B, M, F]
                return pred.gather(2, fsel.unsqueeze(-1).expand(
                    pred.shape[0], -1, 1)).squeeze(-1)    # [B, M]
        cond = get_conditioning_method(name="ps", operator=_Op(), noiser=noiser,
                                       scale=args.dps_scale)
        sample_fn = partial(sampler.p_sample_loop, model=denoiser,
                            measurement_cond_fn=partial(cond.conditioning),
                            record=False, save_root=None)

        ens = []
        for k in range(args.K):
            torch.manual_seed(5000 + 97 * int(snap) + k)
            z0 = torch.randn(1, int(dk["dim"]), device=device)
            z_n = sample_fn(x_start=z0, measurement=y.unsqueeze(0))
            z = op._unnorm(z_n.detach().reshape(1, -1))
            with torch.no_grad():
                preds = []
                for s in range(0, coords_full.shape[0], args.chunk):
                    preds.append(decoder(coords_full[s:s+args.chunk].unsqueeze(0),
                                         z.unsqueeze(1))[0])
                pred_pm1 = torch.cat(preds, dim=0)
            # back to the z-scored units every other method is scored in
            phys = from_pm1(pred_pm1, lo_f, hi_f)
            ens.append(phys.float().cpu().numpy())
        ens = np.stack(ens, axis=0)
        m = ensemble_metrics(ens, truth[0].float().cpu().numpy(), field_names)
        m["snapshot"] = int(snap)
        agg = m["aggregate"]
        print(f"[confild-cond] snap {snap} " + " ".join(
            f"{k}={v:.5f}" for k, v in agg.items()), flush=True)
        json.dump(m, open(out / "Evaluation" / f"crps_snap{snap}.json", "w"), indent=1)
        summary[int(snap)] = agg

    json.dump(summary, open(out / "Evaluation" / "crps_summary.json", "w"), indent=1)
    print("[confild-cond] done", flush=True)


if __name__ == "__main__":
    main()
