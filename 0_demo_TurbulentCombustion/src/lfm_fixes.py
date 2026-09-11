"""Fidelity fixes for the latent flow-matching baseline, applied by monkey-patch.

Kept out of the shared checkout deliberately: five agents are editing
src/model_baseline.py concurrently, and these changes are additive and gated on
new config values, so patching at import time is equivalent to editing the file
and cannot conflict.  Both fixes are strictly opt-in:

  * they activate only when `cond_mode: image_norm` and/or
    `latent_scale_mode: {global,per_channel}` appear in the stage-2 config;
  * with the shipped config (`cond_mode: image`, no latent_scale_mode) every
    patched method reproduces the original numerics exactly.

--------------------------------------------------------------------------
FIX 1 -- conditioning attenuation  (cond_mode: "image_norm")
--------------------------------------------------------------------------
The shipped `image` conditioner builds a masked sparse grid (zeros everywhere
except at sensors) and average-pools it by the AE downsampling factor, here
8 in each axis = 512 voxels per latent cell.  A latent cell holding one sensor
of value v therefore presents v/512 to the denoiser.  Measured on cube 3:

    sensors     cond_feat std      true field std     attenuation
    1,953       0.00141            0.841              ~600x
    19,531      0.00846            0.841              ~100x

while the other input to the same `in_conv`, z_t, has std ~1-3.6.  The
conditioning signal enters the network 2-3 orders of magnitude below the noise
it is supposed to steer, and its magnitude encodes sensor *density* rather than
sensor *value*.  The mask channels cannot undo this: they are max-pooled, i.e.
binary, so the per-cell count needed to rescale is not available anywhere.

`image_norm` replaces the average with a masked mean (sum of observed values
divided by the observed count, zero where a cell holds no sensor), which
restores O(1) conditioning, and replaces the binary occupancy mask with a
saturating density count/(1+count).  The density is strictly more informative
than the binary mask (binary = 1[count>0]) and uses the same channel count, so
the velocity network's input channels, and hence its parameter count, are
unchanged.

--------------------------------------------------------------------------
FIX 2 -- latent scale  (latent_scale_mode: "global" | "per_channel")
--------------------------------------------------------------------------
The rectified flow's source is N(0, I), but the AE latents are not unit scale:
measured on the trained stage-1 model, std = 3.62 overall with per-channel std
spanning 0.77 to 6.31 (8x) and |z|max = 32.6.  LDM (Rombach et al. 2022) carries
a `scale_factor` (0.18215 for the released model) for exactly this reason, and
its KL term keeps the per-channel spread small; this autoencoder has neither.
Without rescaling, the flow-matching MSE on `x1 - x0` is dominated by the few
highest-variance latent channels and the low-variance channels receive almost
no gradient.

`global` reproduces LDM's single scalar 1/std(z).  `per_channel` divides each
latent channel by its own train-split std, which is the minimal substitute for
the missing KL regularizer.  Statistics are computed from the TRAIN split only.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

import model_baseline as MB

_LFM = MB.LatentFlowMatching

# ---------------------------------------------------------------- fix 1 ----
_orig_init = _LFM.__init__


def _init(self, ae, velocity_net, Num_x, Num_y, cond_mode="image",
          pointnet_encoder=None):
    requested = cond_mode
    _orig_init(self, ae, velocity_net, Num_x, Num_y,
               cond_mode=("image" if cond_mode == "image_norm" else cond_mode),
               pointnet_encoder=pointnet_encoder)
    self.cond_mode = requested
    self.register_buffer("latent_scale", torch.ones(1))


_orig_encode_condition = _LFM._encode_condition


def _encode_condition(self, cond_inputs: dict):
    if self.cond_mode != "image_norm":
        return _orig_encode_condition(self, cond_inputs)

    gv = cond_inputs["obs_value_grid"]
    gm = cond_inputs["obs_mask_grid"]
    factor = 2 ** self.n_levels
    pool = F.avg_pool3d if self.spatial_dim == 3 else F.avg_pool2d

    v = self._pad_to_ae(gv * gm)
    m = self._pad_to_ae(gm)
    num = pool(v, kernel_size=factor, stride=factor)
    den = pool(m, kernel_size=factor, stride=factor)
    cond_feat = torch.where(den > 0, num / den.clamp(min=1e-8),
                            torch.zeros_like(num))

    cnt = den * float(factor ** self.spatial_dim)
    m_ds = cnt / (1.0 + cnt)
    cond_mask = torch.cat([m_ds, m_ds.amax(dim=1, keepdim=True)], dim=1)
    return cond_feat, cond_mask


# ---------------------------------------------------------------- fix 2 ----
def _training_loss(self, fields_grid: torch.Tensor, cond_inputs: dict):
    with torch.no_grad():
        x1 = self.ae.encode(fields_grid) * self.latent_scale
    cond_feat, cond_mask = self._encode_condition(cond_inputs)

    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device, dtype=x1.dtype)
    t_ = t.view((B,) + (1,) * (x1.ndim - 1))
    x_t = (1 - t_) * x0 + t_ * x1
    target = x1 - x0

    pred = self.velocity_net(t, x_t, cond_feat, cond_mask)
    loss = F.mse_loss(pred, target)
    return loss, {"loss": float(loss.detach().cpu())}


@torch.no_grad()
def _sample(self, cond_inputs: dict, n_steps: int = 8, ode_solver: str = "euler"):
    if ode_solver not in ("euler", "heun"):
        raise ValueError(f"Unsupported ode_solver={ode_solver!r}.")
    cond_feat, cond_mask = self._encode_condition(cond_inputs)
    B = cond_feat.shape[0]
    x = torch.randn((B, self.latent_ch, *cond_feat.shape[2:]),
                    device=cond_feat.device, dtype=cond_feat.dtype)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=x.device, dtype=x.dtype)
        v = self.velocity_net(t, x, cond_feat, cond_mask)
        if ode_solver == "euler":
            x = x + dt * v
        else:
            t_next = torch.full((B,), (k + 1) * dt, device=x.device, dtype=x.dtype)
            v_next = self.velocity_net(t_next, x + dt * v, cond_feat, cond_mask)
            x = x + 0.5 * dt * (v + v_next)
    return self.ae.decode(x / self.latent_scale)


# --------------------------------------------------- adapter: set the scale --
_orig_build = MB.LatentFMAdapter.build_for_training


def _build_for_training(self, cfg, device, run_dir, train_set, val_set):
    bundle = _orig_build(self, cfg, device, run_dir, train_set, val_set)
    if int(cfg["training_stage"]) != 2:
        return bundle

    arch = cfg["latent_fm_params"]["stage2"]["architecture"]
    mode = str(arch.get("latent_scale_mode", "none")).lower()
    model = bundle.model
    print(f"[lfm_fixes] cond_mode={model.cond_mode} latent_scale_mode={mode}",
          flush=True)
    if mode == "none":
        return bundle

    num_x = int(cfg["shared"]["data"]["num_x"])
    num_y = int(cfg["shared"]["data"]["num_y"])
    num_z_raw = cfg["shared"]["data"].get("num_z")
    num_z = None if num_z_raw is None else int(num_z_raw)

    zs = []
    with torch.no_grad():
        for i in range(min(len(train_set), 24)):
            f = train_set[i]["fields"][None].to(device)
            g = (MB.pointcloud_to_grid3d(f, num_z, num_y, num_x) if num_z
                 else MB.pointcloud_to_grid(f, num_y, num_x))
            zs.append(model.ae.encode(g).float())
    z = torch.cat(zs)

    if mode == "per_channel":
        dims = [0] + list(range(2, z.ndim))
        scale = 1.0 / z.std(dim=dims, keepdim=True).clamp(min=1e-6)
    elif mode == "global":
        scale = (1.0 / z.std().clamp(min=1e-6)).reshape(1)
    else:
        raise ValueError(f"latent_scale_mode={mode!r} must be "
                         "'none', 'global' or 'per_channel'")

    model.latent_scale = scale.to(device=device, dtype=torch.float32)
    print(f"[lfm_fixes] raw latent std={float(z.std()):.4f} "
          f"per-ch std [{float(z.std(dim=[0,2,3,4]).min()):.3f},"
          f"{float(z.std(dim=[0,2,3,4]).max()):.3f}] -> "
          f"scale[:6]={[round(v, 4) for v in scale.flatten()[:6].tolist()]}",
          flush=True)
    return bundle


# ------------------------------------ persist the scale in the checkpoint --
# Recomputing from the train split at eval time would resample the octahedral
# augmentation, so store the exact tensor the run trained with.
_orig_build_ckpt = MB.LatentFMAdapter.build_checkpoint


def _build_checkpoint(self, bundle, epoch, train_loss, val_loss):
    ck = _orig_build_ckpt(self, bundle, epoch, train_loss, val_loss)
    if int(bundle.training_stage) == 2:
        ck["latent_scale"] = bundle.model.latent_scale.detach().cpu()
        ck["cond_mode"] = bundle.model.cond_mode

    # --- checkpoint-noise window ------------------------------------------
    # Both trainers keep only best.pt and last.pt (last.pt is OVERWRITTEN each
    # eval), so checkpoint-to-checkpoint variance cannot be measured from
    # existing artifacts. Archive a dense window around the budget point so the
    # headline comparison can be tested against checkpoint noise.
    # LFM_CKPT_WINDOW="lo:hi:step", epochs inclusive.
    spec = os.environ.get("LFM_CKPT_WINDOW", "")
    if spec and int(bundle.training_stage) == 2:
        lo, hi, step = (int(v) for v in spec.split(":"))
        if lo <= epoch <= hi and (epoch - lo) % step == 0:
            d = Path(bundle.run_dir) / "ckpt_window"
            d.mkdir(parents=True, exist_ok=True)
            torch.save(ck, d / f"epoch_{epoch:05d}.pt")
            print(f"[lfm_fixes] archived window checkpoint epoch={epoch}",
                  flush=True)
    return ck


_orig_load_ckpt = MB.LatentFMAdapter.load_checkpoint


def _load_checkpoint(self, bundle, checkpoint):
    _orig_load_ckpt(self, bundle, checkpoint)
    ls = checkpoint.get("latent_scale")
    if int(bundle.training_stage) == 2 and ls is not None:
        bundle.model.latent_scale = torch.as_tensor(ls).to(
            device=bundle.device, dtype=torch.float32)
        print(f"[lfm_fixes] restored latent_scale from checkpoint "
              f"(numel={bundle.model.latent_scale.numel()})", flush=True)


_LFM.__init__ = _init
_LFM._encode_condition = _encode_condition
_LFM.training_loss = _training_loss
_LFM.sample = _sample
MB.LatentFMAdapter.build_for_training = _build_for_training
MB.LatentFMAdapter.build_checkpoint = _build_checkpoint
MB.LatentFMAdapter.load_checkpoint = _load_checkpoint
print("[lfm_fixes] patches installed", flush=True)
