"""Senseiver improvement-arm patches (PLAN_IMPROVE_2026-08-30.md section 1).

Applied by monkey-patch, lfm_fixes.py-style: the main checkout is
write-guarded and several agents share src/model_baseline.py, so these
additive, env-gated patches live outside the repo.  Execution copy:
/home/ntricard/.claude/jobs/3ac3fd02/tmp/senseiver_sweep/; durable copy staged
on the eval-infra-audit worktree.  Every patch is strictly opt-in via
environment variables; with none of them set, all patched functions reproduce
the original numerics exactly.

---------------------------------------------------------------------------
PATCH A -- fixed-mask TUNE-split validation + patience early stopping
    SEN_VAL_FIXED=1        validation passes use a FIXED sensor draw:
                           torch.manual_seed(SEN_VAL_SEED, default 0) right
                           before the val epoch and n_obs pinned to
                           SEN_VAL_NOBS (default 19531 = 1%) per channel,
                           instead of the training-time random 0.1-1% draw.
                           RNG state (CPU+CUDA) is saved/restored around the
                           pass so the training stream is untouched.  best.pt
                           therefore becomes the fixed-validation optimum
                           (PLAN section "Fixed validation masks").
    SEN_VAL_SUBSET=odd     the val dataset is restricted to the TUNE split =
                           cube-3 ODD indices {1,3,...,49} (25 snaps), per
                           the fleet-wide tuning-hygiene rule: patience-based
                           stopping is selection beyond standard best.pt, so
                           it must not see the TEST (even) split.
    SEN_ES_PATIENCE_EPOCHS patience in EPOCHS (not val evals); when the fixed
                           val metric has not improved for this many epochs,
                           the run prints an [earlystop] marker, records
                           earlystop_tune.json, and exits cleanly (code 0).
                           best.pt (the fixed-val optimum) is already on disk.
                           The BASELINE_MAX_HOURS wall budget remains the
                           backstop.
    NOTE this makes the val metric within a run comparable across epochs, but
    it is NOT the canonical per-snapshot eval layout (that seeds per snapshot
    id); the paper numbers come from the chained eval_senseiver_iclr.py run.

PATCH B -- "Senseiver+local": IDW residual path  (ENHANCED VARIANT, not
    upstream).  Nearby measurements must not pass entirely through the global
    latent bottleneck, so the final prediction becomes

        y(q) = senseiver_decoder(q) + gate[c] * IDW_k(q, sensors_c)

    per OBSERVED channel c (unobserved channels get no residual: their IDW
    field is identically zero).  gate is a per-channel plain scalar
    nn.Parameter initialised to 0, so at init the model IS the unpatched
    Senseiver; the parameter delta is +n_fields = +4.  The IDW interpolant
    mirrors baseline_classical_jhu.kd_predict (mode='idw'): k nearest
    sensors of the same channel, weights 1/max(d,1e-12) with an exact-hit
    override at d < 1e-11, NON-periodic distances (audit ruling) -- but
    implemented with torch.cdist + topk so it runs on-GPU inside the
    training loop, chunked over query points.  The interpolant is computed
    under no_grad: gradients flow to the gate (and the decoder) only.
        SEN_LOCAL_IDW=1    activate (adds the gate parameter -> checkpoints
                           trained with this flag REQUIRE it at eval time).
        SEN_IDW_K          neighbours per query (default 8).
        SEN_IDW_CHUNK      query-chunk size for the cdist (default 1024).
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn

import model_baseline as MB

# --------------------------------------------------------------------------
# env gates
# --------------------------------------------------------------------------
_VAL_FIXED = os.environ.get("SEN_VAL_FIXED", "0") == "1"
_VAL_SEED = int(os.environ.get("SEN_VAL_SEED", "0"))
_VAL_NOBS = int(os.environ.get("SEN_VAL_NOBS", "19531"))
_VAL_SUBSET = os.environ.get("SEN_VAL_SUBSET", "")
_PATIENCE = int(os.environ.get("SEN_ES_PATIENCE_EPOCHS", "0"))

_LOCAL = os.environ.get("SEN_LOCAL_IDW", "0") == "1"
_IDW_K = int(os.environ.get("SEN_IDW_K", "8"))
_IDW_CHUNK = int(os.environ.get("SEN_IDW_CHUNK", "2048"))
# cdist mode.  fp32 matmul-expansion cdist loses ~5e-4 absolute accuracy
# near d=0 (cancellation); the direct kernel is accurate to ~1e-7 but was
# measured 23x slower in the training loop (smoke 17105466: 18.3 s/step vs
# 0.80 base).  Queries and sensors both sit on the 125^3 grid, so true
# distances are either exactly 0 or >= 1/124 ~ 8.06e-3: an exact-hit
# threshold of 1e-3 separates the two cases with >10x margin over the mm
# error, and non-exact neighbour distances carry <= ~6% relative weight
# perturbation, which the learned gate absorbs.  mm mode is therefore the
# default; SEN_IDW_CDIST_MODE=donot_use_mm_for_euclid_dist restores the
# exact kernel.
_IDW_CDIST_MODE = os.environ.get("SEN_IDW_CDIST_MODE",
                                 "use_mm_for_euclid_dist")
_IDW_EXACT_EPS = float(os.environ.get("SEN_IDW_EXACT_EPS", "1e-3"))

print(f"[senfix] SEN_VAL_FIXED={int(_VAL_FIXED)} SEN_VAL_SEED={_VAL_SEED} "
      f"SEN_VAL_NOBS={_VAL_NOBS} SEN_VAL_SUBSET='{_VAL_SUBSET}' "
      f"SEN_ES_PATIENCE_EPOCHS={_PATIENCE} SEN_LOCAL_IDW={int(_LOCAL)} "
      f"SEN_IDW_K={_IDW_K} SEN_IDW_CHUNK={_IDW_CHUNK}", flush=True)


# --------------------------------------------------------------------------
# PATCH A1: TUNE-split (odd-index) validation subset
# --------------------------------------------------------------------------
class _SubsetView:
    """Index-subset view of a dataset that forwards every attribute
    (mean/std/field_names/num_fields/...) to the base dataset."""

    def __init__(self, base, indices):
        self._base = base
        self._indices = list(indices)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._base[self._indices[i]]

    def __getattr__(self, name):
        return getattr(self._base, name)


_orig_build_dataset = MB.build_dataset


def _build_dataset(cfg, split, stats_path=None, **kw):
    ds = _orig_build_dataset(cfg, split=split, stats_path=stats_path, **kw)
    if _VAL_SUBSET == "odd" and split == "val":
        idx = list(range(1, len(ds), 2))
        print(f"[senfix] val = TUNE split (cube-3 ODD indices): "
              f"{len(idx)} of {len(ds)} snapshots {idx[:5]}...", flush=True)
        return _SubsetView(ds, idx)
    return ds


if _VAL_SUBSET:
    MB.build_dataset = _build_dataset


# --------------------------------------------------------------------------
# PATCH A2: fixed-mask validation + patience early stop
# --------------------------------------------------------------------------
_A = MB.SenseiverAdapter
_orig_run_epoch = _A.run_epoch
_es = {"best": float("inf"), "best_epoch": None, "history": []}


def _run_epoch(self, bundle, loader, training, epoch):
    if training or not _VAL_FIXED:
        return _orig_run_epoch(self, bundle, loader, training, epoch)

    cond = bundle.config["shared"]["conditioning"]
    saved = (cond["n_obs_min_list"], cond["n_obs_max_list"])
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        nf = len(cond["cond_fields"])
        cond["n_obs_min_list"] = [_VAL_NOBS] * nf
        cond["n_obs_max_list"] = [_VAL_NOBS] * nf
        torch.manual_seed(_VAL_SEED)  # seeds CUDA generators as well
        val = _orig_run_epoch(self, bundle, loader, training, epoch)
    finally:
        cond["n_obs_min_list"], cond["n_obs_max_list"] = saved
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    improved = math.isfinite(val) and val < _es["best"]
    if improved:
        _es["best"] = float(val)
        _es["best_epoch"] = int(epoch)
    _es["history"].append({"epoch": int(epoch), "val_fixed": float(val),
                           "improved": bool(improved)})
    since = (epoch - _es["best_epoch"]) if _es["best_epoch"] is not None else 0
    print(f"[senfix] epoch={epoch:04d} val_fixed={val:.6e} "
          f"best={_es['best']:.6e}@ep{_es['best_epoch']} "
          f"epochs_since_best={since} patience={_PATIENCE}", flush=True)
    try:
        with open(bundle.run_dir / "earlystop_tune.json", "w", encoding="utf-8") as h:
            json.dump({
                "policy": "patience on fixed-mask TUNE-split val "
                          "(SEN_VAL_FIXED seed draw, best.pt = fixed-val optimum)",
                "val_subset": _VAL_SUBSET or "full_val",
                "val_seed": _VAL_SEED, "val_n_obs_per_field": _VAL_NOBS,
                "patience_epochs": _PATIENCE,
                "best_val_fixed": _es["best"], "best_epoch": _es["best_epoch"],
                "epochs_since_best": int(since),
                "history": _es["history"],
            }, h, indent=2)
    except Exception as exc:  # bookkeeping must never kill training
        print(f"[senfix] could not write earlystop_tune.json: {exc}", flush=True)

    if _PATIENCE > 0 and _es["best_epoch"] is not None and since >= _PATIENCE:
        print(f"[earlystop] fixed-val optimum {_es['best']:.6e} at epoch "
              f"{_es['best_epoch']}; no improvement for {since} epochs "
              f">= patience {_PATIENCE}. best.pt is the fixed-validation "
              f"optimum; exiting cleanly.", flush=True)
        raise SystemExit(0)
    return val


_A.run_epoch = _run_epoch


# --------------------------------------------------------------------------
# PATCH B: IDW residual path ("Senseiver+local")
# --------------------------------------------------------------------------
@torch.no_grad()
def _idw_interp(query_coords, obs_coords, obs_values, obs_mask, obs_field_ids,
                n_fields, k, chunk):
    """Per-observed-channel k-NN inverse-distance interpolation on GPU.

    Numerics mirror baseline_classical_jhu.kd_predict(mode='idw'):
    w = 1/max(d, 1e-12), exact hit (d < _IDW_EXACT_EPS, see mode note above)
    takes the nearest value, non-periodic Euclidean distances.  Unobserved channels stay 0 (== the
    train-split mean in z-score units, same convention as kd_predict).
    Returns [B, Q, n_fields].
    """
    B, Q, D = query_coords.shape
    device, dtype = query_coords.device, query_coords.dtype
    out = torch.zeros(B, Q, n_fields, device=device, dtype=dtype)
    valid = obs_mask.bool()
    fids = obs_field_ids.long()
    vals = obs_values[..., 0]

    for f in range(n_fields):
        m_f = (fids == f) & valid                       # [B, M]
        counts = m_f.sum(dim=1)                         # [B]
        if int(counts.min()) == 0:
            if int(counts.max()) == 0:
                continue  # channel unobserved everywhere: no residual
            # mixed presence cannot happen with fixed cond_fields; be safe
            raise RuntimeError(
                f"[senlocal] field {f} observed in only part of the batch")
        m_max = int(counts.max())
        # valid-first ordering per row -> padded per-field sensor tensors
        order = torch.argsort((~m_f).to(torch.uint8), dim=1, stable=True)
        sel = order[:, :m_max]                          # [B, m_max]
        coords_f = torch.gather(
            obs_coords, 1, sel.unsqueeze(-1).expand(-1, -1, D)).clone()
        vals_f = torch.gather(vals, 1, sel)
        pad = torch.arange(m_max, device=device).unsqueeze(0) >= counts.unsqueeze(1)
        # push padded sensors far outside the unit box so topk never picks
        # them (real neighbours exist: counts >= k is checked below)
        coords_f[pad] = 2.0e3
        kk = min(int(k), int(counts.min()))
        for s in range(0, Q, chunk):
            q = query_coords[:, s:s + chunk]            # [B, q, D]
            d = torch.cdist(q, coords_f, compute_mode=_IDW_CDIST_MODE)
            d_k, i_k = torch.topk(d, kk, dim=2, largest=False)
            v_k = torch.gather(
                vals_f.unsqueeze(1).expand(-1, d_k.shape[1], -1), 2, i_k)
            exact = d_k[..., 0] < _IDW_EXACT_EPS
            w = 1.0 / d_k.clamp_min(1e-12)
            est = (w * v_k).sum(-1) / w.sum(-1)
            est = torch.where(exact, v_k[..., 0], est)
            out[:, s:s + chunk, f] = est
    return out


_S = MB.Senseiver
_orig_s_init = _S.__init__
_orig_s_forward = _S.forward


def _s_init(self, *args, **kwargs):
    _orig_s_init(self, *args, **kwargs)
    if _LOCAL:
        # plain scalar (not sigmoid): init 0 makes the model exactly the
        # unpatched Senseiver at step 0, and the learned value is directly
        # the blend weight.  +n_fields parameters.
        self.local_gate = nn.Parameter(torch.zeros(self.n_fields))
        self.local_idw_k = _IDW_K
        print(f"[senlocal] IDW residual path ACTIVE: k={_IDW_K} "
              f"gate_params={self.n_fields} chunk={_IDW_CHUNK}", flush=True)


def _s_forward(self, query_coords, obs_coords, obs_values, obs_mask, obs_field_ids):
    base = _orig_s_forward(self, query_coords, obs_coords, obs_values,
                           obs_mask, obs_field_ids)
    gate = getattr(self, "local_gate", None)
    if gate is None:
        return base
    idw = _idw_interp(query_coords, obs_coords, obs_values, obs_mask,
                      obs_field_ids, self.n_fields, self.local_idw_k,
                      _IDW_CHUNK)
    return base + gate.view(1, 1, -1) * idw


if _LOCAL:
    _S.__init__ = _s_init
    _S.forward = _s_forward
