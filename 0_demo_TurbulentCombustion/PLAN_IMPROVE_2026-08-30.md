# Baseline improvement campaign — 2026-08-30 (Nick's directive)

Compute constraint LIFTED ("assume plenty of compute hours"). DeepONet runs STOPPED
(architecturally weak; wide arm 17060802 cancelled at 16.2h). The improvement arms below are
Nick-specified. Upstream-faithful rows in the table are frozen; every arm here is a labelled
enhanced/tuned variant.

## Fleet-wide tuning hygiene (applies to every arm)
- TUNE split: cube-3 odd val indices {1,3,...,49} (25 snaps) — guidance sweeps, blend
  weights, early-stopping decisions, checkpoint selection.
- TEST split: even indices {0,...,48} — untouched until the tuned setting is frozen;
  primary reported number. Full-50 reported secondary, labelled "tuned on odd split".
- Fixed validation masks: canonical seed-0 sensor draw at 19,531/channel for all val evals.
- Canonical fingerprint gate + compute-node rule everywhere; account f2pde.
- New code via the proven wrapper/monkey-patch pattern (lfm_fixes.py): execution copies in
  the job tmp dir, durable copies staged on the eval-infra-audit worktree for commit.

## 1. Senseiver
(a) Capacity sweep: 128 (= sen_wide 17060697, already training — reuse), 256, 512 latent
    tokens × 320 width; fixed val masks; early stopping at fixed-validation optimum
    (patience on the seed-0 val eval, not wall budget). Expectation recorded: capacity
    alone gives limited improvement.
(b) ENHANCED variant (the important one): local conditioning path so nearby measurements
    bypass the global latent bottleneck — primary: IDW residual path (output = senseiver
    prediction + learned per-channel gate × IDW-k8 field, gate init 0); secondary if time:
    local query-to-sensor cross-attention (top-K neighbours, K=8, KeOps or masked dense).
    Param-matched ±10%; labelled "Senseiver+local".

## 2. DeepONet (redesign BEFORE any capacity increase)
Branch: replace global mean pooling with spatially structured aggregation — multiresolution
binned pooling (e.g. 4^3+8^3 occupancy-weighted bins) and/or attention pooling; Fourier
moments as a cheap alternative. Trunk: Fourier features or SIREN. Then tune basis rank p on
the TUNE split; early-stop at fixed-validation optimum. Labelled "DeepONet++ (structured
branch)"; the vanilla row remains as the faithful reference.

## 3. CoNFiLD
Stage A — failure isolation (cheap, before any retraining):
  (i) full-field latent fitting on test fields (codec ceiling per arm);
  (ii) sparse sensor-only latent optimization (no diffusion — how much do 39k sensors
      constrain the latent?);
  (iii) DPS guidance-scale sweep on the TUNE split.
Stage B — fixes: validation-based stage-2 checkpoint selection; codec (stage 1) trained to
  held-out plateau (not wall budget); final sensor-consistency correction (post-DPS
  projection of observed entries + local relaxation); single-frame arm (window=1) so its
  temporal information matches the other baselines. Arms labelled; P (published prior)
  remains the headline CoNFiLD row.

## 4. S3GM (training healthy — sampler is the problem)
(a) Regenerate the completed run's figures with stable guidance (monitor fix 2e6481b is on
    disk; run visualize on best.pt for a few epochs' worth of panels).
(b) Sampler upgrade: NORMALIZE the sensor- and slab-consistency gradients (divide by their
    norms) and give them a proper step size (ζ/||grad||, DPS-style), so guidance strength
    is noise-level- and scale-invariant; tune ζ_obs/ζ_consis on the TUNE split (include the
    α=1.0 finding: stability edge moved outward on the converged model — agg 0.652 at
    N=200/K=1).
(c) Noise-level-specific diagnostics: per-σ obs_rmse and max|x| traces during sampling.
Retraining / true-3D architecture: explicitly deferred until (a)-(c) are measured.

## 5. LDW-FFM (ours; deferral lifted for this specific test)
Stage 1 (cheap, no training): per-channel blend w·IDW + (1-w)·FFM on the TUNE split,
  w ∈ [0,1] grid including w=1 (interpolation fallback) and w=0 (FFM unchanged); tuned per
  channel; verify on TEST split + across sensor densities (0.1%, 1%, 10%). Success bar:
  consistent observed-channel improvement on untouched snapshots.
Stage 2 (only if Stage 1 passes): residual-FFM config — LDW (IDW) supplies the deterministic
  baseline + support features; FFM models u - û_LDW; unobserved channels use the training
  mean as baseline. Optional config; existing FFM stays default. Retention bar: held-out
  accuracy improvement without materially degrading calibration, spectral fidelity, or
  scaling.

## Success/stop criteria
Every arm reports TUNE-split selection evidence + TEST-split primary numbers + the same
cost instrumentation as the fleet table. An arm that fails its bar is reported and dropped,
not tuned until it passes.

## LOGIN-NODE RULE (Nick, 2026-08-30)
No heavy compute on login nodes. Anything >1 min CPU or >1 GB RAM — model builds for
param counting, latent fitting, figure compositing, npz analysis — goes through
sbatch/srun (CPU standby partition is fine for non-GPU work). Login-node python is for
quick file/JSON operations only. Never leave detached background loops (an orphaned
heredoc burned 4 cores for 2.5h on kl6 before being killed). Eval correctness already
requires compute nodes (CUDA randperm portability); this extends the rule to etiquette.
