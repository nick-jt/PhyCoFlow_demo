# Upstream-faithful CoNFiLD for packed JHU cubes

This directory is isolated from the existing baseline implementation. It imports
the original CoNFiLD networks from the clone at
`/projects/ammoniacomb/generative_reconstruction/baselines/CoNFiLD` and stores
all new checkpoints under caller-selected output directories.

## Fidelity contract

The primary stage-1 configuration preserves the upstream decoder, zero latent
initialization, separate Adam optimizers, per-minibatch latent updates, and one
decoder update after a complete balanced latent epoch. Field normalization uses
the upstream case-4 `dim: 0` behavior: one temporal min/max for every spatial
point and channel, computed from training cubes only.

The required JHU adaptations are:

- uniform coordinate subsampling because a full `125^3` field does not fit in
  each SIREN minibatch;
- deterministic physical octahedral transforms before normalization;
- explicit packed-cube train/validation indices;
- a standalone frozen-decoder auto-decoding evaluator.

No validation or test-time latent optimization occurs in the training process.

## Tests

```bash
cd temp/confild_upstream_jhu
python -m unittest discover -s tests -v
```

The tests cover upstream normalization semantics, all 48 unique symmetry
elements and their inverses, balanced epochs, the two-timescale optimizer
boundary, and decoder-gradient isolation during latent fitting.

## Stage 1

Start with a tractable fidelity run before a 48-group production run:

```bash
python train_stage1.py \
  --out-dir runs/stage1_g8_d384 \
  --groups 8 --latent-dim 384 --hidden 384 \
  --epochs 3000 --batch-size 4 --points-per-item 65536
```

The production protocol uses `--groups 48`. Training duration is measured in
epochs, and therefore decoder updates, rather than ambiguous minibatch steps.

Held-out auto-decoding is always a separate command:

```bash
python evaluate_stage1.py \
  --checkpoint runs/stage1_g8_d384/best.pt \
  --out-dir runs/stage1_g8_d384/heldout
```

The evaluator freezes every decoder parameter, fits latents against an 80%
point subset, selects on a fixed subset of the disjoint 20%, and reports final
full-field errors in the same global training z-score units used by the JHU
benchmark.

`launch_stage1_screen.slurm` defines a six-arm controlled screen over latent
dimension, decoder/latent learning rates, and coordinate budget. Compare those
runs with `python summarize_stage1.py runs/screen_arm*` before launching
`launch_stage1_production.slurm`; held-out scores are intentionally not used to
select the training checkpoint.

## Checkpoint policy

`latest.pt` is a resumable periodic checkpoint. `best.pt` is selected only by
training-code reconstruction, never by held-out cube performance. Evaluation
outputs cannot overwrite either checkpoint.

## Stage 2

Stage 2 uses the original CoNFiLD/OpenAI-style 2D diffusion UNet rather than a
replacement MLP. It constructs `[time, latent]` images without crossing cube or
symmetry boundaries. The default 32-frame window is the largest practical
power-of-two window inside each 50-frame packed cube.
The UNet retains case 4's nominal `image_size=384`, channel multipliers, and
attention-resolution metadata; the model accepts the rectangular latent image
without changing its convolutional weights.

```bash
python train_stage2.py \
  --stage1-checkpoint runs/stage1_g48_d384/best.pt \
  --out-dir runs/stage2_g48_d384
```

Latent images use the single global min/max normalization in the upstream
diffusion training script. The UNet, cosine diffusion, epsilon objective, and
EMA behavior are imported from the upstream clone.

## Conditional evaluation

```bash
python evaluate_conditional.py \
  --stage1-checkpoint runs/stage1_g48_d384/best.pt \
  --stage2-checkpoint runs/stage2_g48_d384/latest.pt \
  --out-dir runs/stage2_g48_d384/conditional
```

The evaluator uses upstream posterior sampling through the frozen neural field,
observes only `Ux` and `Uz`, and never hard-clamps generated fields. Sensor draws
and probabilistic metrics use the repository's shared JHU helpers.
