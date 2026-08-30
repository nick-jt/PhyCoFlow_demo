# Fleet summary — accuracy, UQ, runtime, memory (2026-08-30)

JHU cross-cube, 1% sensors/observed channel (Ux,Uz observed; Uy,p unobserved), canonical
seeded protocol, n=50 snapshots, K=8 ensembles at each method's listed sampler setting.
Per-channel rel-L2 is primary. All GPU numbers on H100-SXM 80GB.

| Method | Ux | Uy* | Uz | p* | agg | CRPS | sp/err | cov90 | Params | Opt. steps | Train s/step | Train GPU-h | Train peak GB | Infer s/field | Infer peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Ours (N29, NFE 4)** | 0.176 | 1.048 | 0.182 | 0.964 | 0.593 | 0.291 | 0.63 | 0.54 | 6.51M | 48,000 | 0.574 | 7.7 | 53.6 | ~17 (d) | — |
| Latent FM (fixed) | 0.248 | 0.727 | 0.257 | 0.646 | 0.469 | 0.244 | 0.68 | 0.56 | 6.68M | 209k (+95k AE) | 0.238 | 13.9 | 3.6 | 0.019 | 0.6 |
| SiT-point (N=32) | 0.143 | 0.968 | 0.132 | 0.937 | 0.545 | 0.338 | 0.75 | 0.58 | 6.57M | 38,500 | 1.81 | 19.4 | 26.8 | ~150 (d) | — |
| FNO3D (ours framework, NFE 4) | 0.188 | 0.904 | 0.191 | 1.061 | 0.586 | 0.269 | 0.94 | 0.63 | 6.73M | 79,914 | 0.766 | 17.0 | 48.6 | 0.097 | 6.7 |
| CoNFiLD P (pub. prior, DPS 1000) | 0.409 | 0.751 | 0.374 | 1.007 | 0.635 | 0.371 | 0.82 | 0.57 | 118.9M (waived) | wall-budgeted | 35.5 s/ep (st.1) | ~19 | 4.7 (st.1) | ~130/snap (d) | — |
| CoNFiLD C (cap1024) | 0.488 | 1.089 | 0.420 | 1.487 | 0.871 | 0.562 | 0.62 | 0.45 | 6.62M | wall-budgeted | 35.5 s/ep (st.1) | ~19 | 4.7 (st.1) | ~126/snap (d) | — |
| CoNFiLD F (384d) | 0.524 | 0.604 | 0.451 | 1.401 | 0.745 | 0.480 | 0.69 | 0.50 | 6.62M | wall-budgeted | 57.5 s/ep (st.1) | ~19 | 6.8 (st.1) | ~213/snap (d) | — |
| Gen4Turb (anneal, 32-step) | 0.224 | 0.996 | 0.455 | 1.005 | 0.670 | 0.407 | 0.37 | 0.33 | 6.16M | 105,000 | 0.598 | 17.4 | 47.4 | 1.65 | 1.4 |
| S3GM (jhu-tuned, N=200) | (eval running) | | | | | | | | 6.35M | 105,000 | 0.635 | 18.5 | 26.3 | ~46 | 13.3 |
| Senseiver (det.) | 0.361 | 1.077 | 0.380 | 0.978 | 0.699 | 0.479 | n/a | n/a | 6.42M | 79,224 | 0.776 | 17.0 | 22.4 | <1 (fwd) | — |
| DeepONet (det.) | 0.593 | 0.972 | 0.652 | 0.985 | 0.800 | 0.570 | n/a | n/a | 6.51M | 81,992 | 0.29 (compute) | 6.7 | 15.7 | 0.23 | 3.0 |
| KD-tree (det., CPU) | 0.209 | 1.000 | 0.219 | 1.000 | 0.607 | 0.406 | n/a | n/a | — | — | — | — | 10 (RSS) | 0.16 | cpu |
| IDW k=8 (det., CPU) | 0.174 | 1.000 | 0.181 | 1.000 | 0.589 | 0.396 | n/a | n/a | — | — | — | — | 10 (RSS) | 0.68 | cpu |
| Gappy POD r80 (det., CPU) | 0.523 | 1.306 | 0.540 | 1.443 | 0.953 | 0.673 | n/a | n/a | — | 45 s fit | — | — | 10 (RSS) | 0.14 | cpu |
| Constant (train mean) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.786 | n/a | n/a | — | — | — | — | — | ~0 | cpu |

Pending rows: S3GM canonical numbers (finalize eval in progress), Senseiver-wide (40,960
bottleneck) and DeepONet-wide (1,240 bottleneck frontier) — both training, results today.

Notes:
- (d) = derived from eval-job wall time, not a dedicated bench. Ours ≈17 s/field is one
  NFE-4 sample of the full 1.95M-point cloud through the KeOps top-K path; FNO3D's 0.097 s
  is the same framework with a grid-FFT backbone. SiT ≈150 s is 239 sequential 8,192-token
  chunks at N=32 steps. CoNFiLD is 1000-step DPS optimization per window.
- Train s/step is the honest production step (data-resident, spectral loss on where used).
  DeepONet lists strict compute (0.29); its fleet step incl. loader wait is 0.74
  (loader = 53% of wall — augmentation CPU tax). Steps are NOT budget-equalized: every
  external baseline received ≥ our 48k steps except SiT (wall-matched, 38.5k).
- sp/err + cov90 at (1%, K=8, listed sampler steps, best.pt) — the operating point moves
  these more than method gaps (NFE 4→16: +0.22 sp/err; K 8→32: +0.11 cov90).
- Deterministic methods: CRPS ≡ MAE, spread undefined (n/a).
- Uy*, p* unobserved: >1.000 is worse than predicting the training mean.
- CoNFiLD stage-2 trained on a wall-clock budget (19,800–21,600 s) rather than a step count.
