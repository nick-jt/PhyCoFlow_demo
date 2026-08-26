# ICLR 2027 Paper Plan

## Working title
**LOCUS: Scalable Generative Reconstruction of Three-Dimensional Fields on Point Clouds**
(LOCUS = a set of points satisfying a condition; the model is built on locality-structured conditioning. Name is a placeholder — easy to change globally.)

## Core research gap (from verified literature sweep, Aug 2026)

The literature factorizes into three camps, none of which occupies the cell we target:

1. **Function-space generative models** (FFM Kerrigan AISTATS'24; DDO Lim JMLR'25; OpFlow/OFM Shi TMLR'24/NeurIPS'25; FunDPS Yao NeurIPS'25) are discretization-agnostic in principle but demonstrated at 64x64 2D with FFT/uniform-grid backbones; FunDPS explicitly concedes higher dimensions are unexplored. Infinite-dimensional stochastic interpolants are an open problem (Albergo & Vanden-Eijnden theory is R^d only).
2. **3D generative reconstruction** exists only (a) on voxel grids with documented memory pathologies — Gen4Turb (Oommen et al., Nat. Comm. '26) trains on 32^3 crops "to avoid out-of-memory errors" and disables attention; TurbDiff (ICLR'24) caps at 192x48x48; FactFormer at 64^3; P3D (ICLR'26) exists because full-volume 512^3 training is infeasible — or (b) through latent compression (CoNFiLD Nat. Comm. '24; FunDiff Nat. Comm. '26; Text2PDE ICLR'25), which pays a quantified dissipation-range price: MSE-trained VAEs retain 20-25% of dissipation-range spectral power (Rafiq & Nair '26); CoNFiLD self-admits small-scale/near-wall discrepancies; conditioning distorts compressed priors (Steinbrenner '26).
3. **The point-cloud scaling race is deterministic-only.** Transolver++ (ICML'25, 1M pts/GPU), AB-UPT (TMLR'25, 140M cells), GINO, UPT — all regression models that collapse to the posterior mean exactly when sparse observation makes the problem ill-posed. Generative point-cloud physics models are tiny (DGN ICLR'25 Oral: <=7k nodes; Kashefi '26: 1,024 points, 2D).

Additionally (evaluation gap): **realistic measurement operators are almost universally faked** — noiseless CFD point masks dominate (Shu JCP'23, DiffusionPDE NeurIPS'24, Amoros-Trepat PoF'26). Surface-only sensing assumes fully-resolved wall fields (Guastoni JFM'21); cross-modality inference (species -> flow) is 2D/deterministic/per-rig (Barwey '21/'22, Kildare '24). Zhuang (CMAME'25): deterministic wins when noiseless — the generative advantage must be argued via noise/ambiguity/UQ, which most papers don't test.

**The empty cell:** a generative model that samples full 3D fields *directly in ambient function space on scattered points* — no autoencoder, no voxelization, end-to-end — at ~10^6-point scale, conditioned on arbitrary sensor sets, evaluated under realistic measurement operators with calibrated posteriors.

## Punchline

Existing generative reconstructors either voxelize 3D fields and hit a memory wall, or compress them into latent codes and lose the dissipation range. We perform flow matching directly on field values at scattered query points. This is tractable because of two architectural decisions:

1. **Locality-structured conditioning:** each query point's velocity is assembled from (i) a compact global summary computed once per field by cross-attention over sensor tokens (O(M) in sensors), and (ii) a strictly local, importance-weighted k-NN retrieval over sensor tokens (O(K) per query, KeOps, no [N,M] materialization). Total cost is linear in query count with no all-to-all term — the model never builds a global volumetric representation.
2. **Monte Carlo function-space training:** because the flow lives pointwise on function values (with a smooth GP/RFF prior tying points together), the training loss is an integral over the domain that can be estimated on a random 1% subset of points per step. Voxel U-Nets and patchified transformers cannot subsample this way — their computation graph is global. This decouples training cost from resolution and is what makes 2M-point 3D fields trainable on one GPU.

Consequences we demonstrate: (a) resolution/discretization agnosticism — train at 1% scattered points, sample at full 125^3 or on unstructured aero meshes; (b) one amortized model handles arbitrary sensor counts, layouts, noise, and observed-channel patterns zero-shot; (c) few-NFE posterior sampling via rectified flow (2-16 steps vs thousands for guided diffusion); (d) calibrated posterior spread where deterministic point-cloud models regress to the mean.

## Section plan (problem-per-section, one dataset per problem)

- **Sec 4.1 — The 3D memory wall & spectral fidelity.** Dataset: JHU isotropic1024 cutout (125^3 ~= 2M pts, 617 snapshots, u,v,w,p). Show: training memory/wall-clock vs voxel latent-FM, SiT-patchify, guided voxel diffusion; dissipation-range spectra (LSD, band energies) vs latent-FM AE floor; error vs NFE (rectified flow few-step advantage); query-subsampling ablation (1% MC training works).
- **Sec 4.2 — Irregular geometry and surface-only sensing.** Dataset: SHIFT-WING (unstructured RANS volume meshes ~ millions of cells, varying CRM wing geometry, transonic). Task: surface pressure/shear observations (realistic taps) -> volumetric velocity/pressure/temperature posterior; generalization across geometries; no voxelization possible for baselines without resampling artifacts. Baselines: Senseiver, GINO/latent-grid, CoNFiLD.
- **Sec 4.3 — Realistic measurement operators: noise, occlusion, calibration.** Dataset: FireBench LES subset (wildfire plumes, complex 3D buoyant turbulence) [+ JHU for controlled noise sweeps]. Operators: Gaussian/heteroscedastic sensor noise, occluded slabs (smoke blocks optical access — physically motivated), dropout of sensors. Show: posterior calibration (spread-error, CRPS, rank histograms), deterministic baselines' mean-regression, robustness curves vs noise level.
- **Sec 4.4 — Cross-variable inference (modality gap).** Dataset: BLASTNet 3D combustion DNS subvolumes (67GB Kaggle benchmark) or FireBench channels. Task: observe velocity only -> infer temperature/species (the OH-in-flames problem); observe scalar -> infer velocity. Cross-channel joint PDFs, JSD.
- (Appendix: CoastalBench irregular coastal 3D circulation — optional extra domain; The Well subsets for breadth if time permits.)

## Baselines (fair representation of the three camps)

- **Latent generative:** in-repo latent flow matching (ConvAE3D + latent RF U-Net, param-matched) — the latent route, on every gridded dataset. External anchor: CoNFiLD (point-cloud native latent diffusion) adapted to JHU + SHIFT-WING.
- **Voxel generative:** guided diffusion with hard observation masking (Amoros-Trepat/Thuerey-style masked guidance; S3GM-style in-repo video-diffusion adapter) on JHU/FireBench grids; SiT with patchify tokenizer (in-repo) to exhibit the patchify wall.
- **Deterministic point-cloud:** Senseiver (in-repo Det baseline, closest published architecture); optionally GINO/Transolver on SHIFT-WING.
- Report memory ceilings honestly: where a baseline OOMs at full resolution, train it at the largest feasible crop/resolution and say so (this IS the point of Sec 4.1).

## Model attributes that must be kept (user constraint)
3D; ambient (non-latent) space; end-to-end (no AE); function space; generative. All already true of PointCloudFFM + GL_rbf_ENH backbone (1-rectified flow, RFF-GP prior, sensor-token conditioning, kNN-RBF local gather + global latent summary).

## Code changes planned (small-to-medium, aligned with punchline)
1. **Training efficiency:** bf16 autocast + GradScaler-free AMP, EMA weights, optional gradient checkpointing on the gather; target >=2x epoch speedup at 55.9GB -> <30GB.
2. **Generalized datasets:** loader for SHIFT-WING VTU/VTP (pyvista) -> [n_cases, n_p_i, n_c] variable-size point clouds + per-case geometry; loaders for FireBench/BLASTNet crops; unified normalized-coords interface (already there).
3. **Measurement operator module:** sensor noise injection (train + eval), occlusion masks (slab/ball), surface-only sampling (SHIFT-WING surface mesh indices), channel dropout for cross-modality (already have field_ids; add training-time random channel subsets).
4. **Ensemble evaluation:** K-sample posterior metrics (CRPS, spread-error, rank hist, coverage) — currently absent; cheap and high-value.
5. **Baseline 3D fixes:** s3gm/SiT adapters for 3D crops where needed.

## Writing notes
- ICLR format, professional publication tone; no narrative/explanatory tone, minimal italics, no rhetorical questions, not compressed to fragments. Contributions bulleted at end of intro pointing at result sections.
- Main text self-contained; architecture figure ICLR-style (encoder / local-global conditioning / flow integration panels + scaling inset).
- Reviewers must not need the appendix.
