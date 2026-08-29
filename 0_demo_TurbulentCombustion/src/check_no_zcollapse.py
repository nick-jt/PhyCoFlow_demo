#!/usr/bin/env python3
"""GUARD: catch 3-D fields being collapsed onto (x,y) in any figure path.

    python check_no_zcollapse.py            # audit every .py here
    python check_no_zcollapse.py foo.py     # audit one file
    exit 0 = clean, exit 1 = suspect paths found

WHY THIS EXISTS -- read before deleting.
---------------------------------------
This bug has now appeared THREE times and is invisible in every metric:

  1. commit 988bc86 -- the SiT and S3GM visualizers passed all 1,953,125 points
     of a 125^3 cube to a 2-D triangulation keyed on (x,y) only, superposing 125
     z-planes onto 15,625 locations. The result is a SMEAR, not a cross-section,
     with the colour scale stretched to the volume min/max.
  2. reported again 2026-08-29 on the periodic (per-epoch) figures.
  3. the same class of error as taking a z-MEAN instead of a z-SLICE.

Why it is dangerous: a z-mean or a z-superposition is far smoother than any real
slice, so EVERY method looks better and more similar than it is. It survives
review because the numbers are computed on the full volume and are correct --
only the picture lies. And the picture is what we use to catch everything else.

THE RULE
--------
Any 2-D render of a volumetric field must take a genuine SLICE at fixed z
(`helpers_baseline.midplane_slice`, or `reshape(nx,ny,nz,-1)[:, :, zmid, :]`).
Never `.mean(axis=z)`. Never hand all N points to a triangulation on (x,y).
Use the SAME plane for every method so figures are comparable.
"""
import ast, sys, glob, os

# Only matplotlib renders. Bare "scatter" is excluded: torch's scatter_/index ops
# and helper names like scatter_sensors_to_nodes are not plots.
PLOT = ("tripcolor", "tricontourf", "tricontour", "Triangulation", "imshow", "pcolormesh")
# "zcollapse-ok" implements the documented false-positive silencer: a
# `# zcollapse-ok: <reason>` comment inside the flagged function counts as a
# guard (it was documented but not implemented before 2026-08-29).
GUARD = ("midplane", "slice_mask", "zmid", "z_slice", "slice_idx", "on_slice",
         "_slice", "zcollapse-ok")

def audit(path):
    try: src = open(path, encoding="utf-8", errors="replace").read()
    except Exception: return []
    if not any(p in src for p in PLOT): return []
    try: tree = ast.parse(src)
    except SyntaxError: return []
    out = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        seg = ast.get_source_segment(src, fn) or ""
        calls = [p for p in PLOT if p in seg]
        if not calls: continue
        guarded = any(g in seg for g in GUARD)
        # a z-mean is the other face of the same bug
        zmean = any(t in seg for t in (".mean(axis=2)", ".mean(2)", "mean(axis=-2)"))
        if not guarded or zmean:
            out.append((fn.lineno, fn.name, ",".join(sorted(set(calls))),
                        "z-MEAN present" if zmean else "no slice guard"))
    return out

def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py")))
    bad = 0
    for f in files:
        if os.path.basename(f) == os.path.basename(__file__): continue
        for lineno, name, calls, why in audit(f):
            print(f"SUSPECT {os.path.basename(f)}:{lineno} {name}()  [{calls}]  -- {why}")
            bad += 1
    if bad:
        print(f"\n{bad} figure path(s) plot without a slice guard.")
        print("If the data is genuinely 2-D or an unstructured surface (wing), that is fine --")
        print("add a `# zcollapse-ok: <reason>` comment and re-run.")
        return 1
    print("clean: every plotting function that renders a field takes a slice.")
    return 0

# ---------------------------------------------------------------------------
# TRIAGE as of 2026-08-29. Re-check if any of these change.
#
# ACTIVE per-epoch paths -- ALL VERIFIED TO SLICE CORRECTLY, do not appear below:
#   helpers.visualize_reconstruction ............ ours
#   model_baseline.visualize_reconstruction_* ... latent FM / SiT / Senseiver / AE
#   s3gm3d.py ................................... via helpers_baseline.midplane_slice
#   confild_upstream_training.py ................ reshape(nx,ny,nz,-1) -> plane -> imshow
#   train_deeponet.py ........................... reshape(side,side,side,-1)[:, :, zmid, :]
#   baseline_classical_figs.py .................. midplane() helper
#
# FLAGGED, triaged:
#   NOT A FIELD (heatmap of a matrix) -- benign:
#     evaluate_coherence.save_pairwise_heatmap, evaluate_full_dataset._plot_summary_heatmap
#   GENUINELY UNSTRUCTURED SURFACE -- benign:
#     qualitative_wing.main
#   BUILDER, caller slices -- benign:
#     helpers_baseline._build_structured_triangulation
#   CLOSED OUT 2026-08-29:
#     helpers.save_smooth_mask_plot ................ WAS COLLAPSING (full-volume
#         coords_xy triangulation); fixed -- z-midplane slice added.
#     evaluate_coherence.save_worst_direction_spatial_map ... WAS COLLAPSING
#         (coords[:, :2] over the whole volume); fixed -- z-midplane slice,
#         rmse/linf stay full-volume.
#     qualitative_firebench.main ................... CLEAN (genuine fixed-y
#         slice through the fire front); zcollapse-ok comment added.
#   STILL FLAGGED, deliberately untouched (one-off scripts):
#     View_Dataset.create_png, train_finetune._save_rollout_field_figure
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
