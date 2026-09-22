"""Surface-to-field reconstruction gallery: the SAME cylinder frame as
fig_recon_gallery_cylinder.py, reconstructed from sensors confined to the
cylinder wall instead of scattered through the volume.

Read the two galleries side by side: identical flow, identical held-out
Reynolds number, identical frame, identical seeded draw machinery. The only
difference is where the sensors are allowed to sit, and the method ranking
inverts between them -- gappy POD wins the volume task at 0.056 and reaches
only 0.689 here, while Senseiver goes from 0.145 to 0.103.

Frame: val 300 (absolute 1500, held-out Re 250). Taps: 128 per field on the
wall ring, all three fields observed AT THE TAPS and nowhere else, drawn by
helpers.build_sparse_condition under the surface pool mask with the canonical
per-snapshot seed. Grid-locked methods see the ring rasterized onto 62 unique
cells, which is why their accuracy saturates at ~62 taps.

Panels whose npz is absent render as labelled empty slots, so the figure is
re-runnable as dumps land. Displayed in PHYSICAL units with truth-anchored
per-row scales; per-panel numbers are rel-L2 in that dump's own z-units.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from pof_style import use_style, save, INK_2, MUTED  # noqa: E402
# reuse the volume gallery's drawing primitives verbatim so the two figures are
# visually identical apart from the data
from fig_recon_gallery_cylinder import (  # noqa: E402
    draw_field, row_scale, rel_l2, grid_shape, triangulation,
)

WT = _HERE.parents[1]
DUMPS = WT / "Save_TrainedModel_pof" / "field_dumps"

ROWS = [(0, r"$u_x$", "tapped"), (1, r"$u_y$", "tapped"), (2, r"$p$", "tapped")]
PANELS = [
    ("gappy POD ($r{=}20$)", "cylsurf_classical.npz", "pred_gappy_pod_r20"),
    ("IDW", "cylsurf_classical.npz", "pred_idw"),
    ("Senseiver", "cylsurf_senseiver.npz", "pred_mean"),
    ("Geo-FNO", "cylsurf_geofno.npz", "pred_mean"),
    ("\\dmfgen", "cylsurf_dmfgen.npz", "pred_sample"),
    ("SiT", "cylsurf_sit.npz", "pred_sample"),
]


def load_dump(fname):
    d = np.load(DUMPS / fname, allow_pickle=False)
    coords = np.asarray(d["coords_raw"] if "coords_raw" in d.files else d["coords"],
                        dtype=np.float64)[:, :2]
    return d, coords


def main() -> None:
    use_style()
    cls_path = DUMPS / "cylsurf_classical.npz"
    if not cls_path.is_file():
        raise SystemExit(f"need {cls_path} first (dump_classical_gallery.py --dataset cylinder2d_surface)")
    cls, cls_coords = load_dump("cylsurf_classical.npz")
    meta = json.loads(str(cls["meta"]))
    mean = np.asarray(cls["norm_mean"])
    std = np.asarray(cls["norm_std"])
    truth_ph = np.asarray(cls["truth"]) * std + mean

    n_cols = 1 + len(PANELS)
    fig = plt.figure(figsize=(7.0, 1.80))
    gs = fig.add_gridspec(len(ROWS), n_cols + 1,
                          width_ratios=[1.0] * n_cols + [0.07],
                          left=0.055, right=0.945, top=0.90, bottom=0.025,
                          wspace=0.06, hspace=0.22)

    def style_ax(ax, coords=None):
        ax.set_xticks([]); ax.set_yticks([])
        if coords is not None:
            ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
            ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
            ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.55")

    s_idx = np.asarray(cls["sensor_indices"])
    s_fid = np.asarray(cls["sensor_field_ids"])

    for r, (j, name, _) in enumerate(ROWS):
        vmin, vmax, cmap = row_scale(truth_ph[:, j])
        ax = fig.add_subplot(gs[r, 0])
        im = draw_field(ax, cls_coords, truth_ph[:, j], vmin, vmax, cmap)
        taps = s_idx[s_fid == j]
        ax.scatter(cls_coords[taps, 0], cls_coords[taps, 1], s=2.2, c="k",
                   marker=".", linewidths=0, rasterized=True, zorder=4)
        style_ax(ax, cls_coords)
        ax.set_ylabel(name, fontsize=7, labelpad=2.0)
        if r == 0:
            ax.set_title("truth + taps", fontsize=7, pad=2.5)

        for c, (title, fname, key) in enumerate(PANELS, start=1):
            ax = fig.add_subplot(gs[r, c])
            pending = fname is None or not (DUMPS / fname).is_file()
            if not pending:
                d, pcoords = load_dump(fname)
                if key not in d.files:
                    pending = True
            if pending:
                ax.set_facecolor("#f0efe9")
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                        va="center", fontsize=6.5, color=MUTED)
                style_ax(ax, cls_coords)
                if r == 0:
                    ax.set_title(title.replace("\\dmfgen", "DMF-Gen"), fontsize=7,
                                 pad=2.5, color=MUTED)
                continue
            pm, ps = np.asarray(d["norm_mean"]), np.asarray(d["norm_std"])
            pred_z, tr_z = np.asarray(d[key]), np.asarray(d["truth"])
            err = rel_l2(pred_z[:, j], tr_z[:, j])
            draw_field(ax, pcoords, pred_z[:, j] * ps[j] + pm[j], vmin, vmax, cmap)
            style_ax(ax, pcoords)
            if r == 0:
                ax.set_title(title.replace("\\dmfgen", "DMF-Gen"), fontsize=7, pad=2.5)
            ax.text(0.985, 0.96, f"{err:.3f}", transform=ax.transAxes, fontsize=6.2,
                    ha="right", va="top", color="k",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))

        cax = fig.add_subplot(gs[r, n_cols])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([vmin, vmax] if cmap == "viridis" else [vmin, 0.0, vmax])
        cb.ax.tick_params(labelsize=5.6, length=2, pad=1)
        cb.outline.set_linewidth(0.4)

    # The explanatory text formerly drawn into the figure (tap count, what the
    # numbers are, the POD/interpolation reading) lives in the LaTeX caption:
    # 6 pt text inside a print figure is unreadable, and captions are edited
    # with the paper, not the dumps.
    print(f"[surface gallery] n_obs={meta.get('n_obs', '?')} taps per field")
    save(fig, "recon_gallery_cylinder_surface", pdf_dpi=300, png_dpi=220)


if __name__ == "__main__":
    main()
