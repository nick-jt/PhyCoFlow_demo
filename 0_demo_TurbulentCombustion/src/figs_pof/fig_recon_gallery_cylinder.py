"""All-baselines reconstruction gallery: cylinder wake, ONE fixed frame.

Rows: Ux, Uy (observed channels) and p (UNOBSERVED -- no pressure taps).
Columns: truth+taps | gappy-POD (r=20) | IDW | DMF-Gen | SiT | Senseiver
| latent-FM* (pending: only the Stage1 AE checkpoint exists).

Frame: val 300 = absolute 1500 (held-out Re250 tail).  Sensors: 238/field
(1% of the 23,800-pt mesh) on Ux and Uy only, canonical draw
torch.manual_seed(0*777+300).  The p row is the punchline: gappy POD
reconstructs the unobserved channel through the basis's cross-channel
correlations, while IDW/kdtree have nothing to interpolate and fall back to
the train mean (rel-L2 = 1 by construction).

Data: Save_TrainedModel_pof/field_dumps/cyl_classical.npz
(dump_classical_gallery.py, mesh) and cyl_{dmfgen,sit,senseiver}.npz
(dump_kolm_gallery.sh; dmfgen+senseiver on the mesh at 238/field, SiT on the
400x200 grid at 800/field = the same 1%).  Missing npz render as labeled
empty slots -- the script is RE-RUNNABLE to fill panels as dumps land.

Display: PHYSICAL units (each npz is standardized with its OWN run's train
stats -- mesh vs grid stats differ -- so z-units are not a common currency
across columns); shared truth-anchored scale per row.  Per-panel rel-L2 is
computed in that dump's own z-units (the table's metric convention).
Rasterized, 7in, 8pt.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pof_style import use_style, save, INK_2, MUTED  # noqa: E402

WT = Path(__file__).resolve().parents[2]
DUMPS = WT / "Save_TrainedModel_pof" / "field_dumps"

ROWS = [(0, r"$u_x$" + "\n(obs.)", "observed"),
        (1, r"$u_y$" + "\n(obs.)", "observed"),
        (2, r"$p$" + "\n(unobs.)", "unobserved")]

# (title, npz, key); classical first per the gallery spec, then learned.
PANELS = [
    ("gappy POD ($r{=}20$)", "cyl_classical.npz", "pred_gappy_pod_r20"),
    ("IDW", "cyl_classical.npz", "pred_idw"),
    ("DMF-Gen (ours)", "cyl_dmfgen.npz", "pred_sample"),
    ("SiT", "cyl_sit.npz", "pred_sample"),
    ("Senseiver", "cyl_senseiver.npz", "pred_mean"),
    ("latent-FM", None, None),        # cylinder Stage2 flow not trained yet
]


def rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) /
                 (np.linalg.norm(true) + 1e-12))


def load_dump(fname):
    """npz -> dict(truth_z, pred keys..., coords_raw/coords, mean, std)."""
    d = np.load(DUMPS / fname, allow_pickle=False)
    coords = np.asarray(d["coords_raw"] if "coords_raw" in d.files
                        else d["coords"], dtype=np.float64)[:, :2]
    return d, coords


def grid_shape(coords):
    """(ny, nx, fast_x) when coords form a regular grid, else None."""
    xs, ys = np.unique(coords[:, 0]), np.unique(coords[:, 1])
    if xs.size * ys.size != coords.shape[0]:
        return None
    fast_x = np.ptp(coords[:xs.size, 0]) > 0
    return (ys.size, xs.size, fast_x)


_TRI_CACHE = {}


def triangulation(coords):
    import matplotlib.tri as mtri
    key = (coords.shape[0], float(coords[:, 0].sum()))
    if key not in _TRI_CACHE:
        _TRI_CACHE[key] = mtri.Triangulation(coords[:, 0], coords[:, 1])
    return _TRI_CACHE[key]


def draw_field(ax, coords, vals, vmin, vmax, cmap):
    gs_ = grid_shape(coords)
    if gs_ is not None:
        ny, nx, fast_x = gs_
        img = vals.reshape((ny, nx) if fast_x else (nx, ny))
        if not fast_x:
            img = img.T
        return ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                         extent=[coords[:, 0].min(), coords[:, 0].max(),
                                 coords[:, 1].min(), coords[:, 1].max()],
                         aspect="equal", interpolation="nearest",
                         rasterized=True)
    # unstructured mesh: Delaunay + gouraud fill; the cylinder disk is
    # interpolated across and then overdrawn with a white patch
    im = ax.tripcolor(triangulation(coords), vals, shading="gouraud",
                      cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    r = float(np.sqrt((coords[:, :2] ** 2).sum(1)).min())
    ax.add_patch(plt.Circle((0.0, 0.0), r, fill=True, fc="white",
                            ec="0.45", lw=0.4, zorder=3))
    return im


def row_scale(vals):
    """(vmin, vmax, cmap): diverging only for genuinely two-sided fields."""
    lo, hi = np.percentile(vals, [2, 98])
    if lo < 0 < hi and min(abs(lo), abs(hi)) / max(abs(lo), abs(hi)) > 0.25:
        v = float(max(abs(lo), abs(hi)))
        return -v, v, "RdBu_r"
    return float(lo), float(hi), "viridis"


def main() -> None:
    use_style()
    cls_path = DUMPS / "cyl_classical.npz"
    if not cls_path.is_file():
        raise SystemExit(f"need {cls_path} first (dump_classical_gallery.py)")
    cls, cls_coords = load_dump("cyl_classical.npz")
    meta = json.loads(str(cls["meta"]))
    approx = "APPROX" in meta.get("sensor_source", "")
    mean = np.asarray(cls["norm_mean"])
    std = np.asarray(cls["norm_std"])
    truth_z = np.asarray(cls["truth"])
    truth_ph = truth_z * std + mean
    n_cols = 1 + len(PANELS)
    # domain aspect ~2.09 (32.6 x 15.6): size the rows so panels stay flat
    fig = plt.figure(figsize=(7.0, 2.05))
    gs = fig.add_gridspec(len(ROWS), n_cols + 1,
                          width_ratios=[1.0] * n_cols + [0.07],
                          left=0.055, right=0.945, top=0.815, bottom=0.025,
                          wspace=0.06, hspace=0.22)

    def style_ax(ax, coords=None):
        ax.set_xticks([])
        ax.set_yticks([])
        if coords is not None:
            # each dump lives on its own coordinate scaling (mesh raw vs
            # normalized grid), but all cover the same physical domain --
            # per-panel limits keep the panels visually congruent
            ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
            ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
            ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_linewidth(0.4)
            s.set_color("0.55")

    s_idx = np.asarray(cls["sensor_indices"])
    s_fid = np.asarray(cls["sensor_field_ids"])

    for r, (j, name, obs) in enumerate(ROWS):
        vmin, vmax, cmap = row_scale(truth_ph[:, j])

        # ---- truth + taps --------------------------------------------------
        ax = fig.add_subplot(gs[r, 0])
        im = draw_field(ax, cls_coords, truth_ph[:, j], vmin, vmax, cmap)
        if obs == "observed":
            taps = s_idx[s_fid == j]
            ax.scatter(cls_coords[taps, 0], cls_coords[taps, 1], s=0.9,
                       c="k", marker=".", linewidths=0, rasterized=True)
        else:
            ax.text(0.985, 0.04, "no taps", transform=ax.transAxes,
                    fontsize=5.8, ha="right", va="bottom", color="k",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        style_ax(ax, cls_coords)
        ax.set_ylabel(name, fontsize=7, labelpad=2.0)
        if r == 0:
            ax.set_title("truth + taps" + ("*" if approx else ""),
                         fontsize=7, pad=2.5)

        # ---- method panels -------------------------------------------------
        for c, (title, fname, key) in enumerate(PANELS, start=1):
            ax = fig.add_subplot(gs[r, c])
            pending = fname is None or not (DUMPS / fname).is_file()
            d = None
            if not pending:
                d, pcoords = load_dump(fname)
                if key not in d.files:
                    pending = True
            if pending:
                ax.set_facecolor("#f0efe9")
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6.5, color=MUTED)
                style_ax(ax, cls_coords)   # match the field panels' shape
                if r == 0:
                    ax.set_title(title, fontsize=7, pad=2.5, color=MUTED)
                continue
            pm = np.asarray(d["norm_mean"])
            ps = np.asarray(d["norm_std"])
            pred_z = np.asarray(d[key])
            tr_z = np.asarray(d["truth"])
            err = rel_l2(pred_z[:, j], tr_z[:, j])
            draw_field(ax, pcoords, pred_z[:, j] * ps[j] + pm[j],
                       vmin, vmax, cmap)
            style_ax(ax, pcoords)
            if r == 0:
                star = "*" if (fname == "cyl_classical.npz" and approx) else ""
                ax.set_title(title + star, fontsize=7, pad=2.5)
            ax.text(0.985, 0.96, f"{err:.3f}", transform=ax.transAxes,
                    fontsize=6.2, ha="right", va="top", color="k",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))

        cax = fig.add_subplot(gs[r, n_cols])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([vmin, vmax] if cmap == "viridis" else [vmin, 0.0, vmax])
        cb.ax.tick_params(labelsize=5.6, length=2, pad=1)
        cb.outline.set_linewidth(0.4)
        cb.ax.set_yticklabels(["0.0" if abs(t) < 0.05 else f"{t:.1f}"
                               for t in cb.get_ticks()])

    line1 = ("Cylinder wake, held-out Re 250 frame (val 300): 1% taps on "
             "$u_x$,$u_y$ (mesh 238/field, grid 800/field); $p$ UNOBSERVED "
             "-- POD infers it, IDW cannot (rel-$L_2$=1 by construction).")
    line2 = ("Numbers: per-field rel-$L_2$ (z-units); physical-unit fields, "
             "truth-anchored row scales; generative panels: one posterior "
             "sample (K=8, member 0).")
    if approx:
        line2 += " *classical taps: same-seed CPU draw (approximate)."
    fig.text(0.035, 0.988, line1, fontsize=6.3, color=INK_2, va="top")
    fig.text(0.035, 0.936, line2, fontsize=6.3, color=INK_2, va="top")

    save(fig, "recon_gallery_cylinder", pdf_dpi=300, png_dpi=220)


if __name__ == "__main__":
    main()
