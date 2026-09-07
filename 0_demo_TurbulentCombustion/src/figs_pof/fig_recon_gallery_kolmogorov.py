"""All-baselines reconstruction gallery: 2D Kolmogorov flow, ONE fixed frame.

Single row of panels on the SAME held-out frame (val 256 = absolute 2816, a
canonical eval frame, si=20) with IDENTICAL canonical sensors (655 = 1% on
vorticity, torch.manual_seed(0*777+256)):

    truth+sensors | DMF-Gen | SiT | latent-FM | Senseiver | MLP-RBF* | S3GM*
                  | IDW | gappy-POD (r80)

Learned panels come from Save_TrainedModel_pof/field_dumps/kolm_<model>.npz
(src/dump_kolm_gallery.sh); classical panels from kolm_classical.npz
(src/dump_classical_gallery.py).  Panels whose npz has not landed yet render
as labeled empty slots, so the script is RE-RUNNABLE to fill them in
(* = no driver leg exists yet for MLP-RBF / S3GM).

Generative panels show ONE posterior sample (sample k=0 of the fleet-eval
ensemble); Senseiver its deterministic prediction.  Per-panel rel-L2 is the
displayed field vs truth in z-score units (the table's metric convention);
display units are z-scores too (all kolm dumps share the same train stats).
Shared truth-anchored symmetric color scale, rasterized fields, 7in, 8pt.
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
SIDE = 256

# (panel title, npz file, array key)  -- None file = known-pending slot
PANELS = [
    ("DMF-Gen (ours)", "kolm_dmfgen.npz", "pred_sample"),
    ("SiT", "kolm_sit.npz", "pred_sample"),
    ("latent-FM", "kolm_latent_fm.npz", "pred_sample"),
    ("Senseiver", "kolm_senseiver.npz", "pred_mean"),
    ("MLP-RBF", None, None),          # no eval-driver leg yet
    ("S3GM", None, None),             # no eval-driver leg yet
    ("IDW", "kolm_classical.npz", "pred_idw"),
    ("gappy POD $r$80", "kolm_classical.npz", "pred_gappy_pod_r80"),
]


def to_grid(arr_1d: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """[N] -> [SIDE, SIDE] image with x horizontal, y vertical (origin lower)."""
    a = arr_1d.reshape(SIDE, SIDE)
    fast_x = np.ptp(coords[:SIDE, 0]) > np.ptp(coords[:SIDE, 1])
    # rows of `a` follow the slow index; imshow wants [y, x]
    return a if fast_x else a.T


def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) /
                 (np.linalg.norm(true) + 1e-12))


def main() -> None:
    use_style()
    cls_path = DUMPS / "kolm_classical.npz"
    if not cls_path.is_file():
        raise SystemExit(f"need {cls_path} first (dump_classical_gallery.py)")
    cls = np.load(cls_path, allow_pickle=False)
    cls_meta = json.loads(str(cls["meta"]))
    coords = np.asarray(cls["coords_raw"], dtype=np.float64)
    truth = np.asarray(cls["truth"])[:, 0]
    approx_sensors = "APPROX" in cls_meta.get("sensor_source", "")

    p2, p98 = np.percentile(truth, [2, 98])
    vlim = float(max(abs(p2), abs(p98)))

    n_cols = 1 + len(PANELS)               # truth + 8 method slots
    fig = plt.figure(figsize=(7.0, 1.12))
    gs = fig.add_gridspec(1, n_cols + 1,
                          width_ratios=[1.0] * n_cols + [0.07],
                          left=0.005, right=0.955, top=0.615, bottom=0.02,
                          wspace=0.07)

    def style_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, SIDE - 0.5)
        ax.set_ylim(-0.5, SIDE - 0.5)
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_linewidth(0.4)
            s.set_color("0.55")

    # ---- truth + sensors ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(to_grid(truth, coords), origin="lower", cmap="RdBu_r",
                   vmin=-vlim, vmax=vlim, interpolation="nearest",
                   rasterized=True)
    s_idx = np.asarray(cls["sensor_indices"])
    si, sj = divmod(s_idx, SIDE)
    fast_x = np.ptp(coords[:SIDE, 0]) > np.ptp(coords[:SIDE, 1])
    sx, sy = (sj, si) if fast_x else (si, sj)
    ax.scatter(sx, sy, s=0.5, c="k", marker=".", linewidths=0,
               rasterized=True)
    style_ax(ax)
    sens_tag = "sens.*" if approx_sensors else "sens."
    ax.set_title(f"truth+{sens_tag}", fontsize=7, pad=2.5)

    # ---- method panels -----------------------------------------------------
    for c, (title, fname, key) in enumerate(PANELS, start=1):
        ax = fig.add_subplot(gs[0, c])
        pending = fname is None or not (DUMPS / fname).is_file()
        if not pending:
            d = np.load(DUMPS / fname, allow_pickle=False)
            if key not in d.files:
                pending = True
        if pending:
            ax.set_facecolor("#f0efe9")
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5, color=MUTED)
            style_ax(ax)
            ax.set_title(title, fontsize=7, pad=2.5, color=MUTED)
            continue
        pred = np.asarray(d[key])[:, 0]
        tr = np.asarray(d["truth"])[:, 0]
        err = rel_l2(pred, tr)
        ax.imshow(to_grid(pred, coords), origin="lower", cmap="RdBu_r",
                  vmin=-vlim, vmax=vlim, interpolation="nearest",
                  rasterized=True)
        style_ax(ax)
        star = "*" if (fname == "kolm_classical.npz" and approx_sensors) else ""
        ax.set_title(title + star, fontsize=7, pad=2.5)
        ax.text(0.965, 0.955, f"{err:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.2, color="k",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))

    cax = fig.add_subplot(gs[0, n_cols])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-vlim, 0.0, vlim])
    cb.set_ticklabels([f"{-vlim:.0f}", "0", f"{vlim:.0f}"])
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.outline.set_linewidth(0.4)

    line1 = ("Kolmogorov flow, held-out frame (val 256), 655 sensors (1%) "
             "on vorticity (z-units, truth-anchored scale).")
    line2 = ("Numbers: rel-$L_2$; generative panels: one posterior sample "
             "(K=8 eval, member 0).")
    if approx_sensors:
        line2 += " *classical: same-seed CPU sensor draw (approximate)."
    fig.text(0.005, 0.985, line1, fontsize=6.3, color=INK_2, va="top")
    fig.text(0.005, 0.885, line2, fontsize=6.3, color=INK_2, va="top")

    save(fig, "recon_gallery_kolmogorov", pdf_dpi=300, png_dpi=220)


if __name__ == "__main__":
    main()
