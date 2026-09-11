"""All-baselines reconstruction galleries -- one figure per dataset.

Rows    : every reconstructed channel, each labelled observed / unobserved.
Columns : truth (+ that channel's sensors on observed rows), then one baseline
          per column in a fixed order shared by all three datasets --
          generative learned -> deterministic learned -> classical.

Every column conditions on the sensors its TABLE ROW scored:
  Kolmogorov  val frame 256, 655 vorticity sensors, all dumps idx_sum 22128955.
  Cylinder    val frame 300 (held-out Re 250), 238 sensors per field on u_x,u_y.
              Mesh methods (DMF-Gen, Senseiver, MLP-RBF, IDW, gappy POD) share
              the H100 fleet draw (idx_sum 5581141) -- MLP-RBF and the classical
              floors were re-dumped with those indices INJECTED, because the
              23,800-point mesh draw does not reproduce on GH200. Grid-native
              methods (SiT, latent FM, S3GM, Geo-FNO) receive the same count on
              the 400x200 grid export, as in the fleet (idx_sum 19084851).
  JHU         held-out cube, absolute frame 153, z-midplane; 19,531 sensors on
              each of U_x, U_z (qual draw, seed 103), injected identically.

Generative columns show ONE posterior sample (never an ensemble mean);
deterministic columns their single prediction. The per-panel number is that
displayed field's relative L2 error for that channel, in its dump's own
standardized units, over the whole field (the whole cube for JHU) -- a
single-sample error, so it sits above the table's ensemble-mean relative L2.

Color (dataviz reference palette): signed fields use a blue <-> red diverging
map around a neutral gray midpoint, symmetric about zero; one-sided fields a
single-hue blue ramp. Scales are anchored on truth (2-98th percentile), shared
along each row. A per-panel error table is written next to each figure.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pof_style import use_style  # noqa: E402

WT = HERE.parents[1]
FD = WT / "Save_TrainedModel_pof" / "field_dumps"
OUT = WT / "Paper" / "pof2026" / "figures"

INK, INK_2, MUTED, RULE = "#262624", "#52514e", "#898781", "#c9c8c2"
MID = "#f0efec"                                   # neutral diverging midpoint
BLUE_ARM = ["#0d366b", "#1c5cab", "#2a78d6", "#86b6ef", "#cde2fb"]   # dark -> light


def _mix(c, other, t):
    a, b = np.array(to_rgb(c)), np.array(to_rgb(other))
    return to_hex((1 - t) * a + t * b)


# The palette specifies the diverging pair (blue <-> red) and the midpoint but
# no red ramp, so the red arm is stepped from categorical red #e34948 to mirror
# the blue arm's lightness positions (light, light, base, dark, dark).
RED = "#e34948"
RED_ARM = [_mix(RED, "#ffffff", 0.78), _mix(RED, "#ffffff", 0.45), RED,
           _mix(RED, "#000000", 0.28), _mix(RED, "#000000", 0.52)]      # light -> dark
DIVERGING = LinearSegmentedColormap.from_list("div", BLUE_ARM + [MID] + RED_ARM)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
for cm in (DIVERGING, SEQUENTIAL):
    cm.set_bad(MID)

# (column title, npz, key) -- one order for every dataset
COLUMNS = {
    "kolmogorov": [
        ("DMF-Gen", "kolm_dmfgen.npz", "pred_sample"),
        ("SiT", "kolm_sit.npz", "pred_sample"),
        ("latent FM", "kolm_latent_fm.npz", "pred_sample"),
        ("S3GM", "kolm_s3gm.npz", "pred_sample"),
        ("Senseiver", "kolm_senseiver.npz", "pred_mean"),
        ("MLP-RBF", "kolm_mlprbf.npz", "pred_mean"),
        ("Geo-FNO", "kolm_geofno.npz", "pred_mean"),
        ("IDW", "kolm_classical.npz", "pred_idw"),
        ("gappy POD ($r{=}80$)", "kolm_classical.npz", "pred_gappy_pod_r80"),
    ],
    "cylinder": [
        ("DMF-Gen", "cyl_dmfgen.npz", "pred_sample"),
        ("SiT", "cyl_sit.npz", "pred_sample"),
        ("latent FM", "cyl_latent_fm.npz", "pred_sample"),
        ("S3GM", "cyl_s3gm.npz", "pred_sample"),
        ("Senseiver", "cyl_senseiver.npz", "pred_mean"),
        ("MLP-RBF", "cyl_mlprbf.npz", "pred_mean"),
        ("Geo-FNO", "cyl_geofno.npz", "pred_mean"),
        ("IDW", "cyl_classical.npz", "pred_idw"),
        ("gappy POD ($r{=}20$)", "cyl_classical.npz", "pred_gappy_pod_r20"),
    ],
    "jhu": [
        ("DMF-Gen", "jhu_dmfgen.npz", "pred_sample"),
        ("SiT", "jhu_sit.npz", "pred_sample"),
        ("latent FM", "jhu_latent_fm.npz", "pred_sample"),
        ("FNO3D", "jhu_fno3d.npz", "pred_sample"),
        ("Senseiver", "jhu_senseiver.npz", "pred_mean"),
        ("IDW", "jhu_classical.npz", "pred_idw"),
        ("gappy POD ($r{=}80$)", "jhu_classical.npz", "pred_gappy_pod_r80"),
    ],
}
# (label, channel index, observed?)
CHANNELS = {
    "kolmogorov": [(r"$\omega$", 0, True)],
    "cylinder": [(r"$u_x$", 0, True), (r"$u_y$", 1, True), (r"$p$", 2, False)],
    "jhu": [(r"$U_x$", 0, True), (r"$U_y$", 1, False), (r"$U_z$", 2, True), (r"$p$", 3, False)],
}
REFERENCE = {"kolmogorov": "kolm_classical.npz", "cylinder": "cyl_dmfgen.npz",
             "jhu": "jhu_sit.npz"}
PANEL_IN = {"kolmogorov": (1.02, 1.02), "cylinder": (1.24, 0.62), "jhu": (1.08, 1.08)}
TITLE = {
    "kolmogorov": ("2D Kolmogorov turbulence $256^2$ -- held-out trajectory, val frame 256",
                   "655 vorticity sensors (1%), identical draw in every column."),
    "cylinder": ("Cylinder wake -- held-out $Re{=}250$, val frame 300",
                 "238 sensors per field on $u_x,u_y$ (1%); pressure never observed. "
                 "Mesh methods share one draw; grid-native methods receive the same count on the grid export."),
    "jhu": ("3D isotropic turbulence $125^3$ -- held-out cube, frame 153, $z$-midplane",
            "19,531 sensors (1%) on each of $U_x$ and $U_z$; $U_y$ and $p$ never observed. "
            "Not shown (no field dumps on this host): CoNFiLD, S3GM, DeepONet++, Gen4Turb."),
}
CYL_ROI, CYL_R = (-3.0, 17.0, -5.0, 5.0), 0.5
G = 125


def load(name):
    p = FD / name
    return dict(np.load(p, allow_pickle=False)) if p.is_file() else None


def to_physical(d, arr, dataset):
    """JHU dumps are stored physical; the 2D dumps in their own z-units."""
    a = np.asarray(arr, dtype=np.float64)
    if dataset == "jhu":
        return a
    return a * np.asarray(d["norm_std"], dtype=np.float64).ravel() + \
        np.asarray(d["norm_mean"], dtype=np.float64).ravel()


def rel_l2_channel(d, key, j, dataset, ref):
    """Relative L2 for channel j in standardized units (table convention)."""
    if dataset == "jhu":
        m = np.asarray(ref["norm_mean"], np.float64)[j]
        s = np.asarray(ref["norm_std"], np.float64)[j]
        pr = (np.asarray(d[key], np.float64)[:, j] - m) / s
        tr = (np.asarray(d["truth"], np.float64)[:, j] - m) / s
    else:
        pr = np.asarray(d[key], np.float64)[:, j]
        tr = np.asarray(d["truth"], np.float64)[:, j]
    return float(np.linalg.norm(pr - tr) / (np.linalg.norm(tr) + 1e-12))


def row_scale(v):
    lo, hi = np.nanpercentile(v, [2, 98])
    if lo < 0 < hi and min(abs(lo), abs(hi)) / max(abs(lo), abs(hi)) > 0.10:
        m = float(max(abs(lo), abs(hi)))
        return -m, m, DIVERGING
    return float(lo), float(hi), SEQUENTIAL


def coords_of(d):
    return np.asarray(d["coords_raw"] if "coords_raw" in d else d["coords"], np.float64)


# ---------------------------------------------------------------- renderers --
def draw_kolm(ax, d, vals, vmin, vmax, cmap):
    c = coords_of(d)
    side = int(round(np.sqrt(vals.size)))
    a = vals.reshape(side, side)
    fast_x = np.ptp(c[:side, 0]) > np.ptp(c[:side, 1])
    img = a if fast_x else a.T
    return ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="nearest", rasterized=True)


_TRI = {}


def draw_cyl(ax, d, vals, vmin, vmax, cmap):
    c = coords_of(d)[:, :2]
    x0, x1, y0, y1 = CYL_ROI
    xs, ys = np.unique(c[:, 0]), np.unique(c[:, 1])
    if xs.size * ys.size == c.shape[0]:            # grid export
        fast_x = np.ptp(c[:xs.size, 0]) > 0
        img = vals.reshape((ys.size, xs.size) if fast_x else (xs.size, ys.size))
        img = img if fast_x else img.T
        X, Y = np.meshgrid(xs, ys)
        img = np.where(X ** 2 + Y ** 2 < CYL_R ** 2, np.nan, img)
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                       interpolation="nearest", rasterized=True)
    else:                                           # body-fitted mesh
        key = (c.shape[0], float(c[:, 0].sum()))
        if key not in _TRI:
            t = mtri.Triangulation(c[:, 0], c[:, 1])
            cx = c[t.triangles].mean(axis=1)
            t.set_mask((cx ** 2).sum(1) < CYL_R ** 2)   # Delaunay spans the body
            _TRI[key] = t
        im = ax.tripcolor(_TRI[key], vals, shading="gouraud", cmap=cmap,
                          vmin=vmin, vmax=vmax, rasterized=True)
    ax.add_patch(plt.Circle((0, 0), CYL_R, fc=MID, ec=MUTED, lw=0.5, zorder=3))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    return im


def draw_jhu(ax, d, vals, vmin, vmax, cmap):
    zc = G // 2
    sl = vals.reshape(G, G, G)[:, :, zc]
    return ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="nearest", rasterized=True)


DRAW = {"kolmogorov": draw_kolm, "cylinder": draw_cyl, "jhu": draw_jhu}


def sensor_xy(dataset, ref, j):
    idx = np.asarray(ref["sensor_indices"]).astype(np.int64)
    fid = np.asarray(ref["sensor_field_ids"]).astype(np.int64)
    pi = idx[fid == j]
    if dataset == "cylinder":
        c = coords_of(ref)
        return c[pi, 0], c[pi, 1], 2.2
    if dataset == "kolmogorov":
        c = coords_of(ref); side = int(round(np.sqrt(c.shape[0])))
        r, q = divmod(pi, side)
        fast_x = np.ptp(c[:side, 0]) > np.ptp(c[:side, 1])
        # 655 sensors on a ~1 in panel: markers must stay near-point-sized or
        # they hide the field they are meant to annotate
        return (q, r, 0.45) if fast_x else (r, q, 0.45)
    zc = G // 2                                     # jhu: sensors in the shown plane
    pi = pi[pi % G == zc]
    return pi // (G * G), (pi // G) % G, 2.4


def style(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color(RULE)


# ------------------------------------------------------------------ figure --
def make(dataset):
    use_style()
    ref = load(REFERENCE[dataset])
    if ref is None:
        raise SystemExit(f"[{dataset}] reference dump {REFERENCE[dataset]} missing")
    cols, chans = COLUMNS[dataset], CHANNELS[dataset]
    pw, ph = PANEL_IN[dataset]
    n_r, n_c = len(chans), 1 + len(cols)
    lab_w, cb_w, top, bot = 0.78, 0.42, 0.62, 0.12
    W = lab_w + n_c * pw + 0.05 * n_c + cb_w
    H = top + n_r * ph + 0.06 * n_r + bot
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(n_r, n_c + 1, width_ratios=[1] * n_c + [0.07],
                          left=lab_w / W, right=1 - 0.30 / W,
                          top=1 - top / H, bottom=bot / H, wspace=0.05, hspace=0.07)
    cache, table = {}, []
    truth_phys = to_physical(ref, ref["truth"], dataset)
    for r, (lab, j, observed) in enumerate(chans):
        vmin, vmax, cmap = row_scale(truth_phys[:, j])
        # --- truth (+ sensors on observed rows) ---
        ax = fig.add_subplot(gs[r, 0])
        im = DRAW[dataset](ax, ref, truth_phys[:, j], vmin, vmax, cmap)
        if observed:
            sx, sy, ms = sensor_xy(dataset, ref, j)
            ax.scatter(sx, sy, s=ms, c="white", edgecolors=INK,
                       linewidths=0.18 if dataset == "kolmogorov" else 0.35,
                       marker="o", rasterized=True, zorder=4)
        style(ax)
        if r == 0:
            ax.set_title("truth + sensors", fontsize=7.5, color=INK, pad=3)
        # row label: channel, then observed / unobserved
        fig.text(0.10 / W, ax.get_position().y0 + ax.get_position().height * 0.5,
                 lab, fontsize=9, color=INK, va="center", ha="left")
        fig.text(0.10 / W, ax.get_position().y0 + ax.get_position().height * 0.5 - 0.13 / H,
                 "observed" if observed else "unobserved", fontsize=6.6,
                 color=INK_2, va="top", ha="left",
                 style="normal" if observed else "italic")
        # --- one baseline per column ---
        for c_i, (title, fname, key) in enumerate(cols, start=1):
            ax = fig.add_subplot(gs[r, c_i])
            if fname not in cache:
                cache[fname] = load(fname)
            d = cache[fname]
            if d is None or key not in d:
                ax.set_facecolor("#f6f5f2")
                ax.text(0.5, 0.5, "not dumped", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6.4, color=MUTED)
                if dataset == "cylinder":
                    ax.set_xlim(*CYL_ROI[:2]); ax.set_ylim(*CYL_ROI[2:]); ax.set_aspect("equal")
                style(ax)
                if r == 0:
                    ax.set_title(title, fontsize=7.5, color=MUTED, pad=3)
                table.append({"channel": lab, "method": title, "rel_l2": None})
                continue
            vals = to_physical(d, d[key], dataset)[:, j]
            DRAW[dataset](ax, d, vals, vmin, vmax, cmap)
            err = rel_l2_channel(d, key, j, dataset, ref)
            table.append({"channel": lab.strip("$"), "method": title.split(" (")[0],
                          "observed": observed, "rel_l2": round(err, 4)})
            ax.text(0.97, 0.05, f"{err:.2f}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=6.3, color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
            style(ax)
            if r == 0:
                ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        cax = fig.add_subplot(gs[r, n_c])
        b = cax.get_position()
        cax.set_position([b.x0, b.y0 + 0.14 * b.height, b.width, 0.72 * b.height])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([vmin, 0.0, vmax] if cmap is DIVERGING else [vmin, vmax])
        cb.ax.tick_params(labelsize=5.8, length=2, pad=1, colors=INK_2)
        cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(RULE)
    t1, t2 = TITLE[dataset]
    fig.text(lab_w / W, 1 - 0.14 / H, t1, fontsize=8.5, color=INK, va="top")
    fig.text(lab_w / W, 1 - 0.32 / H, t2 + "  Number in each panel: single-sample "
             "relative $L_2$ for that channel (standardized units).",
             fontsize=6.3, color=INK_2, va="top")
    stem = f"gallery_{dataset}_allbaselines"
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    fig.savefig(OUT / f"{stem}.pdf", dpi=300)
    plt.close(fig)
    with open(OUT / f"{stem}_rell2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["channel", "method", "observed", "rel_l2"])
        w.writeheader(); w.writerows(table)
    done = sum(1 for t in table if t["rel_l2"] is not None)
    print(f"[{dataset}] wrote {stem}.png/.pdf  ({done}/{len(table)} panels, "
          f"{W:.1f}x{H:.1f} in)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["kolmogorov", "cylinder", "jhu"])
    for ds in ap.parse_args().datasets:
        make(ds)
