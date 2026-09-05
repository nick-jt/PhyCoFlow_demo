"""PoF uncertainty diagnostics for DMF-Gen on JHU cube-3, snapshot 3.

Four panels, recomputed from the saved arrays in qual_jhu.npz (the companion
uncertainty_jhu.json holds only summary scalars, so the binned curves are
rebuilt here exactly as in src/qualitative_jhu.py):

  (a) spread reliability -- binned RMS error vs predicted std per field, with
      the ideal diagonal; marker opacity is count-weighted (equal-width bins);
      corr(std, |err|) over all points and fields stated in an annotation.
  (b) rank histogram of truth within the K=16 ensemble (subsampled points),
      with the uniform reference and its 99% multinomial band.
  (c) empirical vs nominal central-interval coverage.
  (d) posterior std vs distance to the nearest sensor, for the observed Ux
      and unobserved Uy, at 1% and 0.1% sensor density: flat at 1% (every
      site is within a few cells of a sensor -- an honest null), rising for
      Ux once 0.1% sensing leaves real gaps.

Line colors are the validated categorical palette (see check_palette.py);
marker shapes double-encode field identity (relief for the low-contrast
aqua/yellow slots on white).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = Path("/home/ntricard/generative_reconstruction/temp/"
            "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion")
FIGDIR = MAIN / "Paper/iclr2027/figures"
OUT = Path(__file__).resolve().parents[2] / "Paper/pof2026/figures"
OUT.mkdir(parents=True, exist_ok=True)

GRID = 125
COLORS = {"Ux": "#2a78d6", "Uy": "#eb6834", "Uz": "#1baf7a", "p": "#eda100"}
MARKERS = {"Ux": "o", "Uy": "s", "Uz": "^", "p": "D"}
GRIDCOL = "#e1e0d9"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "font.family": "sans-serif", "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def style(ax):
    ax.grid(True, lw=0.5, color=GRIDCOL, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def latexname(nm):
    return {"Ux": r"$u_x$", "Uy": r"$u_y$", "Uz": r"$u_z$", "p": r"$p$"}[nm]


def main():
    q = np.load(FIGDIR / "qual_jhu.npz")
    truth, mean, std, err = q["truth"], q["mean"], q["std"], q["err"]
    dist, std_sp, dist_sp = q["dist"], q["std_sp"], q["dist_sp"]
    ens = q["ens_small"]                       # (K, Nsub, 4), Nsub = every 37th pt
    tsub = truth[::37]
    names = [str(n) for n in q["names"]]
    K = ens.shape[0]

    corr = float(np.corrcoef(std.ravel(), err.ravel())[0, 1])

    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.05), dpi=200,
                            constrained_layout=True)
    fig.get_layout_engine().set(w_pad=5 / 72, h_pad=2 / 72)

    # (a) spread reliability ------------------------------------------------
    ax = axs[0]
    style(ax)
    for j, nm in enumerate(names):
        sd, ee = std[:, j], err[:, j]
        lo, hi = np.percentile(sd, [1, 99])
        edges = np.linspace(lo, hi, 11)
        xs, ys, ws = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (sd >= a) & (sd < b)
            n = int(m.sum())
            if n > 200:
                xs.append(sd[m].mean())
                ys.append(np.sqrt((ee[m] ** 2).mean()))
                ws.append(n)
        xs, ys, ws = map(np.asarray, (xs, ys, ws))
        ax.plot(xs, ys, "-", color=COLORS[nm], lw=1.0, zorder=3)
        alpha = 0.30 + 0.70 * ws / ws.max()
        ax.scatter(xs, ys, s=9, marker=MARKERS[nm], color=COLORS[nm],
                   alpha=alpha, linewidths=0, zorder=4,
                   label=latexname(nm))
    lim = (0.03, 2.0)
    ax.plot(lim, lim, ls=(0, (3, 2)), color="0.45", lw=0.8, zorder=2,
            label="ideal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("predicted std")
    ax.set_ylabel("binned RMS error")
    ax.set_title("(a) spread reliability", loc="left")
    ax.legend(frameon=False, handlelength=1.4, borderaxespad=0.2,
              labelspacing=0.25, loc="lower right")
    ax.text(0.04, 0.96, f"corr(std, |err|) = {corr:.2f}",
            transform=ax.transAxes, fontsize=6.5, va="top")

    # (b) rank histogram ----------------------------------------------------
    ax = axs[1]
    style(ax)
    n = tsub.shape[0]
    p0 = 1.0 / (K + 1)
    half = 2.576 * np.sqrt(p0 * (1 - p0) / n)
    ax.axhspan(p0 - half, p0 + half, color="0.85", zorder=1)
    ax.axhline(p0, color="0.45", lw=0.8, zorder=2)
    for j, nm in enumerate(names):
        below = (ens[:, :, j] < tsub[:, j][None]).sum(0)
        h = np.bincount(below, minlength=K + 1).astype(float)
        h /= h.sum()
        ax.step(np.arange(K + 1), h, where="mid", color=COLORS[nm], lw=1.0,
                zorder=3)
    ax.set_xlabel("rank of truth in ensemble")
    ax.set_ylabel("frequency")
    ax.set_xlim(-0.5, K + 0.5)
    ax.set_ylim(0, None)
    ax.set_title("(b) rank histogram", loc="left")
    ax.text(0.97, 0.97, "gray: uniform,\n99% band",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="right")

    # (c) coverage curve ----------------------------------------------------
    ax = axs[2]
    style(ax)
    noms = np.linspace(0.05, 0.95, 19)
    for j, nm in enumerate(names):
        e, t = ens[:, :, j], tsub[:, j]
        cov = []
        for qn in noms:
            lo = np.quantile(e, (1 - qn) / 2, axis=0)
            hi = np.quantile(e, 1 - (1 - qn) / 2, axis=0)
            cov.append(float(((t >= lo) & (t <= hi)).mean()))
        ax.plot(noms, cov, "-", color=COLORS[nm], lw=1.0, zorder=3,
                marker=MARKERS[nm], ms=2.4, markevery=3)
    ax.plot([0, 1], [0, 1], ls=(0, (3, 2)), color="0.45", lw=0.8, zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("(c) coverage", loc="left")

    # (d) std vs distance to nearest sensor ---------------------------------
    ax = axs[3]
    style(ax)
    for dd, ss, ls_, dens, alpha in ((dist, std, "-", "1%", 1.0),
                                     (dist_sp, std_sp, (0, (3, 2)), "0.1%", 0.6)):
        edges = np.quantile(dd, np.linspace(0, 1, 13))
        for j in (0, 1):
            nm = names[j]
            xs, ys = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (dd >= a) & (dd < b)
                if m.sum() > 200:
                    xs.append(dd[m].mean() * (GRID - 1))
                    ys.append(ss[m, j].mean())
            ax.plot(xs, ys, ls=ls_, color=COLORS[nm], lw=1.0, alpha=alpha,
                    marker=MARKERS[nm], ms=2.4, zorder=3,
                    label=f"{latexname(nm)}, {dens}")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1"])
    ax.set_xlabel("distance to sensor (cells)")
    ax.set_ylabel("posterior std")
    ax.set_title("(d) std vs distance", loc="left")
    ax.legend(frameon=False, handlelength=1.3, borderaxespad=0.1,
              handletextpad=0.4, labelspacing=0.25, loc="center right",
              bbox_to_anchor=(1.0, 0.42), fontsize=6)
    ax.text(0.03, 0.74, "flat at 1%: no site is more\nthan ~6 cells from a sensor",
            transform=ax.transAxes, fontsize=6, va="top")

    fig.savefig(OUT / "uncertainty_jhu.pdf")
    fig.savefig(OUT / "uncertainty_jhu.png")
    print("wrote", OUT / "uncertainty_jhu.pdf")
    for f in ["uncertainty_jhu.pdf", "uncertainty_jhu.png"]:
        p = OUT / f
        print(f"  {f}: {p.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
