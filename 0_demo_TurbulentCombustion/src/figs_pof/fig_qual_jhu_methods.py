"""PoF qualitative centerpiece: cross-method comparison on JHU cube-3, snapshot 3.

Rows: Ux (observed channel) and Uy (unobserved channel), z-midplane slice.
Columns: DNS truth / DMF-Gen single sample / DMF-Gen posterior mean /
latent-FM single sample / DMF-Gen posterior std (K=16).

Data provenance
---------------
- DMF-Gen fields: qual_jhu.npz (snapshot 3, 1% sensors on Ux and Uz, NFE 4).
- Latent-FM sample: spectra_fields.npz key ``latent_fm_3`` -- the only
  latent-FM field available on the SAME snapshot 3.  The ens_latentfm_s0/s12
  ensembles are snapshots 0 and 12 (their ``truth`` does not match
  qual_jhu's; verified corr about 0.6 / 0.2 per channel), so a sample from
  them next to snapshot-3 truth would be misleading.
- Sensor sites recovered as dist == 0 points of qual_jhu's nearest-sensor
  distance field: the union of the Ux and Uz sensor sets (they are drawn
  independently per channel and cannot be separated after the fact).

Message: both generative samples stay sharp; the DMF-Gen posterior mean
collapses to near-constant on the unobserved channel (correct posterior
behavior under 1% cross-channel sensing), while the latent-FM sample carries
more mid-scale texture on that channel.  The std map is near-uniform and
large on Uy, small on Ux.
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
ZC = GRID // 2

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "font.family": "sans-serif", "pdf.fonttype": 42,
})


def main():
    q = np.load(FIGDIR / "qual_jhu.npz")
    sp = np.load(FIGDIR / "spectra_fields.npz")
    truth, mean, sample0, std = q["truth"], q["mean"], q["sample0"], q["std"]
    lfm = sp["latent_fm_3"]
    assert np.allclose(sp["truth_3"], truth), "latent_fm_3 is not on snapshot 3"

    coords = q["coords"]
    c3 = coords.reshape(GRID, GRID, GRID, 3)
    # index axes are (x, y, z): column k varies along reshape axis k
    assert np.all(np.diff(c3[:, 0, 0, 0]) > 0)
    assert np.all(np.diff(c3[0, :, 0, 1]) > 0)
    assert np.all(np.diff(c3[0, 0, :, 2]) > 0)

    # sensor sites (union of the Ux and Uz draws) that lie in the slice
    idx = np.round(coords * (GRID - 1)).astype(int)
    s_all = np.where(q["dist"] == 0.0)[0]
    s_sl = s_all[idx[s_all, 2] == ZC]
    sx, sy = idx[s_sl, 0], idx[s_sl, 1]

    def sl(a, j):
        return a[:, j].reshape(GRID, GRID, GRID)[:, :, ZC]

    rows = [(0, r"$u_x$", "observed"), (1, r"$u_y$", "unobserved")]
    cols = ["dns truth", "dmf-gen sample", "dmf-gen mean",
            "latent-fm sample", "dmf-gen std"]

    fig = plt.figure(figsize=(7.0, 3.15))
    # cols: 4 shared-scale field panels | cbar | spacer | std panel | cbar
    gs = fig.add_gridspec(2, 8,
                          width_ratios=[1, 1, 1, 1, 0.055, 0.30, 1, 0.055],
                          left=0.055, right=0.935, top=0.90, bottom=0.015,
                          wspace=0.08, hspace=0.10)

    tag = iter("abcdefghij")
    for r, (j, name, obs) in enumerate(rows):
        panels = [sl(truth, j), sl(sample0, j), sl(mean, j), sl(lfm, j)]
        sd = sl(std, j)
        # symmetric limits from truth 2-98 percentiles (diverging map centered on 0)
        p2, p98 = np.percentile(panels[0], [2, 98])
        vlim = float(max(abs(p2), abs(p98)))
        im_f = None
        for c, arr in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            im_f = ax.imshow(arr.T, origin="lower", cmap="RdBu_r",
                             vmin=-vlim, vmax=vlim, interpolation="nearest",
                             rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.55")
            if r == 0:
                ax.set_title(cols[c], fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(f"{name} ({obs})", fontsize=8)
                if r == 0:
                    ax.scatter(sx, sy, s=0.6, c="k", marker=".",
                               linewidths=0, rasterized=True)
                    ax.text(0.03, 0.03, "sensors", transform=ax.transAxes,
                            fontsize=6, color="k", va="bottom",
                            bbox=dict(fc="white", ec="none", alpha=0.75,
                                      pad=0.8))
            ax.text(0.03, 0.97, f"({next(tag)})", transform=ax.transAxes,
                    fontsize=7, va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
        cax = fig.add_subplot(gs[r, 4])
        cb = fig.colorbar(im_f, cax=cax)
        cb.set_ticks([-vlim, 0.0, vlim])
        cb.set_ticklabels([f"{-vlim:.1f}", "0", f"{vlim:.1f}"])
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.outline.set_linewidth(0.4)

        ax = fig.add_subplot(gs[r, 6])
        im_s = ax.imshow(sd.T, origin="lower", cmap="viridis",
                         vmin=0, vmax=float(np.percentile(sd, 99)),
                         interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.55")
        if r == 0:
            ax.set_title(cols[4], fontsize=8, pad=3)
        ax.text(0.03, 0.97, f"({next(tag)})", transform=ax.transAxes,
                fontsize=7, va="top", color="w",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=0.8))
        cax = fig.add_subplot(gs[r, 7])
        cb = fig.colorbar(im_s, cax=cax)
        smax = float(np.percentile(sd, 99))
        cb.set_ticks([0.0, smax / 2, smax])
        cb.set_ticklabels(["0", f"{smax/2:.2f}", f"{smax:.2f}"])
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.outline.set_linewidth(0.4)

    fig.suptitle("JHU isotropic turbulence, held-out cube, z-midplane "
                 "(1% sensors on $u_x$, $u_z$; K=16, NFE 4)",
                 fontsize=8, y=0.985)
    fig.savefig(OUT / "qual_jhu_methods.pdf", dpi=300)
    fig.savefig(OUT / "qual_jhu_methods.png", dpi=200)
    print("wrote", OUT / "qual_jhu_methods.pdf")
    for f in ["qual_jhu_methods.pdf", "qual_jhu_methods.png"]:
        p = OUT / f
        print(f"  {f}: {p.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
