"""PoF figure: FireBench wildfire-LES posterior, conditioned on wind sensors only.

Rows: u (observed) and theta (unobserved) on a vertical slice through the fire
front; rho_f (unobserved) on the horizontal fuel-bed plane of maximum fuel
variance -- a vertical slice barely intersects the fuel bed and renders its
noise floor (see src/replot_firebench.py in the main checkout).
Columns: truth / single posterior sample (truth-anchored scale) /
posterior std / |posterior mean - truth| (std and |error| share one scale per
row so the std-tracks-error claim is directly readable).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pof_style import use_style, save, MAIN, FULL_W, INK_2

NX, NY, NZ = 152, 126, 192
SRC = MAIN / "Paper" / "iclr2027" / "figures"

use_style()
d = np.load(SRC / "qual_firebench.npz")
truth, mean, std, sample0 = d["truth"], d["mean"], d["std"], d["sample0"]
yc = int(d["yc"])
err = np.abs(mean - truth)
unc = json.load(open(SRC / "uncertainty_firebench.json"))


def cube(a, j):
    return a[:, j].reshape(NX, NY, NZ)


# fuel bed: horizontal plane where the truth fuel density varies most
zc = int(np.argmax(cube(truth, 4).var(axis=(0, 1))))

# (channel index, json key, math label, observed?, plane, field cmap)
rows = [
    (0, "u", r"$u$", True, "vert", "RdBu_r"),
    (3, "theta", r"$\theta$", False, "vert", "inferno"),
    (4, "rho_f", r"$\rho_f$", False, "horiz", "viridis"),
]
plane_lbl = {"vert": f"vertical slice $y={yc}$",
             "horiz": f"fuel-bed plane $z={zc}$"}
cols = ["truth", "posterior sample", "posterior std", "|error|"]

fig, axes = plt.subplots(3, 4, figsize=(FULL_W, 4.35), constrained_layout=True)
for r, (j, key, lab, obs, plane, cmap) in enumerate(rows):
    def sl(a):
        c = cube(a, j)
        return c[:, yc, :] if plane == "vert" else c[:, :, zc]

    t, s1, sd, e = sl(truth), sl(sample0), sl(std), sl(err)
    p1, p99 = np.percentile(t, [1, 99])
    if cmap == "RdBu_r":                       # signed field, neutral center
        v = max(abs(p1), abs(p99))
        vmin, vmax = -v, v
    else:
        vmin, vmax = p1, p99
    smax = max(np.percentile(sd, 99), np.percentile(e, 99))

    panels = [(t, cmap, vmin, vmax), (s1, cmap, vmin, vmax),
              (sd, "Blues", 0.0, smax), (e, "Blues", 0.0, smax)]
    ims = []
    for c, (arr, cm, lo, hi) in enumerate(panels):
        ax = axes[r, c]
        ims.append(ax.imshow(arr.T, origin="lower", cmap=cm, vmin=lo, vmax=hi,
                             aspect="auto", interpolation="nearest"))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(cols[c])
    axes[r, 0].set_ylabel(f"{lab} ({'observed' if obs else 'unobserved'})\n"
                          f"{plane_lbl[plane]}", fontsize=7.5)

    ratio = unc[key]["std_hot"] / unc[key]["std_cold"]
    axes[r, 2].text(0.03, 0.96, f"std front/ambient\n$\\approx${ratio:.1f}$\\times$",
                    transform=axes[r, 2].transAxes, ha="left", va="top",
                    fontsize=6.3, color="#0b0b0b",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    for group, im in [((0, 1), ims[0]), ((2, 3), ims[2])]:
        cb = fig.colorbar(im, ax=[axes[r, c] for c in group],
                          fraction=0.05, pad=0.015, aspect=14)
        cb.ax.tick_params(labelsize=5.8)
        cb.locator = MaxNLocator(4)
        cb.update_ticks()

fig.suptitle("Wildfire LES, held-out snapshot: posterior from wind sensors only "
             f"($K$=16 ensemble);  corr(std, |err|) = {unc['corr_std_err']:.2f}",
             fontsize=9)
fig.text(0.0, -0.015,
         "|error| = |posterior mean $-$ truth|; sample shares the truth color "
         "scale; std and |error| share one scale per row.",
         fontsize=6.3, color=INK_2)
save(fig, "qual_firebench")
print("fuel-bed plane z =", zc)
for j, key, lab, *_ in rows:
    rel = np.linalg.norm(mean[:, j] - truth[:, j]) / np.linalg.norm(truth[:, j])
    print(f"  {key}: rel L2 (mean) = {rel:.4f}, "
          f"std hot/cold = {unc[key]['std_hot']/unc[key]['std_cold']:.1f}x")
