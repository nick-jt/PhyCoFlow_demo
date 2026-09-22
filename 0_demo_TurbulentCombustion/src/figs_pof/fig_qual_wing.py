"""PoF figure: transonic-wing volume reconstruction from surface sensors.

Midspan-slice gallery {truth, posterior mean, posterior std} for Ux and Cp.
The 4e5-node triangulations are rasterized (vector PDF was 22 MB); the
instrumented wing surface -- where the 512 pressure taps + 128 shear gauges
live -- is marked on the truth panels via the saved distance-to-skin field.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pof_style import use_style, save, MAIN, FULL_W, INK_2

SRC = MAIN / "Paper" / "iclr2027" / "figures"

use_style()
d = np.load(SRC / "qual_wing.npz")
truth, mean, std = d["truth"], d["mean"], d["std"]
xyz, band, dist = d["coords"], d["band"], d["dist"]
names = [str(n) for n in d["names"]]
unc = json.load(open(SRC / "uncertainty_wing.json"))

xs, zs = xyz[band, 0], xyz[band, 2]
tri = mtri.Triangulation(xs, zs)

# (The wing "interior" contains real mesh nodes with near-zero flow, so the
# triangulation there renders actual data -- no masking needed.)

# instrumented surface: skin nodes inside the midspan band (subsampled)
surf = dist[band] < 2e-4
rng = np.random.default_rng(0)
idx = np.flatnonzero(surf)
idx = rng.choice(idx, size=min(700, idx.size), replace=False)

rows = [(0, "Ux", r"$U_x$"), (3, "Cp", r"$C_p$")]
cols = ["truth", "posterior mean", "posterior std"]

fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.55), constrained_layout=True)
for r, (j, key, lab) in enumerate(rows):
    p1, p99 = np.percentile(truth[band, j], [1, 99])
    # both fields are signed with unequal arms: diverging map, neutral at 0
    fnorm = TwoSlopeNorm(vmin=min(p1, -0.1), vcenter=0.0, vmax=p99)
    snorm = Normalize(0.0, np.percentile(std[band, j], 99))

    panels = [(truth[band, j], "RdBu_r", fnorm),
              (mean[band, j], "RdBu_r", fnorm),
              (std[band, j], "Blues", snorm)]
    ims = []
    for c, (arr, cm, norm) in enumerate(panels):
        ax = axes[r, c]
        ims.append(ax.tripcolor(tri, arr, cmap=cm, norm=norm,
                                shading="gouraud", rasterized=True))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        if r == 0:
            ax.set_title(cols[c])
    axes[r, 0].set_ylabel(f"{lab} (unobserved volume)", fontsize=7.5)

    # sensor surface on the truth panel (explained in the footnote)
    axes[r, 0].plot(xs[idx], zs[idx], ".", ms=1.0, c="k", ls="none")

    near, far = unc[key]["std_near_surface"], unc[key]["std_far_field"]
    axes[r, 2].text(0.03, 0.03,
                    f"std: {near:.2f} near skin $\\to$ {far:.2f} far field "
                    f"({near/far:.1f}$\\times$)",
                    transform=axes[r, 2].transAxes, ha="left", va="bottom",
                    fontsize=6.3,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    for group, im in [((0, 1), ims[0]), ((2,), ims[2])]:
        cb = fig.colorbar(im, ax=[axes[r, c] for c in group],
                          fraction=0.05, pad=0.015, aspect=14)
        cb.ax.tick_params(labelsize=5.8)
        cb.locator = MaxNLocator(4)
        cb.update_ticks()

fig.suptitle("Transonic wing, held-out geometry: volume reconstruction from 512 "
             "surface pressure taps + 128 shear gauges (midspan slice);  "
             f"corr(std, |err|) = {unc['corr_std_err']:.2f}", fontsize=9)
fig.text(0.0, -0.015,
         "Sensors live only on the wing skin (dots); posterior std contracts "
         "onto the instrumented surface and widens into the wake.",
         fontsize=6.3, color=INK_2)
save(fig, "qual_wing", pdf_dpi=300)
for j, key, _ in rows:
    print(f"  {key}: std near/far = {unc[key]['std_near_surface']:.3f}/"
          f"{unc[key]['std_far_field']:.3f}")
