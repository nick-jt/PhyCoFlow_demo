"""Re-render the wing qualitative figure from saved arrays, rasterized.

The vector triangulation over 4e5 mesh nodes produced a 22 MB PDF; the
panels are images in all but name, so they are rasterized at 300 dpi while
axes, labels and colorbars stay vector.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

OUT = Path("../Paper/iclr2027/figures")
d = np.load(OUT / "qual_wing.npz")
truth, mean, std, sample0 = d["truth"], d["mean"], d["std"], d["sample0"]
xyz, band = d["coords"], d["band"]
names = [str(n) for n in d["names"]]

xs, zs = xyz[band, 0], xyz[band, 2]
tri = mtri.Triangulation(xs, zs)
cols = ["truth", "posterior mean", "single sample", "posterior std"]
show = [3, 0, 2]
fig, axes = plt.subplots(len(show), 4, figsize=(12.0, 2.3 * len(show)), dpi=200)
for r, j in enumerate(show):
    vals = [truth[band, j], mean[band, j], sample0[band, j], std[band, j]]
    vmin, vmax = np.percentile(vals[0], [2, 98])
    for c, arr in enumerate(vals):
        ax = axes[r, c]
        cmap = "viridis" if c == 3 else "RdBu_r"
        vr = (0, np.percentile(arr, 99)) if c == 3 else (vmin, vmax)
        im = ax.tripcolor(tri, arr, cmap=cmap, vmin=vr[0], vmax=vr[1],
                          shading="gouraud", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        if r == 0:
            ax.set_title(cols[c], fontsize=10)
        if c == 0:
            ax.set_ylabel(names[j], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=6)
fig.suptitle("Wing, held-out geometry: volumetric reconstruction from 512 surface "
             "pressure taps + 384 shear gauges (midspan slice)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(OUT / "qual_wing.pdf", dpi=300)
fig.savefig(OUT / "qual_wing.png", dpi=150)
print("rewrote qual_wing.pdf")
