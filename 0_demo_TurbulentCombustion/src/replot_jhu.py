"""Re-render the JHU qualitative figure from saved arrays (2-row main-text version).

Keeps the contrast that carries the message -- one observed channel and one
channel no sensor constrains -- and drops the third row, which repeats it.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = 125
OUT = Path("../Paper/iclr2027/figures")
d = np.load(OUT / "qual_jhu.npz")
truth, mean, std, sample0 = d["truth"], d["mean"], d["std"], d["sample0"]
err = d["err"]; names = [str(n) for n in d["names"]]
zc = GRID // 2

def sl(a, j):
    return a[:, j].reshape(GRID, GRID, GRID)[:, :, zc]

cols = ["truth", "posterior mean", "single sample", "posterior std", "|error|"]
show = [0, 1]           # Ux (observed), Uy (unobserved)
fig, axes = plt.subplots(len(show), 5, figsize=(12.4, 2.45 * len(show)), dpi=200)
for r, j in enumerate(show):
    t, m, s1, sd, e = sl(truth, j), sl(mean, j), sl(sample0, j), sl(std, j), sl(err, j)
    vmin, vmax = np.percentile(t, [1, 99])
    for c, (arr, cmap, vr) in enumerate([
            (t, "RdBu_r", (vmin, vmax)), (m, "RdBu_r", (vmin, vmax)),
            (s1, "RdBu_r", (vmin, vmax)),
            (sd, "viridis", (0, np.percentile(sd, 99))),
            (e, "magma", (0, np.percentile(e, 99)))]):
        ax = axes[r, c]
        im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=vr[0], vmax=vr[1],
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(cols[c], fontsize=10)
        if c == 0:
            ax.set_ylabel(names[j] + (" (observed)" if j in (0, 2)
                                      else " (unobserved)"), fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
fig.suptitle("Held-out cube, z-midplane slice: 16-member ensemble at NFE 4, "
             "1% sensors on Ux and Uz", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "qual_jhu.pdf"); fig.savefig(OUT / "qual_jhu.png")
print("rewrote qual_jhu (2 rows)")
