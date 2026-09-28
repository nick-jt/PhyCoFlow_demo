"""Re-render the FireBench qualitative figure from saved arrays.

Wind and temperature are shown on a vertical slice through the fire front.
Fuel density lives in the ground-level fuel bed, which a vertical slice
barely intersects -- rendering it there stretches a near-constant field over
its noise floor and misrepresents a channel reconstructed to 0.036 relative
L2. That row is therefore drawn on the horizontal plane of maximum fuel
variance, and each row is labelled with its plane.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NX, NY, NZ = 152, 126, 192
OUT = Path("../Paper/iclr2027/figures")
d = np.load(OUT / "qual_firebench.npz")
truth, mean, std, sample0 = d["truth"], d["mean"], d["std"], d["sample0"]
yc = int(d["yc"]); names = [str(n) for n in d["names"]]
err = np.abs(mean - truth)

def cube(a, j):
    return a[:, j].reshape(NX, NY, NZ)

# fuel bed: the horizontal plane where truth fuel density varies most
zc = int(np.argmax(cube(truth, 4).var(axis=(0, 1))))

rows = [(0, "vert"), (3, "vert"), (4, "horiz")]
plane_lbl = {"vert": f"vertical slice (y={yc})", "horiz": f"fuel-bed plane (z={zc})"}
cols = ["truth", "posterior mean", "single sample", "posterior std", "|error|"]

fig, axes = plt.subplots(len(rows), 5, figsize=(13.6, 2.35 * len(rows)), dpi=200)
for r, (j, plane) in enumerate(rows):
    def sl(a):
        c = cube(a, j)
        return c[:, yc, :] if plane == "vert" else c[:, :, zc]
    t, m, s1, sd, e = sl(truth), sl(mean), sl(sample0), sl(std), sl(err)
    vmin, vmax = np.percentile(t, [1, 99])
    if vmax - vmin < 1e-6:
        vmin, vmax = t.min(), t.max()
    for c, (arr, cmap, vr) in enumerate([
            (t, "inferno", (vmin, vmax)), (m, "inferno", (vmin, vmax)),
            (s1, "inferno", (vmin, vmax)),
            (sd, "viridis", (0, np.percentile(sd, 99))),
            (e, "magma", (0, np.percentile(e, 99)))]):
        ax = axes[r, c]
        im = ax.imshow(arr.T, origin="lower", cmap=cmap, aspect="auto",
                       vmin=vr[0], vmax=vr[1], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(cols[c], fontsize=10)
        if c == 0:
            lbl = " (observed)" if j in (0, 1, 2) else " (unobserved)"
            ax.set_ylabel(f"{names[j]}{lbl}\n{plane_lbl[plane]}", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
fig.suptitle("Wildfire LES, held-out snapshot, conditioned on wind sensors only "
             "(16-member ensemble)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(OUT / "qual_firebench.pdf")
fig.savefig(OUT / "qual_firebench.png")
print("rewrote qual_firebench; fuel-bed plane z =", zc)
for j, nm in enumerate(names):
    print(f"  {nm}: truth range on shown plane, rel_l2 = "
          f"{np.linalg.norm(mean[:, j]-truth[:, j])/np.linalg.norm(truth[:, j]):.4f}")
