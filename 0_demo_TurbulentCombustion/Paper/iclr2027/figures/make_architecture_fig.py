"""ICLR architecture schematic for LOCUS (v1, programmatic).

Three panels:
  (a) observation set -> sensor tokens -> latent array (global pathway)
  (b) per-query velocity assembly (point feature + global readout + local retrieval)
  (c) rectified-flow integration from a GP source draw to a posterior sample
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np

plt.rcParams.update({
    "font.size": 8.0, "font.family": "sans-serif",
    "axes.linewidth": 0.0,
})

C_SENSOR = "#2b6cb0"   # blue
C_LATENT = "#c05621"   # orange
C_QUERY = "#276749"    # green
C_FLOW = "#6b46c1"     # purple
C_BOX = "#f7fafc"
C_EDGE = "#4a5568"

fig = plt.figure(figsize=(10.5, 3.1), dpi=200)


def box(ax, x, y, w, h, label, fc=C_BOX, ec=C_EDGE, fs=7.5, lw=1.0, rounded=0.03):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={rounded}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1.0, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, zorder=3)
    return b


def arrow(ax, p0, p1, color=C_EDGE, lw=1.2, style="-|>", shrink=2, alpha=1.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                        color=color, lw=lw, shrinkA=shrink, shrinkB=shrink,
                        alpha=alpha, linestyle=ls, zorder=4)
    ax.add_patch(a)


# ---------------- panel (a): global pathway ----------------
ax = fig.add_axes([0.005, 0.02, 0.31, 0.88])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(a) Observation encoding — $\\mathcal{O}(SM)$", fontsize=8.5, pad=2)

rng = np.random.default_rng(3)
# scattered sensors on a wing-like blob
th = np.linspace(0, 2 * np.pi, 100)
ax.fill(0.16 + 0.11 * np.cos(th), 0.62 + 0.30 * np.sin(th) * (1 + 0.3 * np.cos(th)),
        color="#e2e8f0", zorder=1)
sx = 0.16 + 0.10 * rng.standard_normal(14) * 0.8
sy = 0.62 + 0.22 * rng.standard_normal(14) * 0.9
sx, sy = np.clip(sx, 0.04, 0.28), np.clip(sy, 0.30, 0.94)
ax.scatter(sx, sy, s=12, c=C_SENSOR, zorder=5)
ax.text(0.16, 0.185, "observations\n$(\\vec{x}_j, c_j, y_j)$", ha="center", fontsize=7.2)

# sensor tokens column
for i in range(7):
    box(ax, 0.40, 0.24 + i * 0.093, 0.085, 0.062, "", fc="#bee3f8", ec=C_SENSOR, lw=0.8)
ax.text(0.443, 0.13, "sensor\ntokens $M$", ha="center", fontsize=7.2)
arrow(ax, (0.30, 0.62), (0.385, 0.62), color=C_SENSOR)
ax.text(0.335, 0.68, "embed", fontsize=6.6, ha="center", color=C_SENSOR)

# latent array
for i in range(4):
    box(ax, 0.66, 0.37 + i * 0.093, 0.085, 0.062, "", fc="#feebc8", ec=C_LATENT, lw=0.8)
ax.text(0.70, 0.26, "latent array $S$", ha="center", fontsize=7.2)
arrow(ax, (0.50, 0.60), (0.645, 0.60), color=C_LATENT)
ax.text(0.575, 0.66, "cross-attn", fontsize=6.6, ha="center", color=C_LATENT)
arrow(ax, (0.645, 0.50), (0.50, 0.50), color=C_SENSOR, ls=(0, (2, 2)), lw=1.0)
ax.text(0.575, 0.42, "re-inject", fontsize=6.6, ha="center", color=C_SENSOR)

box(ax, 0.87, 0.52, 0.115, 0.10, "global\n$\\vec{z}$", fc="#feebc8", ec=C_LATENT, fs=7)
arrow(ax, (0.755, 0.57), (0.862, 0.57), color=C_LATENT)

# ---------------- panel (b): per-query assembly ----------------
ax = fig.add_axes([0.345, 0.02, 0.34, 0.88])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(b) Per-query velocity — $\\mathcal{O}(N(L+K))$", fontsize=8.5, pad=2)

# query point + neighborhood
qx, qy = 0.20, 0.55
nb = rng.standard_normal((10, 2)) * 0.09
ax.scatter(qx + nb[:, 0], qy + nb[:, 1], s=10, c=C_SENSOR, alpha=0.75, zorder=4)
far = np.array([[0.36, 0.83], [0.07, 0.20]])
ax.scatter(far[:, 0], far[:, 1], s=16, c=C_SENSOR, zorder=4)
ax.scatter([qx], [qy], s=42, marker="*", c=C_QUERY, zorder=6)
circ = Circle((qx, qy), 0.13, fc="none", ec=C_QUERY, lw=0.9, ls=(0, (2, 2)), zorder=3)
ax.add_patch(circ)
for f in far:
    arrow(ax, tuple(f), (qx, qy), color=C_SENSOR, lw=0.8, ls=(0, (1, 2)), alpha=0.8)
ax.text(qx, 0.18, "query $\\vec{y}_i$ + top-$K$ retrieval\n(importance-warped metric)",
        ha="center", fontsize=6.8)

bx = 0.52
box(ax, bx, 0.72, 0.20, 0.115, "point feature\n$[\\gamma(\\vec{y}_i), u_t(\\vec{y}_i), t]$", fc="#c6f6d5", ec=C_QUERY, fs=6.6)
box(ax, bx, 0.545, 0.20, 0.115, "global readout\n(query$\\to$latents)", fc="#feebc8", ec=C_LATENT, fs=6.6)
box(ax, bx, 0.37, 0.20, 0.115, "local retrieval\n($K$ sensor tokens)", fc="#bee3f8", ec=C_SENSOR, fs=6.6)
box(ax, bx, 0.195, 0.20, 0.115, "coarse scaffold\nFiLM($\\vec{z}$)", fc="#feebc8", ec=C_LATENT, fs=6.6)

box(ax, 0.80, 0.42, 0.135, 0.16, "fused\nhead", fc=C_BOX, fs=7.2)
for yy in (0.7775, 0.6025, 0.4275, 0.2525):
    arrow(ax, (bx + 0.20, yy), (0.797, 0.50), lw=0.9)
arrow(ax, (0.935, 0.50), (0.995, 0.50), color=C_QUERY, lw=1.4)
ax.text(0.968, 0.57, "$v_\\theta$", fontsize=8.5, ha="center", color=C_QUERY)

# ---------------- panel (c): flow integration ----------------
ax = fig.add_axes([0.70, 0.02, 0.295, 0.88])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(c) Rectified-flow sampling (2–16 steps)", fontsize=8.5, pad=2)

xs = np.linspace(0.05, 0.95, 200)
gp = 0.16 * np.sin(9 * xs) * np.exp(-1.5 * xs) + 0.10 * np.sin(23 * xs) * 0.4
ax.plot(xs, 0.80 + gp, color=C_FLOW, lw=1.1)
ax.text(0.5, 0.935, "GP source draw $u_0\\sim\\pi_0$ (RFF)", fontsize=7.0,
        ha="center", color=C_FLOW)

turb = (0.10 * np.sin(21 * xs + 1.2) + 0.05 * np.sin(47 * xs) +
        0.035 * np.sin(89 * xs + 0.4))
ax.plot(xs, 0.16 + turb, color=C_QUERY, lw=1.1)
ax.text(0.5, 0.035, "posterior sample $u_1\\sim p(u\\,|\\,\\mathcal{O})$",
        fontsize=7.0, ha="center", color=C_QUERY)

for i, xx in enumerate(np.linspace(0.22, 0.78, 4)):
    arrow(ax, (xx, 0.74), (xx, 0.30), color=C_FLOW, lw=1.0, alpha=0.85)
ax.text(0.87, 0.52, "$\\dot{u}=v_\\theta(t,u,\\vec{y};\\mathcal{O})$", fontsize=7.4, ha="center")
ax.scatter(np.linspace(0.3, 0.7, 5), np.full(5, 0.52), s=7, c=C_SENSOR, zorder=5)
ax.text(0.5, 0.585, "sensors", fontsize=6.2, ha="center", color=C_SENSOR)

import os
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
