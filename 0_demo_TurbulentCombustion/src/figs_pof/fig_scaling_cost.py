"""PoF figure: full 2x2 cost-of-resolution panel (appendix version).

(a) inference peak memory, (b) training peak memory, (c) inference wall-clock,
(d) training wall-clock -- all log-log vs number of field points, four model
families, measured OOM walls as X on the 80 GB line. The main-text memory-only
variant is fig_scaling_cost_mem.py.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pof_style import use_style, save, FULL_W, C_OURS, C_SENS, INK_2, MUTED
from scaling_common import (load_series, draw, res_ticks, annotate_h100,
                            crossover)

use_style()
s = load_series()

fig, ax = plt.subplots(2, 2, figsize=(FULL_W, 5.5), sharex=True,
                       constrained_layout=True)

# (a) inference memory
a = ax[0, 0]
draw(a, s, "infer", "mem")
a.set_ylabel("peak GPU memory (MB)")
a.set_title("(a) inference — peak memory", fontsize=8.5)
a.annotate("flat 1.4 GB, $64^3$–$1024^3$", xy=(s[("ours", "infer")]["n"][-1],
           s[("ours", "infer")]["mem"][-1]), xytext=(0.45, 0.10),
           textcoords="axes fraction", fontsize=6.5, color=C_OURS,
           arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.7))
nc = crossover(s[("ours", "infer")], s[("conv", "infer")])
a.plot(nc, 1385, "o", ms=5, mfc="none", mec=INK_2, mew=0.8)
a.annotate(f"ConvAE cheaper below $\\approx${round(nc**(1/3)/10)*10}$^3$",
           xy=(nc, 1385), xytext=(0.04, 0.62), textcoords="axes fraction",
           fontsize=6.5, color=INK_2,
           arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
a.legend(loc="upper left", fontsize=6.0, frameon=True)

# (b) training memory
a = ax[0, 1]
draw(a, s, "train", "mem")
annotate_h100(a)
a.set_title("(b) training — peak memory", fontsize=8.5)
a.text(0.97, 0.30, "DMF-Gen: flat 39.9 GB (39k queries/step)",
       transform=a.transAxes, ha="right", fontsize=6.5, color=C_OURS)
a.text(0.97, 0.22, "Senseiver: flat 19.4 GB (batch 20)",
       transform=a.transAxes, ha="right", fontsize=6.5, color=C_SENS)
nct = crossover(s[("ours", "train")], s[("conv", "train")])
a.annotate(f"ConvAE cheaper below $\\approx${round(nct**(1/3)/10)*10}$^3$",
           xy=(0.03, 0.06), xycoords="axes fraction", fontsize=6.5, color=INK_2)

# (c) inference time
a = ax[1, 0]
draw(a, s, "infer", "t")
a.set_ylabel("wall-clock (s)")
a.set_xlabel("field points $N$")
a.set_title("(c) inference — wall-clock per sample", fontsize=8.5)
a.annotate("$\\propto N$; 41 min at $1024^3$",
           xy=(s[("ours", "infer")]["n"][-1], s[("ours", "infer")]["t"][-1]),
           xytext=(0.50, 0.78), textcoords="axes fraction", fontsize=6.5,
           color=C_OURS,
           arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.7))
a.annotate("Gen4Turb: 32 diffusion steps", xy=(0.35, 0.06),
           xycoords="axes fraction", fontsize=6.2, color=MUTED)

# (d) training time
a = ax[1, 1]
draw(a, s, "train", "t")
a.set_xlabel("field points $N$")
a.set_ylabel("s / step")
a.set_title("(d) training — wall-clock per step", fontsize=8.5)
a.annotate("flat 0.44 s/step (subsampled queries)",
           xy=(s[("ours", "train")]["n"][2], 0.44), xytext=(0.24, 0.72),
           textcoords="axes fraction", fontsize=6.5, color=C_OURS)

for a in ax.flat:
    res_ticks(a)

fig.suptitle("Cost of resolution (H100 80 GB, bf16;  X = measured out-of-memory)",
             fontsize=9.5)
save(fig, "scaling_cost")
print(f"inference crossover: {nc:.3g} points = {nc**(1/3):.0f}^3; "
      f"training crossover: {nct:.3g} points = {nct**(1/3):.0f}^3")
