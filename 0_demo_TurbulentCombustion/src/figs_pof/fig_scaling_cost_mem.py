"""PoF figure: main-text memory-only cost panel (1x2).

The two memory panels carry the OOM-wall claim; timing panels live in the
appendix version (fig_scaling_cost.py).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pof_style import use_style, save, FULL_W, C_OURS, C_SENS, INK_2
from scaling_common import (load_series, draw, res_ticks, annotate_h100,
                            crossover)

use_style()
s = load_series()

fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.0), sharex=True,
                       constrained_layout=True)

a = ax[0]
draw(a, s, "infer", "mem")
a.set_ylabel("peak GPU memory (MB)")
a.set_xlabel("output points $N$")
a.set_title("(a) inference", fontsize=8.5)
a.annotate("flat 1.4 GB, $64^3$–$1024^3$", xy=(s[("ours", "infer")]["n"][-1],
           s[("ours", "infer")]["mem"][-1]), xytext=(0.47, 0.10),
           textcoords="axes fraction", fontsize=6.5, color=C_OURS,
           arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.7))
nc = crossover(s[("ours", "infer")], s[("conv", "infer")])
a.plot(nc, 1385, "o", ms=5, mfc="none", mec=INK_2, mew=0.8)
a.annotate(f"ConvAE cheaper below $\\approx${round(nc**(1/3)/10)*10}$^3$",
           xy=(nc, 1385), xytext=(0.04, 0.60), textcoords="axes fraction",
           fontsize=6.5, color=INK_2,
           arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.7))
a.legend(loc="upper left", fontsize=6.0, frameon=True)

a = ax[1]
draw(a, s, "train", "mem")
annotate_h100(a)
a.set_xlabel("field points $N$")
a.set_title("(b) training", fontsize=8.5)
a.text(0.97, 0.30, "DMF-Gen: flat 39.9 GB (39k queries/step)",
       transform=a.transAxes, ha="right", fontsize=6.5, color=C_OURS)
a.text(0.97, 0.22, "Senseiver: flat 19.4 GB (batch 20)",
       transform=a.transAxes, ha="right", fontsize=6.5, color=C_SENS)
nct = crossover(s[("ours", "train")], s[("conv", "train")])
a.annotate(f"ConvAE cheaper below $\\approx${round(nct**(1/3)/10)*10}$^3$",
           xy=(0.03, 0.06), xycoords="axes fraction", fontsize=6.5, color=INK_2)

for a in ax.flat:
    res_ticks(a)

fig.suptitle("Peak GPU memory vs resolution (H100 80 GB, bf16;  "
             "X = measured out-of-memory)", fontsize=9.5)
save(fig, "scaling_cost_mem")
