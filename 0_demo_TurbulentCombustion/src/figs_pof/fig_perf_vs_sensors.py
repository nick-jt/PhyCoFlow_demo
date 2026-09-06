"""Performance vs number of sensor points, 2D and 3D side by side.

Left: 2D Kolmogorov 256^2 (single observed channel, vorticity).
Right: 3D JHU 125^3 cross-cube (Ux/Uz observed; solid = observed-channel
rel-L2, dashed = aggregate over all four channels).

The panels are deliberately the two "opposite sides" of the density story:
at the SAME sensor count the 2D error collapses toward the interpolation
limit while the 3D aggregate saturates against the unobserved-channel
identifiability wall (200x more sensors move DMF-Gen's aggregate only
0.68 -> 0.54, while IDW's observed-channel error keeps falling 0.31 -> 0.08).

Sources (read-only):
  $MAIN/Save_TrainedModel/JHU/baseline_classical/classical_baselines_sweep.json
  $MAIN/Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_xcube_spec02_DemoN29_*/
      Evaluation/calib_sweep_nfe4_n*_K8.json   (DMF-Gen, K=8, 50 snapshots)
  $MAIN/Save_TrainedModel/kolmogorov2d/baseline_classical/
      classical_baselines_sweep_nonperiodic_2d.json   (added when job lands;
      falls back to the single-density main_1pct json with a note)
Learned 2D rows are appended automatically once their eval JSONs exist
(scan pattern below) -- rerun this script after the 2D fleet finishes.
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN = "/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion"
# 2D classical runs write into the worktree's Save_TrainedModel, not $MAIN.
WT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "Paper", "pof2026", "figures")

# Validated categorical palette (see check_palette.py); fixed method order.
C = {
    "dmfgen": "#2a78d6",
    "idw": "#eb6834",
    "kdtree": "#1baf7a",
    "pod": "#eda100",
    "constant": "#9a9890",
    "grid": "#e1e0d9",
    "ink": "#3a3a37",
    "muted": "#8a887f",
}

N2D = 256 * 256
N3D = 125 ** 3

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "pdf.fonttype": 42, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def classical_curves(path, obs_fields):
    res = json.load(open(path))["results"]
    out = {}
    for key, v in res.items():
        m = re.match(r"([a-z_]+)_n(\d+)$", key)
        if not m:
            continue
        meth, n = m.group(1), int(m.group(2))
        pf = v["per_field"]
        vals = [pf[f]["rel_l2_mean"] for f in obs_fields if f in pf]
        if vals:
            out.setdefault(meth, {})[n] = float(np.mean(vals))
    return {m: sorted(d.items()) for m, d in out.items()}


def dmfgen_curves():
    pat = os.path.join(
        MAIN, "Save_TrainedModel/JHU/pointcloud_ffm",
        "iclr_jhu_xcube_spec02_DemoN29_*", "Evaluation", "calib_sweep_nfe4_n*_K8.json")
    obs, agg = {}, {}
    for f in glob.glob(pat):
        j = json.load(open(f))
        n = int(re.search(r"_n(\d+)_K8", f).group(1))
        per_snap_obs = [
            np.mean([s["per_field"]["Ux"]["rel_l2_mean"], s["per_field"]["Uz"]["rel_l2_mean"]])
            for s in j["snapshots"]
        ]
        obs[n] = float(np.mean(per_snap_obs))
        agg[n] = float(j["summary"]["rel_l2_mean"])
    return sorted(obs.items()), sorted(agg.items())


def plot_series(ax, pairs, color, label, ls="-", marker="o"):
    if not pairs:
        return
    ns, ys = zip(*pairs)
    ax.plot(ns, ys, ls, color=color, lw=1.4, marker=marker, ms=3.5,
            mfc="white", mew=1.1, mec=color, label=label, zorder=3)


def style(ax, n_total, title):
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.grid(True, color=C["grid"], lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", color=C["ink"])
    ax.set_xlabel("sensor points per observed field")
    top = ax.secondary_xaxis("top", functions=(lambda n: 100 * n / n_total,
                                               lambda p: p * n_total / 100))
    top.set_xlabel("% of grid points", fontsize=7, color=C["muted"])
    top.tick_params(labelsize=7, colors=C["muted"])
    top.spines["top"].set_visible(False)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)

# ---- Panel A: 2D Kolmogorov -------------------------------------------------
sweep2d = os.path.join(WT, "Save_TrainedModel/kolmogorov2d/baseline_classical",
                       "classical_baselines_sweep_nonperiodic_2d.json")
single2d = os.path.join(WT, "Save_TrainedModel/kolmogorov2d/baseline_classical",
                        "classical_baselines_main_1pct_nonperiodic_2d.json")
if os.path.exists(sweep2d):
    c2 = classical_curves(sweep2d, ["vorticity"])
else:
    c2 = classical_curves(single2d, ["vorticity"])
    axA.annotate("full sweep job queued;\nsingle-density points shown",
                 xy=(0.03, 0.06), xycoords="axes fraction", fontsize=6.5,
                 color=C["muted"])
plot_series(axA, c2.get("idw", []), C["idw"], "IDW $k{=}8$")
plot_series(axA, c2.get("kdtree", []), C["kdtree"], "nearest sensor")
plot_series(axA, c2.get("gappy_pod", []), C["pod"], "gappy POD $r{=}80$")
if "constant" in c2:
    plot_series(axA, c2["constant"], C["constant"], "train mean", ls=":", marker="")
else:
    axA.axhline(1.0, color=C["constant"], ls=":", lw=1.0)
    axA.annotate("train mean", xy=(0.97, 0.965), xycoords="axes fraction",
                 ha="right", fontsize=6.5, color=C["constant"])
# Learned 2D rows: appended automatically when the fleet's eval JSONs land.
for meth, color in (("dmfgen", C["dmfgen"]),):
    hits = []
    for root in (WT, MAIN):
        pat = os.path.join(root, "Save_TrainedModel/kolmogorov2d", "**",
                           f"sensor_sweep_{meth}.json")
        hits += glob.glob(pat, recursive=True)
    for h in hits:
        j = json.load(open(h))
        pairs = sorted((int(k), v) for k, v in j.get("rel_l2_by_n", {}).items())
        plot_series(axA, pairs, color, "DMF-Gen (observed)")  # label matches 3D entry -> legend dedupes
axA.set_xlim(45, 9000)  # the eventual sweep range {65..6554}, so single points sit in context
style(axA, N2D, "2D Kolmogorov $256^2$ (observed channel)")
axA.set_ylabel("relative $L_2$ error")

# ---- Panel B: 3D JHU --------------------------------------------------------
c3 = classical_curves(os.path.join(MAIN, "Save_TrainedModel/JHU/baseline_classical",
                                   "classical_baselines_sweep.json"), ["Ux", "Uz"])
d_obs, d_agg = dmfgen_curves()
plot_series(axB, d_obs, C["dmfgen"], "DMF-Gen (observed)")
plot_series(axB, d_agg, C["dmfgen"], "DMF-Gen (all channels)", ls="--", marker="s")
plot_series(axB, c3.get("idw", []), C["idw"], "IDW $k{=}8$")
plot_series(axB, c3.get("kdtree", []), C["kdtree"], "nearest sensor")
plot_series(axB, c3.get("gappy_pod", []), C["pod"], "gappy POD $r{=}80$")
style(axB, N3D, "3D isotropic turbulence $125^3$")

# identifiability-wall annotation on the dashed aggregate curve
if d_agg:
    n_hi, y_hi = d_agg[-1]
    axB.annotate("unobserved-channel wall:\n$200\\times$ more sensors,\naggregate only "
                 f"{d_agg[0][1]:.2f}$\\rightarrow${y_hi:.2f}",
                 xy=(n_hi, y_hi), xytext=(0.44, 0.80), textcoords="axes fraction",
                 fontsize=6.5, color=C["ink"],
                 arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.6))

# matched-count guide across both panels
for ax, gap, frac in ((axA, (N2D / 1953) ** 0.5, 100 * 1953 / N2D),
                      (axB, (N3D / 1953) ** (1 / 3), 100 * 1953 / N3D)):
    ax.axvline(1953, color=C["muted"], lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.annotate(f"N=1953 ({frac:.2g}%)\nmean gap {gap:.1f} cells",
                xy=(1953, 0.02), xytext=(3, 2), textcoords="offset points",
                fontsize=6.0, color=C["muted"], va="bottom")

# one shared legend below both panels (colors are consistent across panels)
handles, labels = axB.get_legend_handles_labels()
for h, l in zip(*axA.get_legend_handles_labels()):
    if l not in labels:
        handles.append(h)
        labels.append(l)
fig.legend(handles, labels, loc="lower center", frameon=False, ncol=6,
           handlelength=1.8, columnspacing=1.2, fontsize=7,
           bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "perf_vs_sensors.pdf"))
fig.savefig(os.path.join(OUT, "perf_vs_sensors.png"), dpi=220)
print("wrote", os.path.join(OUT, "perf_vs_sensors.pdf"))
