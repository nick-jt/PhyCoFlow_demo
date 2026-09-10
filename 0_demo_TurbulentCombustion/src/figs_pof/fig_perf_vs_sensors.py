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

MAIN = "/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion"
# 2D classical runs write into the worktree's Save_TrainedModel, not $MAIN.
WT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "Paper", "pof2026", "figures")

# Validated categorical palette (see check_palette.py); fixed method order.
C = {
    # learned fleet -- categorical slots, same family colors as every other figure
    "dmfgen": "#2a78d6",
    "latentfm": "#eb6834",
    "senseiver": "#1baf7a",
    "geofno": "#eda100",
    "sit": "#7a5cc6",
    "mlprbf": "#d6446e",
    "s3gm": "#3f8f8a",
    # classical floors -- neutral, separated by linestyle. Deliberately gray:
    # the panel's message is learned-vs-floor, and reusing the categorical
    # slots for IDW/POD would collide with latent-FM and Geo-FNO.
    "idw": "#55534f",
    "kdtree": "#8a887f",
    "pod": "#b0aea5",
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


# 2D Kolmogorov learned fleet: each row writes one JSON per sensor density.
# Two naming conventions are in the tree (the DMF-Gen runner predates the fleet
# launcher), so both are scanned; the per-density files are authoritative and
# the roll-up is only a fallback.
LEARNED_2D = {
    "dmfgen":    ("pointcloud_ffm",     "sensor_sweep_dmfgen_n*.json"),
    "senseiver": ("baseline_det",       "kolm_sweep_senseiver_n*.json"),
    "mlprbf":    ("baseline_mlp_rbf",   "kolm_sweep_mlprbf_n*.json"),
    "geofno":    ("baseline_geofno",    "kolm_sweep_geofno_n*.json"),
    "sit":       ("baseline_sit",       "kolm_sweep_sit_n*.json"),
    "latentfm":  ("baseline_latent_fm", "kolm_sweep_latentfm_n*.json"),
    # S3GM's sweep points come from the density-override launcher, which names
    # them kolm_fleet_ovr_n<N>_s3gm_...; its canonical 655 row keeps the plain
    # fleet name. Both are read, and the n_obs recorded INSIDE each file is what
    # places the point -- never the filename.
    "s3gm":      ("baseline_s3gm",      "kolm_fleet*s3gm_K*.json"),
}
LEARNED_LABEL = {"dmfgen": "DMF-Gen (observed)", "senseiver": "Senseiver",
                 "mlprbf": "MLP-RBF", "geofno": "Geo-FNO", "sit": "SiT",
                 "latentfm": "latent FM", "s3gm": "S3GM"}


def learned_2d_curves():
    """{method: [(n, rel_L2), ...]} for whatever has landed. Rows still running
    are simply absent, so the figure is re-runnable as jobs finish."""
    out = {}
    for meth, (fam, pat) in LEARNED_2D.items():
        pairs = {}
        for root in (WT, MAIN):
            for f in glob.glob(os.path.join(root, "Save_TrainedModel/kolmogorov2d",
                                            fam, "*", "Evaluation", pat)):
                j = json.load(open(f))
                n = j.get("n_obs")
                if isinstance(n, list):
                    n = int(n[0])
                elif n is not None:
                    n = int(n)
                else:
                    m = re.search(r"_n(\d+)[._]", os.path.basename(f))
                    if m is None:
                        continue          # unlabelled density: not plottable
                    n = int(m.group(1))
                # every row must be on the frozen protocol, or it is not
                # comparable with the rest of the panel
                if j.get("protocol") != "kolm2d_matched_v1":
                    continue
                pairs[n] = float(j["summary"]["aggregate"]["rel_l2_mean"])
        if pairs:
            out[meth] = sorted(pairs.items())
    return out


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


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.25), sharey=True)

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
# classical floors first, so the learned fleet draws on top of them
plot_series(axA, c2.get("idw", []), C["idw"], "IDW $k{=}8$", ls="--", marker="^")
plot_series(axA, c2.get("kdtree", []), C["kdtree"], "nearest sensor",
            ls="-.", marker="v")
plot_series(axA, c2.get("gappy_pod", []), C["pod"], "gappy POD $r{=}80$",
            ls=":", marker="s")
if "constant" in c2:
    plot_series(axA, c2["constant"], C["constant"], "train mean", ls=":", marker="")
else:
    axA.axhline(1.0, color=C["constant"], ls=":", lw=1.0)
    axA.annotate("train mean", xy=(0.97, 0.965), xycoords="axes fraction",
                 ha="right", fontsize=6.5, color=C["constant"])

# learned fleet: all seven rows, plotted in a fixed order so the legend is stable
L2 = learned_2d_curves()
for meth in ("dmfgen", "sit", "geofno", "latentfm", "mlprbf", "senseiver", "s3gm"):
    plot_series(axA, L2.get(meth, []), C[meth], LEARNED_LABEL[meth])
# A row with one or two densities is a point, not a curve, and saying nothing
# about it would let a single marker read as a finished sweep.
partial = [f"{LEARNED_LABEL[m]} ({len(L2.get(m, []))}/5)"
           for m in LEARNED_2D if len(L2.get(m, [])) < 5]
if partial:
    axA.annotate("sweep incomplete: " + ", ".join(partial), xy=(0.03, 0.04),
                 xycoords="axes fraction", fontsize=6.0, color=C["muted"])

# the crossover is the panel's finding: below it the learned fleet beats
# interpolation, above it plain IDW is the best method on the plot
idw = dict(c2.get("idw", []))
best_learned = {n: min(v for m, pairs in L2.items() for k, v in pairs if k == n)
                for n in idw if any(k == n for pairs in L2.values() for k, _ in pairs)}
cross = [n for n in sorted(best_learned) if idw[n] < best_learned[n]]
if cross:
    n0 = cross[0]
    axA.annotate(f"at {n0} sensors plain IDW\nbeats every learned row",
                 xy=(n0, idw[n0]), xytext=(0.42, 0.70), textcoords="axes fraction",
                 fontsize=6.2, color=C["ink"], ha="left",
                 arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.6,
                                 shrinkB=2))

axA.set_xlim(45, 9000)  # the eventual sweep range {65..6554}, so single points sit in context
style(axA, N2D, "2D Kolmogorov $256^2$ (observed channel)")
axA.set_ylabel("relative $L_2$ error")

# ---- Panel B: 3D JHU --------------------------------------------------------
c3 = classical_curves(os.path.join(MAIN, "Save_TrainedModel/JHU/baseline_classical",
                                   "classical_baselines_sweep.json"), ["Ux", "Uz"])
d_obs, d_agg = dmfgen_curves()
plot_series(axB, d_obs, C["dmfgen"], "DMF-Gen (observed)")
plot_series(axB, d_agg, C["dmfgen"], "DMF-Gen (all channels)", ls="--", marker="s")
plot_series(axB, c3.get("idw", []), C["idw"], "IDW $k{=}8$", ls="--", marker="^")
plot_series(axB, c3.get("kdtree", []), C["kdtree"], "nearest sensor",
            ls="-.", marker="v")
plot_series(axB, c3.get("gappy_pod", []), C["pod"], "gappy POD $r{=}80$",
            ls=":", marker="s")
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
fig.legend(handles, labels, loc="lower center", frameon=False, ncol=5,
           handlelength=1.9, columnspacing=1.1, fontsize=6.6,
           bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(w_pad=1.2, rect=(0, 0.13, 1, 1))
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "perf_vs_sensors.pdf"))
fig.savefig(os.path.join(OUT, "perf_vs_sensors.png"), dpi=220)
print("wrote", os.path.join(OUT, "perf_vs_sensors.pdf"))
