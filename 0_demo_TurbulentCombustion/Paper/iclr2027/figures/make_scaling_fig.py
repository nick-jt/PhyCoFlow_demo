"""Cost-scaling figure: ambient point-cloud FM vs grid-locked baselines.

Reads the four scaling_*.json sweeps produced by benchmark_cost.py
(job 16628458, H100 80GB, bf16) and renders a 2x2 log-log panel:
peak GPU memory / wall-clock  x  inference / training.
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EV = pathlib.Path("../../../Save_TrainedModel/JHU/pointcloud_ffm/"
                  "iclr_jhu_xcube_spec02_DemoN29_20260822_140100/Evaluation")
load = lambda n: json.load(open(EV / f"{n}.json"))

C_OURS, C_CONV, C_VOX = "#1f77b4", "#d62728", "#7f7f7f"
H100 = 81559  # MB

def split(rows, tkey):
    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok") and r.get("error") == "OOM"]
    return ([r["n_points"] for r in ok], [r["peak_mb"] for r in ok],
            [r[tkey] for r in ok], [r["n_points"] for r in bad])

oi_n, oi_m, oi_t, _        = split(load("scaling_ours_infer"), "seconds")
def split_named(name, model, tkey):
    rows = [r for r in load(name) if r.get("model") == model]
    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok") and r.get("error") == "OOM"]
    return ([r["n_points"] for r in ok], [r["peak_mb"] for r in ok],
            [r[tkey] for r in ok], [r["n_points"] for r in bad])
si_n, si_m, si_t, _      = split_named("scaling_senseiver", "senseiver_infer", "seconds")
st_n, st_m, st_t, _      = split_named("scaling_senseiver", "senseiver_train", "sec_per_step")
gi_n, gi_m, gi_t, _      = split_named("scaling_gen4turb", "gen4turb_infer_step", "seconds")
gt_n, gt_m, gt_t, gt_oom = split_named("scaling_gen4turb", "gen4turb_train", "sec_per_step")
gt_oom = gt_oom[:1]  # first measured wall (320^3); later rows are downstream of it
C_SEN = "#2ca02c"
ci_n, ci_m, ci_t, ci_oom   = split(load("scaling_convae_infer"), "seconds")
ct_n, ct_m, ct_t, ct_oom   = split(load("scaling_convae_train"), "sec_per_step")
ct_oom += split(load("scaling_convae_train_wall"), "sec_per_step")[3]
ot_n, ot_m, ot_t, _        = split(load("scaling_ours_train"), "sec_per_step")

import os as _os
MEM_ONLY = _os.environ.get("SCALING_MEM_ONLY", "") == "1"
if MEM_ONLY:
    # Main-text version: only the two memory panels, which carry the OOM-wall
    # claim. The timing panels are drawn on a discarded figure so the plotting
    # code below can stay identical.
    import numpy as _np
    fig, _a = plt.subplots(1, 2, figsize=(9.2, 3.35), dpi=200)
    _junk, _ja = plt.subplots(1, 2, figsize=(4, 2))
    ax = _np.array([[_a[0], _a[1]], [_ja[0], _ja[1]]], dtype=object)
else:
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.8), dpi=200, sharex=True)

def oom_marks(a, ns, ref_y):
    for n in ns:
        a.plot(n, ref_y, marker="x", ms=11, mew=2.6, color=C_CONV, ls="none")

# (0,0) inference memory
a = ax[0, 0]
a.plot(oi_n, oi_m, "o-", color=C_OURS, label="ours (chunked sampler)")
a.plot(ci_n, ci_m, "s-", color=C_CONV, label="ConvAE decode (latent-FM lower bound)")
a.plot(si_n, si_m, "^-", color=C_SEN, label="Senseiver (chunked decode)")
a.plot(gi_n, gi_m, "D-", color=C_VOX, label="voxel diffusion, per step (Gen4Turb)")
a.axhline(H100, color="k", lw=0.8, ls=":"); a.text(2.5e5, H100 * 1.15, "H100 80 GB", fontsize=7)
oom_marks(a, ci_oom, H100)
a.set_ylabel("peak GPU memory (MB)"); a.set_title("Inference", fontsize=10)
a.annotate("flat: 1.39 GB at $64^3\\!-\\!1024^3$", xy=(oi_n[-1], oi_m[-1]),
           xytext=(0.35, 0.16), textcoords="axes fraction", fontsize=8, color=C_OURS,
           arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.7))
# (0,1) training memory
a = ax[0, 1]
a.plot(ot_n, ot_m, "o-", color=C_OURS, label="ours (39k queries/step)")
a.plot(ct_n, ct_m, "s-", color=C_CONV, label="ConvAE train step (batch 1)")
a.plot(st_n, st_m, "^-", color=C_SEN, label="Senseiver train step")
a.plot(gt_n, gt_m, "D-", color=C_VOX, label="voxel diffusion train step (batch 1)")
oom_marks_vox = [gt_oom[0]] if gt_oom else []
for n in oom_marks_vox:
    a.plot(n, H100, marker="x", ms=11, mew=2.6, color=C_VOX, ls="none")
a.axhline(H100, color="k", lw=0.8, ls=":")
oom_marks(a, ct_oom, H100)
a.set_title("Training", fontsize=10)
a.annotate("supervised fraction per step:\nforced 100% (grid)\nvs chosen 2%$\\to$0.004% (ours)",
           xy=(0.03, 0.68), xycoords="axes fraction", fontsize=7.5)
# (1,0) inference time
a = ax[1, 0]
a.plot(oi_n, oi_t, "o-", color=C_OURS)
a.plot(ci_n, ci_t, "s-", color=C_CONV)
a.plot(si_n, si_t, "^-", color=C_SEN)
a.plot(gi_n, [32 * t for t in gi_t], "D-", color=C_VOX)
oom_marks(a, ci_oom, max(ci_t) * 2)
a.set_ylabel("wall-clock (s)"); a.set_xlabel("output points")
a.annotate("linear in points;\ncontinues to $10^9$", xy=(0.55, 0.25), xycoords="axes fraction",
           fontsize=8, color=C_OURS)
# (1,1) training time
a = ax[1, 1]
a.plot(ot_n, ot_t, "o-", color=C_OURS)
a.plot(ct_n, ct_t, "s-", color=C_CONV)
a.plot(st_n, st_t, "^-", color=C_SEN)
a.plot(gt_n, gt_t, "D-", color=C_VOX)
a.set_xlabel("field points"); a.set_ylabel("s / step")

for a in ax.flat:
    a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=0.25, which="both", lw=0.4)
ax[0, 0].legend(fontsize=7.5, loc="center left")
ax[0, 1].legend(fontsize=7.5, loc="lower right")
fig.suptitle("Cost scaling with output resolution (H100, bf16; X = out-of-memory)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("png", "pdf"):
    if MEM_ONLY:
        for _k in (0, 1):
            ax[0, _k].set_xlabel("output points" if _k == 0 else "field points")
        fig.tight_layout()
    _name = "scaling_cost_mem" if MEM_ONLY else "scaling_cost"
    fig.savefig(f"{_name}.{ext}", bbox_inches="tight")
print("wrote scaling_cost.png/.pdf")
