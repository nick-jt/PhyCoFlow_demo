"""PoF figure: windowed energy spectra + per-channel band-ratio bars.

The JHU cutouts are non-periodic, so every spectrum here uses the Hann-windowed
shell estimator (src/spectral_utils.py in the main checkout); unwindowed FFTs
have a leakage floor that invalidates dissipation-band ratios. Single posterior
samples only (ensemble means destroy small scales); spectra are averaged over
the 4 saved held-out snapshots, one sample each.

Verdict the figure carries: latent-FM holds more inertial-range energy than
DMF-Gen but truncates the dissipation band ~6x.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pof_style import (use_style, save, MAIN, OUT, FULL_W,
                       C_OURS, C_CONV, C_TRUTH, INK_2, MUTED)

sys.path.insert(0, str(MAIN / "src"))
from spectral_utils import shell_spectrum, reliable_kmax, band_ratio

SRC = MAIN / "Paper" / "iclr2027" / "figures"
N = 125
SNAPS = [0, 3, 12, 23]
INERTIAL = (8, 31)
DISSIP_LO = 32          # upper edge set below from the shaded band
DISSIP_HI = 50
CH = ["Ux", "Uy", "Uz"]
OBS = [True, False, True]   # Uy is the unobserved velocity component

use_style()
cache = OUT / "spectra_windowed_cache.npz"
if cache.exists():
    z = np.load(cache)
    spec = {k: z[k] for k in ("truth", "ours", "latent_fm")}
else:
    d = np.load(SRC / "spectra_fields.npz")
    spec = {}
    for key in ("truth", "ours", "latent_fm"):
        s = np.zeros((3, N // 2))
        for sn in SNAPS:
            f = d[f"{key}_{sn}"]
            for c in range(3):
                s[c] += shell_spectrum(f[:, c].reshape(N, N, N), window=True)
        spec[key] = s / len(SNAPS)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **spec)

tot = {k: v.sum(0) for k, v in spec.items()}
kmax = reliable_kmax(tot["truth"])
k = np.arange(1, kmax + 1)

ratios = {m: {"inertial": [band_ratio(spec[m][c], spec["truth"][c], *INERTIAL)
                           for c in range(3)],
              "dissip": [band_ratio(spec[m][c], spec["truth"][c],
                                    DISSIP_LO, DISSIP_HI) for c in range(3)]}
          for m in ("ours", "latent_fm")}
tot_r = {m: {"inertial": band_ratio(tot[m], tot["truth"], *INERTIAL),
             "dissip": band_ratio(tot[m], tot["truth"], DISSIP_LO, DISSIP_HI)}
         for m in ("ours", "latent_fm")}

fig = plt.figure(figsize=(FULL_W, 2.8), constrained_layout=True)
ga, gb, gc = fig.subplots(1, 3, width_ratios=[1.5, 1, 1])

# --- (a) total energy spectrum -------------------------------------------
ga.axvspan(*INERTIAL, color=C_OURS, alpha=0.05, lw=0)
ga.axvspan(DISSIP_LO, DISSIP_HI, color=C_CONV, alpha=0.07, lw=0)
for key, col, lab, lw in [("truth", C_TRUTH, "DNS truth", 1.6),
                          ("ours", C_OURS, "DMF-Gen (ours)", 1.3),
                          ("latent_fm", C_CONV, "latent FM", 1.3)]:
    ga.loglog(k, tot[key][:kmax], color=col, lw=lw, label=lab)
kk = np.array([8.0, 31.0])
ga.loglog(kk, 2.5 * tot["truth"][7] * (kk / 8.0) ** (-5 / 3), lw=0.7,
          color=MUTED)
ga.text(14, 5.5 * tot["truth"][13], r"$k^{-5/3}$", fontsize=6.5, color=MUTED)
ga.text(np.sqrt(8 * 31), 2.2e12, "inertial\n$k$ 8–31", fontsize=6.3,
        color=INK_2, ha="center", va="top")
ga.text(np.sqrt(32 * 50), 2.2e12, "dissipation\n$k$ 32–50", fontsize=6.3,
        color=INK_2, ha="center", va="top")
ga.text(np.sqrt(8 * 31), 4e3,
        f"$E/E_{{\\rm truth}}$:\nours {tot_r['ours']['inertial']:.2f}\n"
        f"lat-FM {tot_r['latent_fm']['inertial']:.2f}",
        fontsize=6.0, color=INK_2, ha="center", va="bottom")
ga.text(np.sqrt(32 * 50), 4e3,
        f"ours {tot_r['ours']['dissip']:.2f}\n"
        f"lat-FM {tot_r['latent_fm']['dissip']:.2f}",
        fontsize=6.0, color=INK_2, ha="center", va="bottom")
ga.set_xlim(1, kmax)
ga.set_ylim(1e3, 4e12)
ga.set_xlabel("wavenumber $k$")
ga.set_ylabel("$E(k)$")
ga.set_title("(a) energy spectrum (windowed, single samples)", fontsize=8)
ga.legend(loc="lower left", frameon=False, fontsize=6.3)
ga.grid(True, which="major", lw=0.4, alpha=0.6)

# --- (b),(c) per-channel band ratios -------------------------------------
x = np.arange(3)
w = 0.36
for ax, bandname, title in [(gb, "inertial", f"(b) inertial band  $k$ {INERTIAL[0]}–{INERTIAL[1]}"),
                            (gc, "dissip", f"(c) dissipation band  $k$ {DISSIP_LO}–{DISSIP_HI}")]:
    ro = ratios["ours"][bandname]
    rl = ratios["latent_fm"][bandname]
    b1 = ax.bar(x - w / 2, ro, w, color=C_OURS, label="DMF-Gen (ours)")
    b2 = ax.bar(x + w / 2, rl, w, color=C_CONV, label="latent FM")
    ax.axhline(1.0, color=MUTED, lw=0.7)
    ax.text(2.42, 1.015, "truth", fontsize=5.8, color=MUTED, ha="right")
    for bars, vals in [(b1, ro), (b2, rl)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=5.8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"${c[0]}_{c[1].lower()}$\n({'obs' if o else 'unobs'})"
                        for c, o in zip(CH, OBS)], fontsize=6.5)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=8)
    ax.grid(True, axis="y", lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
gb.set_ylabel("band energy  $E/E_{\\rm truth}$")
gb.legend(loc="upper left", frameon=False, fontsize=6.3)
gc.annotate("$\\sim$6$\\times$ truncation", xy=(1 + w / 2, 0.10),
            xytext=(0.98, 0.52), textcoords="axes fraction", ha="right",
            fontsize=6.5, color=C_CONV,
            arrowprops=dict(arrowstyle="-", color=C_CONV, lw=0.7))
fig.text(0.0, -0.02,
         "Hann-windowed shell spectra (cutouts are non-periodic; unwindowed "
         "FFTs leak a broadband floor); single posterior samples, spectra "
         f"averaged over 4 held-out snapshots; $k\\leq$ reliable $k_{{\\max}}$"
         f" = {kmax}.", fontsize=6.3, color=INK_2)

save(fig, "spectra_stats")

summary = {"kmax_reliable": kmax, "bands": {"inertial": list(INERTIAL),
                                            "dissip": [DISSIP_LO, DISSIP_HI]},
           "channels": CH, "observed": OBS,
           "per_channel": ratios, "total": tot_r}
json.dump(summary, open(OUT / "spectra_band_ratios.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
