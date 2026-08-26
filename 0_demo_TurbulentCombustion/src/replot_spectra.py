"""Regenerate the spectra/turbulence-statistics figure from the saved JSON."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("../Paper/iclr2027/figures")
res = json.load(open(OUT / "spectra_stats.json"))
ks = np.array(res["k"]); rs = np.array(res["r"]); ctr = np.array(res["grad_bins"])
styles = [("truth", "k-", "DNS truth"),
          ("ours", "C2-", "LOCUS (ambient)"),
          ("ours_nospec", "C0-.", "LOCUS, no spectral loss"),
          ("latent_fm", "C1--", "latent FM")]
fig, axes = plt.subplots(1, 3, figsize=(13, 2.9), dpi=200)
for key, st, lab in styles:
    axes[0].loglog(ks, res[key]["spectrum"], st, lw=1.5, label=lab)
    axes[1].loglog(rs, res[key]["S2"], st, lw=1.5)
    axes[2].semilogy(ctr, res[key]["grad_pdf"], st, lw=1.5)
axes[0].axvspan(32, 50, color="gray", alpha=0.12)
axes[0].text(40, 3e11, "dissipation\nband", fontsize=7, color="0.35", ha="center")
axes[0].axvspan(50, ks[-1], color="0.85", alpha=0.5)
axes[0].text(56, 3e11, "data\nlimit", fontsize=6.5, color="0.45", ha="center")
axes[0].set_xlabel("wavenumber $k$"); axes[0].set_ylabel("$E(k)$")
axes[0].set_title("Energy spectrum (single samples)", fontsize=10)
axes[0].legend(fontsize=7, frameon=False, loc="lower left")
axes[1].set_xlabel("separation $r$ (grid units)"); axes[1].set_ylabel("$S_2(r)$")
axes[1].set_title("Longitudinal structure function", fontsize=10)
axes[2].set_xlabel(r"$\partial_x u\,/\,\sigma$"); axes[2].set_ylabel("PDF")
axes[2].set_ylim(1e-5, 1.5)
axes[2].set_title("Velocity-gradient PDF (intermittency)", fontsize=10)
fig.tight_layout()
axes[0].set_ylim(1e5, None)
fig.savefig(OUT / "spectra_stats.pdf")
fig.savefig(OUT / "spectra_stats.png")
# quantitative summary for the paper text
t = np.array(res["truth"]["spectrum"])
for key in ("ours", "ours_nospec", "latent_fm"):
    s = np.array(res[key]["spectrum"])
    s2 = np.array(res[key]["S2"]); s2t = np.array(res["truth"]["S2"])
    pdf = np.array(res[key]["grad_pdf"]); pdft = np.array(res["truth"]["grad_pdf"])
    m = pdft > 0
    kl = float(np.sum(pdft[m] * np.log(pdft[m] / np.maximum(pdf[m], 1e-12))) *
               (ctr[1] - ctr[0]))
    print(f"{key}: inertial {s[7:31].sum()/t[7:31].sum():.3f} "
          f"dissip {s[31:62].sum()/t[31:62].sum():.3f} "
          f"S2(r=1) ratio {s2[0]/s2t[0]:.3f} gradPDF KL {kl:.4f}")
print("wrote figure")
