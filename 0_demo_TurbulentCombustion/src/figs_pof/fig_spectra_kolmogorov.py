"""2D counterpart of the 3D spectral-fidelity panel: Kolmogorov vorticity
energy spectra by method on the canonical gallery frame (val 256), from the
field dumps in Save_TrainedModel_pof/field_dumps/kolm_*.npz
(dump_kolm_gallery.sh / dump_classical_gallery.py).

Protocol invariants (handoff sec.4): WINDOWED (Hann) spectra only -- the
Kolmogorov box is periodic, but the dumps are single-sample fields and the
window keeps the estimator identical to the 3D panel's; single posterior
sample per generative method (member 0), never the ensemble mean; the
deterministic rows use their point prediction. Output: E(k) per method and
the ratio to truth, with the inertial / dissipation bands shaded as in the 3D
figure. Missing dumps are skipped (re-run as they land).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pof_style import use_style, save, INK_2, MUTED  # noqa: E402

WT = Path(__file__).resolve().parents[2]
DUMPS = WT / "Save_TrainedModel_pof" / "field_dumps"
NX = NY = 256

# (label, npz, key)
PANELS = [
    ("DMF-Gen (ours)", "kolm_dmfgen.npz", "pred_sample"),
    ("SiT", "kolm_sit.npz", "pred_sample"),
    ("latent-FM", "kolm_latent_fm.npz", "pred_sample"),
    ("Senseiver", "kolm_senseiver.npz", "pred_mean"),
    # Geo-FNO is the accuracy leader on this regime (0.385 rel-L2 at 1%), so
    # the figure is not answering the distortion-perception question without
    # it. MLP-RBF is included for the same reason in reverse: it is a smooth
    # deterministic interpolant and marks where that family sits spectrally.
    ("Geo-FNO", "kolm_geofno.npz", "pred_mean"),
    ("MLP-RBF", "kolm_mlprbf.npz", "pred_mean"),
    ("IDW", "kolm_classical.npz", "pred_idw"),
    ("gappy POD $r$80", "kolm_classical.npz", "pred_gappy_pod_r80"),
]


def to_grid(vals: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Row-major (iy*NX+ix) point ordering -> [NY, NX]; verified from coords."""
    x = coords[:, 0]
    fast_x = np.ptp(x[:NX]) > 0
    g = vals.reshape(NY, NX) if fast_x else vals.reshape(NX, NY).T
    return g


def radial_spectrum(field: np.ndarray):
    """Hann-windowed 2D power spectrum, shell-averaged over |k| (integer bins)."""
    ny, nx = field.shape
    w = np.outer(np.hanning(ny), np.hanning(nx))
    f = field - field.mean()
    F = np.fft.fft2(f * w) / (w.sum())
    P = np.abs(F) ** 2
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    kbins = np.arange(0.5, min(nx, ny) // 2 + 0.5, 1.0)
    idx = np.digitize(K.ravel(), kbins)
    E = np.bincount(idx, weights=P.ravel(), minlength=kbins.size + 1)[1:kbins.size]
    k = 0.5 * (kbins[:-1] + kbins[1:])
    return k, E


def main() -> None:
    use_style()
    cls_path = DUMPS / "kolm_classical.npz"
    if not cls_path.is_file():
        raise SystemExit(f"need {cls_path} first (dump_classical_gallery.py)")
    cls = np.load(cls_path, allow_pickle=False)
    coords = np.asarray(cls["coords_raw"], dtype=np.float64)
    truth = to_grid(np.asarray(cls["truth"])[:, 0], coords)
    k, Et = radial_spectrum(truth)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                   gridspec_kw=dict(wspace=0.32, left=0.08,
                                                    right=0.985, top=0.84, bottom=0.2))
    ax1.loglog(k, Et, color="k", lw=1.4, label="truth")
    ax2.axhline(1.0, color="k", lw=0.8)
    # bands: inertial (k<=16) / dissipation (k>=48) -- 256^2, forcing at k=4
    for ax in (ax1, ax2):
        ax.axvspan(4, 16, color="#dfe9f6", zorder=0)
        ax.axvspan(48, k.max(), color="#f3e6e0", zorder=0)
    summary = {}
    for label, fname, key in PANELS:
        p = DUMPS / fname
        if not p.is_file():
            print(f"[skip] {label}: {fname} missing"); continue
        d = np.load(p, allow_pickle=False)
        if key not in d.files:
            print(f"[skip] {label}: {key} not in {fname}"); continue
        c = np.asarray(d["coords_raw"] if "coords_raw" in d.files else d["coords"], dtype=np.float64)
        pred = to_grid(np.asarray(d[key])[:, 0], c)
        # every dump is in its own run's z-units: rescale to the classical
        # dump's z-units through physical units before comparing spectra
        pm, ps = float(np.asarray(d["norm_mean"]).ravel()[0]), float(np.asarray(d["norm_std"]).ravel()[0])
        cm, cs = float(np.asarray(cls["norm_mean"]).ravel()[0]), float(np.asarray(cls["norm_std"]).ravel()[0])
        pred = (pred * ps + pm - cm) / cs
        _, E = radial_spectrum(pred)
        ax1.loglog(k, E, lw=1.0, label=label)
        ax2.semilogx(k, E / Et, lw=1.0, label=label)
        # BAND CHOICE IS A MEASUREMENT DECISION, NOT A CONVENTION.
        # On this frame the true spectrum peaks at k=3 and the k>=48 shells
        # carry 7e-4 of the peak, so a ratio there divides by ~nothing and
        # reports whatever a method puts into empty modes (values in the
        # hundreds). We therefore score fidelity on the bands that carry
        # energy, and report the far tail separately as SPURIOUS energy.
        inertial = (k >= 4) & (k <= 16)          # 29% of peak
        small = (k > 16) & (k <= 48)             # 3% -> 0.5% of peak
        tail = k > 48                            # 7e-4 of peak: diagnostic only
        summary[label] = {
            "inertial_ratio": float(np.mean(E[inertial] / Et[inertial])),
            "small_scale_ratio": float(np.mean(E[small] / Et[small])),
            "tail_excess_ratio": float(np.mean(E[tail] / Et[tail])),
            "tail_energy_fraction_of_truth_peak": float(Et[tail].mean() / Et.max()),
        }
    ax1.set_xlabel("$k$"); ax1.set_ylabel("$E_\\omega(k)$ (Hann-windowed)")
    ax2.set_xlabel("$k$"); ax2.set_ylabel("$E_\\omega(k)\\,/\\,E_\\omega^{\\rm truth}(k)$")
    ax2.set_ylim(0.0, 2.0)
    # MLP-RBF is smooth enough to fall past 1e-12, which would rescale the left
    # panel until every other curve collapsed into a band. Floor the axis at the
    # decade below the truth's own tail: everything below it is numerically
    # empty and carries no information for this comparison.
    ax1.set_ylim(max(Et.min() * 1e-3, 1e-10), None)
    ax2.axvline(48, color="0.5", lw=0.6, ls=":")
    ax1.legend(fontsize=6, frameon=False, ncol=2)
    # two lines: as a single string this ran far past the axes and tight-bbox
    # then stretched the saved figure to a 5:1 strip
    fig.text(0.08, 0.985,
             "Kolmogorov $256^2$, canonical frame (val 256), 655 vorticity sensors; one posterior "
             "sample per generative method. Shaded: energy-carrying inertial (blue)",
             fontsize=6.3, color=INK_2, va="top")
    fig.text(0.08, 0.945,
             "and small-scale (red) bands. Beyond the dotted line the true spectrum holds <0.1% of "
             "peak energy, so ratios there measure spurious energy, not fidelity.",
             fontsize=6.3, color=INK_2, va="top")
    save(fig, "spectra_kolmogorov", pdf_dpi=300, png_dpi=220)
    (WT / "Paper/pof2026/figures/spectra_kolmogorov_bands.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
