"""Shell-averaged spectra for non-periodic sub-cubes.

The JHU cutouts are sub-blocks of a larger DNS, so they are NOT periodic.
A plain FFT of such a block leaks the edge discontinuity across all
wavenumbers and produces a broadband floor: measured on our 125^3 cutout the
raw spectrum falls only 2.4x from k=32 to the Nyquist wavenumber, and a
deliberately smooth Gaussian-process draw -- whose true energy at k=62 is
e^-1700, i.e. exactly zero -- registers the same floor as the data. Any band
ratio computed above the crossover is then a ratio of two leakage floors and
tends to one regardless of the model.

Applying a separable Hann window before the transform restores the expected
decay (5.8e5x over the same range) at the cost of a mild low-k bias, and is
the estimator used for every spectral number we report.
"""

import numpy as np

_WINDOW_CACHE = {}


def _hann3(n):
    if n not in _WINDOW_CACHE:
        w = np.hanning(n)
        W = w[:, None, None] * w[None, :, None] * w[None, None, :]
        _WINDOW_CACHE[n] = (W / np.sqrt((W ** 2).mean())).astype(np.float64)
    return _WINDOW_CACHE[n]


def shell_spectrum(field3d, kmax=None, window=True):
    """Shell-summed energy spectrum E(k) of a cubic field, k = 1..kmax."""
    g = np.asarray(field3d, dtype=np.float64)
    n = g.shape[0]
    if window:
        g = g * _hann3(n)
    f = np.abs(np.fft.fftn(g)) ** 2
    kk = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
    kb = np.round(np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)).astype(int)
    spec = np.bincount(kb.ravel(), weights=f.ravel())
    kmax = kmax or (n // 2)
    return spec[1:kmax + 1]


def reliable_kmax(truth_spec, floor_ratio=10.0):
    """Largest k where the truth still sits a factor floor_ratio above its tail."""
    tail = np.median(truth_spec[-5:])
    ok = np.nonzero(truth_spec > floor_ratio * max(tail, 1e-300))[0]
    return int(ok[-1] + 1) if len(ok) else len(truth_spec)


def band_ratio(sample_spec, truth_spec, lo, hi):
    return float(sample_spec[lo - 1:hi].sum() / truth_spec[lo - 1:hi].sum())
