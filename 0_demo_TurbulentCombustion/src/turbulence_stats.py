"""Distributional and physics diagnostics for turbulence reconstructions.

Adds the statistics emphasised by Raonic et al. (2026) for generative fluid
models -- one-point Wasserstein distance, higher central moments, longitudinal
structure functions -- plus a divergence / Leray-projection sensitivity check.

Estimator note, and it is the same trap that corrupted our spectra: the JHU
cutouts are sub-blocks of a larger DNS and are NOT periodic. Every derivative
here is therefore a second-order central difference evaluated on the interior,
never an FFT derivative. The one FFT-based quantity (Leray projection) is
explicitly flagged, since it assumes a periodicity the data does not have.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# One-point distributions
# ---------------------------------------------------------------------------

def w1_distance(a: np.ndarray, b: np.ndarray, n_quantiles: int = 4096) -> float:
    """1-Wasserstein distance between two empirical 1-D distributions.

    Computed as the mean absolute difference of matched quantiles, which is the
    closed form of W1 in one dimension. Quantile matching (rather than sorting
    both to equal length) lets the two samples differ in size.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def central_moments(x: np.ndarray) -> Dict[str, float]:
    """Variance, skewness and (non-excess) kurtosis of a field.

    Turbulent velocity increments are strongly non-Gaussian; a model that
    matches the variance but returns Gaussian tails is over-smoothed in a way
    relative L2 does not register.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mu = x.mean()
    d = x - mu
    var = float((d ** 2).mean())
    s = np.sqrt(var) + 1e-30
    return {
        "mean": float(mu),
        "variance": var,
        "skewness": float((d ** 3).mean() / s ** 3),
        "kurtosis": float((d ** 4).mean() / s ** 4),
    }


# ---------------------------------------------------------------------------
# Structure functions
# ---------------------------------------------------------------------------

def longitudinal_structure_function(
    vel: np.ndarray, orders: Sequence[int] = (2, 3), max_r: int | None = None
) -> Dict[int, np.ndarray]:
    """S_p(r) = < [ (u(x + r e) - u(x)) . e ]^p >, averaged over the three axes.

    ``vel`` is [n, n, n, 3]. Separations are whole grid cells. The third-order
    function is the sharp one: Kolmogorov's 4/5 law makes S_3(r) negative and
    linear in r through the inertial range, and its sign is set by the energy
    cascade, so a model that produces plausible-looking fields with no cascade
    fails here while matching S_2.
    """
    vel = np.asarray(vel, dtype=np.float64)
    n = vel.shape[0]
    max_r = max_r or n // 4
    out = {p: np.zeros(max_r) for p in orders}
    for r in range(1, max_r + 1):
        acc = {p: [] for p in orders}
        for ax in range(3):
            # longitudinal component: displace along `ax`, project onto `ax`
            u = np.take(vel[..., ax], indices=range(n - r), axis=ax)
            u_r = np.take(vel[..., ax], indices=range(r, n), axis=ax)
            d = (u_r - u).ravel()
            for p in orders:
                acc[p].append(np.mean(d ** p))
        for p in orders:
            out[p][r - 1] = float(np.mean(acc[p]))
    return out


# ---------------------------------------------------------------------------
# Incompressibility
# ---------------------------------------------------------------------------

def divergence(vel: np.ndarray) -> np.ndarray:
    """Second-order central-difference divergence on the interior.

    Returns the [n-2, n-2, n-2] interior block. Finite differences rather than
    an FFT derivative because the cutout is not periodic.
    """
    vel = np.asarray(vel, dtype=np.float64)
    d = np.zeros_like(vel[1:-1, 1:-1, 1:-1, 0])
    for ax in range(3):
        f = vel[..., ax]
        g = 0.5 * (np.roll(f, -1, axis=ax) - np.roll(f, 1, axis=ax))
        d += g[1:-1, 1:-1, 1:-1]
    return d


def divergence_metrics(vel: np.ndarray) -> Dict[str, float]:
    """Divergence normalised by a velocity-gradient scale.

    The raw magnitude of div(u) is meaningless without a scale; we normalise by
    the RMS of the individual velocity derivatives, so the number answers "how
    large is the failure of incompressibility relative to a typical gradient".
    """
    vel = np.asarray(vel, dtype=np.float64)
    div = divergence(vel)
    grads = []
    for ax in range(3):
        f = vel[..., ax]
        g = 0.5 * (np.roll(f, -1, axis=ax) - np.roll(f, 1, axis=ax))
        grads.append(g[1:-1, 1:-1, 1:-1])
    scale = float(np.sqrt(np.mean(np.stack(grads) ** 2))) + 1e-30
    return {
        "div_rms": float(np.sqrt(np.mean(div ** 2))),
        "grad_rms": scale,
        "div_rms_normalized": float(np.sqrt(np.mean(div ** 2)) / scale),
        "div_max_normalized": float(np.abs(div).max() / scale),
    }


def leray_project(vel: np.ndarray) -> np.ndarray:
    """Project onto the divergence-free subspace in Fourier space.

    CAVEAT: this assumes periodicity, which a DNS sub-block does not satisfy.
    The projection is therefore only indicative; it is used to ask whether
    enforcing incompressibility would move the statistics at all, not to
    produce a corrected field. Compare projected-vs-unprojected statistics and
    read a small difference as "the model already respects the constraint well
    enough that enforcing it changes nothing measurable".
    """
    vel = np.asarray(vel, dtype=np.float64)
    n = vel.shape[0]
    k = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k2[0, 0, 0] = 1.0
    vh = [np.fft.fftn(vel[..., i]) for i in range(3)]
    kdotv = KX * vh[0] + KY * vh[1] + KZ * vh[2]
    out = np.empty_like(vel)
    for i, K in enumerate((KX, KY, KZ)):
        out[..., i] = np.real(np.fft.ifftn(vh[i] - K * kdotv / k2))
    return out
