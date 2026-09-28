#!/usr/bin/env python3
"""Derive the velocity field (u, v) from vorticity.npy on the periodic box.

The upstream dataset ships vorticity only. On [0, 2pi)^2 with periodic
boundaries the velocity is recovered exactly (up to the spatial-mean flow,
which vorticity does not determine and which is set to zero here) through the
streamfunction:

    omega = dv/dx - du/dy,   laplacian(psi) = -omega,
    u = dpsi/dy,             v = -dpsi/dx

solved spectrally. The sign convention omega = v_x - u_y is the standard one
and is ASSUMED: the upstream files do not state theirs. If upstream used the
opposite sign, (u, v) flips sign; nothing else changes.

Writes velocity.npy with shape (traj, frame, y, x, 2) = [u, v], float32
(~1.7 GB). Modes on the grid-Nyquist lines (|kx| or |ky| = 128), whose odd
derivatives are undefined on an even grid, are dropped; they carry ~1e-5 of
the vorticity amplitude. Self-check: the spectral curl of the result
reproduces the input vorticity (minus its spatial mean and those modes) to
machine precision.

Usage:  python scripts/derive_velocity.py [--data DIR] [--out velocity.npy]
"""

import argparse
import os

import numpy as np


def velocity_from_vorticity(w: np.ndarray):
    """w: (..., ny, nx) on [0,2pi)^2 -> (u, v), same shape, float64."""
    ny, nx = w.shape[-2:]
    kx = np.fft.fftfreq(nx, d=1.0 / nx)[None, :]  # integer wavenumbers (L = 2pi)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)[:, None]
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    wh = np.fft.fft2(w.astype(np.float64), axes=(-2, -1))
    psih = wh / k2
    psih[..., 0, 0] = 0.0
    psih[..., ny // 2, :] = 0.0  # Nyquist lines: no well-defined odd derivative
    psih[..., :, nx // 2] = 0.0
    u = np.fft.ifft2(1j * ky * psih, axes=(-2, -1)).real
    v = np.fft.ifft2(-1j * kx * psih, axes=(-2, -1)).real
    return u, v


def curl(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    ny, nx = u.shape[-2:]
    kx = np.fft.fftfreq(nx, d=1.0 / nx)[None, :]
    ky = np.fft.fftfreq(ny, d=1.0 / ny)[:, None]
    # The Nyquist mode has no well-defined odd derivative on an even grid.
    kx = np.where(np.abs(kx) == nx // 2, 0.0, kx)
    ky = np.where(np.abs(ky) == ny // 2, 0.0, ky)
    vx = np.fft.ifft2(1j * kx * np.fft.fft2(v, axes=(-2, -1)), axes=(-2, -1)).real
    uy = np.fft.ifft2(1j * ky * np.fft.fft2(u, axes=(-2, -1)), axes=(-2, -1)).real
    return vx - uy


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=here)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    w = np.load(os.path.join(args.data, "vorticity.npy"), mmap_mode="r")
    out = args.out or os.path.join(args.data, "velocity.npy")
    vel = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=w.shape + (2,))
    worst = 0.0
    for tj in range(w.shape[0]):
        u, v = velocity_from_vorticity(w[tj])
        vel[tj, ..., 0] = u
        vel[tj, ..., 1] = v
        if tj in (0, w.shape[0] - 1):
            rh = np.fft.fft2(np.asarray(w[tj], dtype=np.float64), axes=(-2, -1))
            rh[..., 0, 0] = 0.0
            rh[..., w.shape[-2] // 2, :] = 0.0
            rh[..., :, w.shape[-1] // 2] = 0.0
            ref = np.fft.ifft2(rh, axes=(-2, -1)).real
            err = np.abs(curl(u, v) - ref).max() / np.abs(ref).max()
            worst = max(worst, err)
    vel.flush()
    print(f"wrote {out} {vel.shape}; max relative curl round-trip error {worst:.2e}")


if __name__ == "__main__":
    main()
