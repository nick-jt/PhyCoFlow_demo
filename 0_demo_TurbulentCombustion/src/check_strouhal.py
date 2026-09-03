"""Strouhal-number V&V for the generated 2D cylinder cases.

Reads postProcessing/forceCoeffs1/*/forceCoeffs.dat (OpenFOAM 9 layout;
falls back to coefficient.dat), FFTs the lift coefficient over the sampling
window (t >= 120), and prints St = f D / U (D = U = 1) plus mean Cd.

Literature anchors (Williamson 1996; Norberg 2003): St(Re=60) ~ 0.135,
St(100) ~ 0.164, St(150) ~ 0.183, St(200) ~ 0.196. Mean Cd(100) ~ 1.33.
"""

import argparse
import glob
import os

import numpy as np


def load_coeffs(case_dir: str) -> np.ndarray:
    pats = [
        os.path.join(case_dir, "postProcessing", "forceCoeffs1", "*", "forceCoeffs.dat"),
        os.path.join(case_dir, "postProcessing", "forceCoeffs1", "*", "coefficient.dat"),
    ]
    files = sorted(f for p in pats for f in glob.glob(p))
    if not files:
        raise FileNotFoundError(f"no forceCoeffs output under {case_dir}")
    rows = []
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    rows.append([float(parts[0]), float(parts[2]), float(parts[3])])
    arr = np.array(rows)  # time, Cd, Cl (OF9 columns: Time Cm Cd Cl Cl(f) Cl(r))
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    _, keep = np.unique(arr[:, 0], return_index=True)
    return arr[keep]


def strouhal(case_dir: str, t_min: float = 120.0):
    arr = load_coeffs(case_dir)
    m = arr[:, 0] >= t_min
    if m.sum() < 100:
        raise RuntimeError(f"{case_dir}: only {m.sum()} samples past t={t_min}")
    t, cd, cl = arr[m, 0], arr[m, 1], arr[m, 2]
    tu = np.linspace(t[0], t[-1], len(t))
    clu = np.interp(tu, t, cl)
    clu = clu - clu.mean()
    freqs = np.fft.rfftfreq(len(tu), d=tu[1] - tu[0])
    power = np.abs(np.fft.rfft(clu * np.hanning(len(clu)))) ** 2
    st = freqs[1:][np.argmax(power[1:])]
    return st, float(cd.mean()), float(np.abs(clu).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/projects/ammoniacomb/generative_reconstruction/cylinder2d/runs")
    ap.add_argument("--t-min", type=float, default=120.0)
    args = ap.parse_args()
    for case in sorted(glob.glob(os.path.join(args.runs, "Re*"))):
        try:
            st, cd, cl_amp = strouhal(case, args.t_min)
            print(f"{os.path.basename(case):8s}  St = {st:.4f}   mean Cd = {cd:.3f}   |Cl| amp = {cl_amp:.3f}")
        except Exception as exc:
            print(f"{os.path.basename(case):8s}  FAILED: {exc}")


if __name__ == "__main__":
    main()
