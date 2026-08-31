"""Post-download sanity check for the arriving datasets (metadata + a few
slices only -- login-node safe).

Prints, per file: HDF5 keys, shapes, dtypes, attrs, coordinate ranges, and a
NaN/scale check on the first and last snapshot of each field. Compare against
the protocol expectations before launching anything:
  JHU cross-cube : fields [1, T, n_p, 1, 1, 4] with 4x50 snapshots of
                   125^3 cutouts (layout to be confirmed on arrival),
                   field order (Ux, Uy, Uz, p)
  FireBench case : fields [1, T, 3677184, 1, 1, 5] (152x126x192),
                   (u, v, w, theta, rho_f); protocol uses 60 snaps/case
"""

from __future__ import annotations

import sys

import h5py
import numpy as np


def inspect(path: str) -> None:
    print(f"\n=== {path} ===")
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape} dtype={obj.dtype} "
                      f"chunks={obj.chunks}")
        f.visititems(show)
        for k, v in f.attrs.items():
            print(f"  attr {k} = {v}")
        if "coordinates" in f:
            c = f["coordinates"][:, 0, 0, :]
            print(f"  coords: n_p={c.shape[0]} min={c.min(0)} max={c.max(0)}")
        if "fields" in f:
            d = f["fields"]
            for t in {0, d.shape[1] - 1}:
                snap = d[0, t]
                flat = snap.reshape(-1, snap.shape[-1])
                print(f"  snap {t}: nan={int(np.isnan(flat).sum())} "
                      f"mean={flat.mean(0).round(4)} std={flat.std(0).round(4)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            inspect(p)
        except Exception as exc:
            print(f"{p}: FAILED to inspect: {exc}")
