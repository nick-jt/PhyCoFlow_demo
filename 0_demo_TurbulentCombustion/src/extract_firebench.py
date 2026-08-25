"""Extract a 3D FireBench crop into the TurbulentCombustion H5 layout.

Reads the public FireBench Zarr store (Wang et al., arXiv:2406.08589,
gs://firebench) anonymously and writes
    fields       [1, n_t, n_p, 1, 1, 5]   (u, v, w, theta, rho_f)
    coordinates  [n_p, 1, 1, 3]
so TurbulentCombustionH5Dataset loads it unchanged.

Axis convention of the store (verified against theta/rho_f structure at
t=100: fire front at dim0 idx 296-416, line ignition spanning all of dim1,
combustion confined to dim2 idx < ~100 with the fuel bed at dim2=0):
    dim0 = streamwise x, 1528 pts @ 1 m
    dim1 = lateral y,     252 pts @ 1 m
    dim2 = height z,     3584 pts @ 0.5 m  (mostly quiescent sky above ~200 m)

Crop: case u10/ramp0; x [250, 706) stride 3 (fire ignition at 300 m plus
spread margin, single x-chunk so the transfer stays bounded), full lateral
stride 2, height [0, 384) stride 2 (0-192 m: fuel bed, flame zone, plume);
timesteps 40..145 step 5.
"""

import numpy as np
import h5py
import zarr
import gcsfs

import os
CASE = os.environ.get("FB_CASE", "firebench/v2024.04/u10/ramp0/fire.zarr")
OUT = os.environ.get("FB_OUT",
    "/projects/ammoniacomb/generative_reconstruction/firebench3d/"
    "FireBench_u10_ramp0_3D.h5")
VARS = ["u", "v", "w", "theta", "rho_f"]

T_IDX = list(range(int(os.environ.get("FB_T0", "30")), 150,
                   int(os.environ.get("FB_STRIDE", "2"))))
X_SL = slice(250, 706, 3)                # 152 pts, dx = 3 m (streamwise)
H_SL = slice(0, 252, 2)                  # 126 pts, dy = 2 m (lateral)
Y_SL = slice(0, 384, 2)                  # 192 pts, dz = 1 m (height)

DX, DH, DY = 1.0, 1.0, 0.5               # native spacings (m)


def main():
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fs = gcsfs.GCSFileSystem(token="anon")
    store = zarr.open_group(fs.get_mapper(CASE), mode="r")

    xs = (np.r_[X_SL] * DX).astype(np.float32)
    hs = (np.r_[H_SL] * DH).astype(np.float32)
    ys = (np.r_[Y_SL] * DY).astype(np.float32)
    nx, nh, ny = len(xs), len(hs), len(ys)
    n_p = nx * nh * ny
    print(f"crop: {nx} x {nh} x {ny} = {n_p:,} points, {len(T_IDX)} snapshots")

    X, H, Y = np.meshgrid(xs, hs, ys, indexing="ij")
    coords = np.stack([X.ravel(), H.ravel(), Y.ravel()], axis=-1)

    with h5py.File(OUT, "w") as f:
        f.create_dataset("coordinates", data=coords[:, None, None, :])
        dset = f.create_dataset(
            "fields", shape=(1, len(T_IDX), n_p, 1, 1, len(VARS)),
            dtype=np.float32,
        )
        f.create_dataset("time", data=np.asarray(T_IDX, dtype=np.float32))
        f.create_dataset("field_names",
                         data=np.array([v.encode() for v in VARS]))
        f.attrs.update({
            "source": "gs://firebench/v2024.04 u10/ramp0",
            "grid_nx_nh_ny": [nx, nh, ny],
            "stride_x_h_y": [3, 2, 2],
            "axes": "x (streamwise), y (lateral), z (height)",
        })
        for ti, t in enumerate(T_IDX):
            for vi, var in enumerate(VARS):
                block = store[var][t, X_SL, H_SL, Y_SL]
                dset[0, ti, :, 0, 0, vi] = block.ravel()
            print(f"  t={t} done ({ti + 1}/{len(T_IDX)})", flush=True)

    print("wrote", OUT)


if __name__ == "__main__":
    main()
