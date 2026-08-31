"""Merge per-case FireBench H5 extracts (u10, u12) into one training file.

Concatenates ``fields`` [1, n_t, n_p, 1, 1, C] along the time axis after
verifying identical coordinates/channel counts, preserving case order
(u10 first, then u12) so the temporally blocked split with guard gap
(JHU_SPLIT_MODE=block, JHU_SPLIT_GAP=10) keeps cases coherent, matching the
paper's 60+60 -> 120 protocol.

Streamed snapshot-by-snapshot: peak memory is one snapshot (~70 MB), so it
still belongs in an sbatch CPU job for I/O etiquette, not a login node.

Usage:
    python merge_firebench_cases.py --inputs u10.h5 u12.h5 \
        --out FireBench_u10u12_merged.h5 [--t-slice ::1] [--t-slice ::1]
--t-slice (one per input, python slice syntax start:stop:step) subsamples a
"dense" extract down to the protocol's snapshot count before merging.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def _parse_slice(text: str) -> slice:
    parts = (text or "::").split(":")
    if len(parts) > 3:
        raise ValueError(f"bad slice {text!r}")
    vals = [int(p) if p else None for p in parts] + [None] * (3 - len(parts))
    return slice(*vals)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--t-slice", action="append", default=None,
                   help="per-input time slice, e.g. '0:120:2' (repeatable)")
    args = p.parse_args()
    slices = ([_parse_slice(s) for s in args.t_slice]
              if args.t_slice else [slice(None)] * len(args.inputs))
    if len(slices) != len(args.inputs):
        raise SystemExit("--t-slice count must match --inputs")

    infos = []
    for path, sl in zip(args.inputs, slices):
        with h5py.File(path, "r") as f:
            shp = f["fields"].shape
            t_idx = np.arange(shp[1])[sl]
            infos.append({"path": path, "shape": shp, "t_idx": t_idx})
            print(f"{path}: fields{shp}, using {len(t_idx)}/{shp[1]} timesteps")
    base = infos[0]["shape"]
    for inf in infos[1:]:
        if inf["shape"][2:] != base[2:]:
            raise SystemExit(f"point/channel mismatch: {inf['shape']} vs {base}")

    with h5py.File(infos[0]["path"], "r") as f0:
        coords = f0["coordinates"][:]
    for inf in infos[1:]:
        with h5py.File(inf["path"], "r") as f:
            if not np.array_equal(f["coordinates"][:], coords):
                raise SystemExit(f"coordinates differ: {inf['path']}")

    n_t = sum(len(i["t_idx"]) for i in infos)
    out_shape = (1, n_t) + base[2:]
    with h5py.File(args.out, "w") as fo:
        fo.create_dataset("coordinates", data=coords)
        d = fo.create_dataset("fields", shape=out_shape, dtype="float32",
                              chunks=(1, 1) + base[2:])
        fo.attrs["merged_from"] = [i["path"] for i in infos]
        fo.attrs["case_n_t"] = [len(i["t_idx"]) for i in infos]
        k = 0
        for inf in infos:
            with h5py.File(inf["path"], "r") as f:
                for t in inf["t_idx"]:
                    d[0, k] = f["fields"][0, int(t)]
                    k += 1
            print(f"merged {inf['path']} -> cumulative {k}/{n_t}", flush=True)
    print(f"wrote {args.out}: fields{out_shape}")


if __name__ == "__main__":
    main()
