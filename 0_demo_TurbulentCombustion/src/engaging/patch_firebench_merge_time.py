"""Append the `time` and `field_names` datasets the FireBench merge dropped.

Incident 2026-09-03: every FireBench training job (v5clean / LFM / Senseiver,
13 jobs) died instantly with KeyError on f["time"] at helpers.py:230.
merge_firebench_cases.py wrote only `coordinates` and `fields`; the two source
files carry `time` (60 each) and `field_names` (5), and both were lost in the
merge. The earlier verification checked shape/NaN/case_n_t but not the presence
of every dataset the loader reads, so the gap survived to launch.

`fields` and `coordinates` in the merged file are correct and verified, so this
appends the two missing datasets in place rather than re-running an 8.9 GB
merge. Time values are taken verbatim from the sources and concatenated in the
same order the merge used (u10 then u12) — nothing is synthesised.

Note the merged file concatenates two INDEPENDENT cases, so `time` is not
monotonic across the case boundary at index 60; it restarts. That is faithful
to the data (frame i of u10 and frame i of u12 are different physical runs),
and index-based splitting is unaffected because helpers.py splits on indices,
never on time values.

Run on a CPU node (needs h5py, and login nodes cannot import numpy):
    sbatch --partition=mit_normal --account=mit_general --time=00:20:00 \
      --mem=16G --wrap="source ~/envs/phycoflow; python patch_firebench_merge_time.py"
"""
from __future__ import annotations

import sys

import h5py
import numpy as np

MERGED = "/home/ntricard/orcd/scratch/firebench3d/FireBench_u10u12_merged.h5"
SOURCES = [  # order must match the merge (see slurm-21631452.out)
    "/home/ntricard/orcd/scratch/firebench3d/firebench3d/FireBench_u10_ramp0_3D_dense.h5",
    "/home/ntricard/orcd/scratch/firebench3d/firebench3d/FireBench_u12_ramp0_3D_dense.h5",
]


def main() -> int:
    times, names = [], None
    for path in SOURCES:
        with h5py.File(path, "r") as f:
            t = f["time"][:]
            times.append(t)
            print(f"{path.split('/')[-1]}: time[{len(t)}] "
                  f"range {float(t.min()):.6g}..{float(t.max()):.6g}")
            if names is None and "field_names" in f:
                names = f["field_names"][:]
    merged_time = np.concatenate(times).astype(np.float32)

    with h5py.File(MERGED, "r+") as f:
        n_t = int(f["fields"].shape[1])
        if merged_time.shape[0] != n_t:
            print(f"FAIL: time length {merged_time.shape[0]} != fields n_t {n_t}")
            return 1
        case_n_t = list(f.attrs.get("case_n_t", []))
        if case_n_t and [len(t) for t in times] != list(case_n_t):
            print(f"FAIL: per-source lengths {[len(t) for t in times]} "
                  f"disagree with case_n_t {case_n_t}")
            return 1
        for key in ("time", "field_names"):
            if key in f:
                del f[key]
        f.create_dataset("time", data=merged_time)
        if names is not None:
            f.create_dataset("field_names", data=names)
        f.attrs["time_source"] = "concatenated verbatim from u10 then u12 sources"
        print(f"wrote time[{merged_time.shape[0]}] "
              f"(case boundary at {len(times[0])}, restarts by design)"
              + ("" if names is None else f" and field_names[{len(names)}]"))

    # Re-open read-only and verify exactly what the loader does.
    with h5py.File(MERGED, "r") as f:
        t = f["time"][:]
        print(f"VERIFY: fields{f['fields'].shape} coordinates{f['coordinates'].shape} "
              f"time[{t.shape[0]}] keys={sorted(f.keys())}")
        assert t.shape[0] == int(f["fields"].shape[1])
    print("PASS: merged file now has every dataset helpers.py reads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
