#!/usr/bin/env python3
"""Cross-Reynolds breakdown of the cylinder table (paper TODO "cross-Re breakdown").

The cylinder's held-out set is two Reynolds numbers, not one: frames 0-299 of
the val split are Re 80 and 300-599 are Re 250 (300 frames per Re, cross-Re
holdout ordering fixed in convert_cylinder.py). Every headline cylinder number
is an average over both, which hides the question a reader actually has --
does a method generalize DOWN in Reynolds number, UP, or neither?

It also matters for a claim the paper already makes elsewhere: the scalar
calibration repair fails on the cylinder precisely because the held-out set is
heterogeneous by construction (sec:calibration). This script measures that
heterogeneity on the accuracy side.

Classical rows are scored on all 600 val frames in their own JSONs; they are
subset here to the SAME 50 canonical frames the learned fleet uses, so the two
halves of the table are comparable frame for frame.

Usage:  python cross_re_cylinder.py [--json out.json] [--tex out.tex]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STM = ROOT / "Save_TrainedModel" / "cylinder2d"
RE_LO, RE_HI = 80, 250
SPLIT = 300          # val index < SPLIT -> Re 80, >= SPLIT -> Re 250

LEARNED = [("\\dmfgen{}", "pointcloud_ffm", "cyl_fleet_dmfgen_K8_nfe4.json"),
           ("SiT", "baseline_sit", "cyl_fleet_sit_K8_nfe50.json"),
           ("Latent FM", "baseline_latent_fm", "cyl_fleet_latentfm_K8_nfe4.json"),
           ("S3GM (PC-DPS)", "baseline_s3gm", "cyl_fleet_s3gm_K8_nfe200.json"),
           ("Senseiver", "baseline_det", "cyl_fleet_senseiver_K1.json"),
           ("MLP-RBF", "baseline_mlp_rbf", "cyl_fleet_mlprbf_K1.json"),
           ("Geo-FNO", "baseline_geofno", "cyl_fleet_geofno_K1.json")]
CLASSICAL_FILE = "classical_baselines_main_1pct_pod80_2d.json"
CLASSICAL = [("Gappy POD ($r{=}20$)", "gappy_pod_n238",
              "classical_baselines_main_1pct_pod20_2d.json"),
             ("IDW ($k{=}8$)", "idw_n238", CLASSICAL_FILE),
             ("Nearest neighbour", "kdtree_n238", CLASSICAL_FILE),
             ("Training mean", "constant_train_mean", CLASSICAL_FILE)]


def halves(per_snap, keep=None, field=None):
    """(mean rel-L2 at Re 80, at Re 250, n_lo, n_hi) over the kept frames.

    field=None scores the aggregate; field="p" scores the never-observed
    pressure channel, which is where a cross-Re gap should show up hardest --
    it is inferred entirely from cross-channel structure the training Reynolds
    numbers had to supply.
    """
    lo, hi = [], []
    for s in per_snap:
        i = int(s["snapshot"])
        if keep is not None and i not in keep:
            continue
        if field is not None:
            pf = s["per_field"]
            if field not in pf:
                return float("nan"), float("nan"), 0, 0
            v = float(pf[field]["rel_l2_mean"])
        else:
            # learned rows nest the metric under "aggregate"; the classical
            # per_snapshot records carry it at the top level
            v = float(s["aggregate"]["rel_l2_mean"] if "aggregate" in s
                      else s["rel_l2_mean"])
        (lo if i < SPLIT else hi).append(v)
    return (float(np.mean(lo)) if lo else float("nan"),
            float(np.mean(hi)) if hi else float("nan"), len(lo), len(hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()

    rows, frames = [], None
    for label, fam, fname in LEARNED:
        hits = sorted(glob.glob(str(STM / fam / "*" / "Evaluation" / fname)))
        if not hits:
            print(f"[skip] {label}: no {fname}")
            continue
        d = json.load(open(hits[-1]))
        ps = d["snapshots"]
        if frames is None:
            frames = {int(s["snapshot"]) for s in ps}
        elif {int(s["snapshot"]) for s in ps} != frames:
            raise SystemExit(f"[abort] {label} scores a different frame set; "
                             "the halves would not be comparable.")
        lo, hi, nl, nh = halves(ps)
        plo, phi, _, _ = halves(ps, field="p")
        rows.append({"row": label, "kind": "learned", "re80": lo, "re250": hi,
                     "n80": nl, "n250": nh, "ratio": hi / lo,
                     "p80": plo, "p250": phi})

    for label, key, fname in CLASSICAL:
        hits = sorted(glob.glob(str(STM / "baseline_classical" / fname)))
        if not hits:
            print(f"[skip] {label}: no {fname}")
            continue
        res = json.load(open(hits[-1]))["results"]
        if key not in res:
            print(f"[skip] {label}: {key} absent from {fname}")
            continue
        lo, hi, nl, nh = halves(res[key]["per_snapshot"], keep=frames)
        plo, phi, _, _ = halves(res[key]["per_snapshot"], keep=frames, field="p")
        rows.append({"row": label, "kind": "classical", "re80": lo, "re250": hi,
                     "n80": nl, "n250": nh, "ratio": hi / lo,
                     "p80": plo, "p250": phi})

    print(f"\ncylinder cross-Re, {len(frames)} canonical frames "
          f"({rows[0]['n80']} at Re 80, {rows[0]['n250']} at Re 250)")
    print("Re 80 is INTERPOLATION in Re (train 60,100,150,200 brackets it); "
          "Re 250 is EXTRAPOLATION past the training max.")
    print(f"{'row':<22}{'Re 80':>9}{'Re 250':>9}{'250/80':>9}{'p@80':>9}{'p@250':>9}")
    for r in rows:
        print(f"{r['row'].replace(chr(92)+'dmfgen{}','DMF-Gen'):<22}"
              f"{r['re80']:>9.3f}{r['re250']:>9.3f}{r['ratio']:>9.2f}"
              f"{r['p80']:>9.3f}{r['p250']:>9.3f}")

    if a.tex:
        lines = ["% GENERATED by src/cross_re_cylinder.py -- do not hand-edit"]
        ruled = False       # one rule, between the learned block and the floors
        for r in rows:
            if r["kind"] == "classical" and not ruled:
                lines.append("\\hline")
                ruled = True
            lines.append(f"{r['row']} & {r['re80']:.3f} & {r['re250']:.3f} "
                         f"& {r['ratio']:.2f} & {r['p80']:.3f} & {r['p250']:.3f} \\\\")
        Path(a.tex).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {a.tex}")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"frames": sorted(frames), "split_index": SPLIT,
             "re_lo": RE_LO, "re_hi": RE_HI, "rows": rows}, indent=1))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
