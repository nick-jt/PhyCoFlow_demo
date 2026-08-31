"""LDW-FFM Stage-1 TUNE/TEST analysis (PLAN_IMPROVE_2026-08-30 section 5).

Consumes the sufficient-statistics JSONs written by ldw_ffm_stage1.py
(one per sensor density) and performs the actual experiment:

  TUNE split = cube-3 ODD val indices {1,3,...,49} (25 snaps)
  TEST split = EVEN indices {0,2,...,48}, untouched by fitting

Per channel and density the optimal blend weight has the closed form
  w* = sum_TUNE(a - c) / sum_TUNE(a + b - 2c),   clipped to [0,1]
(from ||u - (w IDW + (1-w) F)||^2 = a - 2w(a-c) + w^2(a+b-2c)).

TEST-split relL2 at any w is exact from (a,b,c,u2); CRPS / coverage /
rmse of the blended ensemble are linearly interpolated on the 0.01 w-grid
(the blend of members is affine per point, so the grids are smooth).

Spread of the blended ensemble is EXACTLY (1-w) * spread(FFM): the same
deterministic IDW field is added to every member, so blending scales the
predictive spread down by (1-w). This is stated, not hidden.

Pure post-processing of saved statistics -- no sensors are drawn and no
model runs, so this script is safe to run anywhere (no compute-node gate).
"""

from __future__ import annotations

import argparse
import json
import numpy as np

FIELD_NAMES = ("Ux", "Uy", "Uz", "p")
OBSERVED = ("Ux", "Uz")
DENSITY_PCT = {1953: "0.1%", 19531: "1%", 195312: "10%"}


def rel_l2_at(s: dict, w: float) -> float:
    a, b, c, u2 = s["a"], s["b"], s["c"], s["u2"]
    err2 = a - 2.0 * w * (a - c) + w * w * (a + b - 2.0 * c)
    return float(np.sqrt(max(err2, 0.0)) / (np.sqrt(u2) + 1e-12))


def interp_grid(grid_vals, w_grid, w: float) -> float:
    return float(np.interp(w, w_grid, np.asarray(grid_vals)))


def fit_w(snaps, field: str) -> float:
    num = sum(s["per_field"][field]["a"] - s["per_field"][field]["c"]
              for s in snaps)
    den = sum(s["per_field"][field]["a"] + s["per_field"][field]["b"]
              - 2.0 * s["per_field"][field]["c"] for s in snaps)
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def split(snaps):
    tune = [s for s in snaps if s["snapshot"] % 2 == 1]
    test = [s for s in snaps if s["snapshot"] % 2 == 0]
    return tune, test


def eval_at_w(snaps, field: str, w: float, w_grid) -> dict:
    """Mean-over-snapshots metrics of the blend at fixed w (ensemble_eval
    convention: per-snapshot metric, then averaged across snapshots)."""
    rel, crps, cov90, cov50, sp_err, spread, rmse = [], [], [], [], [], [], []
    for s in snaps:
        f = s["per_field"][field]
        rel.append(rel_l2_at(f, w))
        crps.append(interp_grid(f["crps_grid"], w_grid, w))
        cov90.append(interp_grid(f["cov90_grid"], w_grid, w))
        cov50.append(interp_grid(f["cov50_grid"], w_grid, w))
        rm = interp_grid(f["rmse_grid"], w_grid, w)
        sp = (1.0 - w) * f["spread0"]
        rmse.append(rm)
        spread.append(sp)
        sp_err.append(sp / (rm + 1e-12))
    return {k: float(np.mean(v)) for k, v in
            [("rel_l2", rel), ("crps", crps), ("cov90", cov90),
             ("cov50", cov50), ("spread", spread), ("rmse", rmse),
             ("spread_error_ratio", sp_err)]}


def per_snap_improvement(snaps, field: str, w: float):
    """Count TEST snapshots where the blend beats each fallback on relL2."""
    beat_ffm = beat_idw = 0
    for s in snaps:
        f = s["per_field"][field]
        r = rel_l2_at(f, w)
        if r < rel_l2_at(f, 0.0):
            beat_ffm += 1
        if r < rel_l2_at(f, 1.0):
            beat_idw += 1
    return beat_ffm, beat_idw, len(snaps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="ldw_ffm_stage1_n{NOBS}.json files (one per density)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data = {}
    for path in args.inputs:
        d = json.load(open(path))
        n = d["protocol"]["n_obs"][0]
        data[n] = d
        # sanity: closed-form relL2 must match the stored grid at the ends
        for s in d["snapshots"][:3]:
            for f in FIELD_NAMES:
                pf = s["per_field"][f]
                assert abs(rel_l2_at(pf, 0.0) - pf["rel_l2_grid"][0]) < 1e-6
                assert abs(rel_l2_at(pf, 1.0) - pf["rel_l2_grid"][-1]) < 1e-6

    densities = sorted(data.keys())
    out = {"densities": densities, "per_density": {}, "shared_w": {},
           "verdict": {}}

    # ---------------- TUNE: fit w* ----------------
    print("=" * 78)
    print("w* fitted on TUNE split (odd cube-3 snapshots), per channel")
    print("=" * 78)
    hdr = f"{'density':>10s} " + " ".join(f"{f:>8s}" for f in FIELD_NAMES)
    print(hdr)
    wstar = {}
    for n in densities:
        snaps = data[n]["snapshots"]
        tune, test = split(snaps)
        assert len(tune) == 25 and len(test) == 25, (len(tune), len(test))
        wstar[n] = {f: fit_w(tune, f) for f in FIELD_NAMES}
        print(f"{DENSITY_PCT.get(n, n):>10s} " +
              " ".join(f"{wstar[n][f]:8.3f}" for f in FIELD_NAMES))

    # density-shared w (pooled TUNE sums across densities)
    shared = {}
    for f in FIELD_NAMES:
        pooled = [s for n in densities for s in split(data[n]["snapshots"])[0]]
        shared[f] = fit_w(pooled, f)
    print(f"{'shared':>10s} " +
          " ".join(f"{shared[f]:8.3f}" for f in FIELD_NAMES))
    spread_across = {f: max(wstar[n][f] for n in densities)
                     - min(wstar[n][f] for n in densities) for f in FIELD_NAMES}
    print("w* range across densities: " +
          " ".join(f"{f}:{spread_across[f]:.3f}" for f in FIELD_NAMES))
    out["shared_w"] = shared
    out["wstar_range_across_densities"] = spread_across

    # ---------------- TEST: frozen w* ----------------
    overall_pass = True
    for n in densities:
        d = data[n]
        w_grid = np.asarray(d["w_grid"])
        tune, test = split(d["snapshots"])
        print()
        print("=" * 78)
        print(f"TEST split (even snapshots, untouched), density "
              f"{DENSITY_PCT.get(n, n)} ({n}/channel)")
        print("=" * 78)
        print(f"{'ch':>4s} {'w*':>6s} | {'FFM':>7s} {'IDW':>7s} {'blend':>7s} "
              f"{'d%vsFFM':>8s} {'d%vsIDW':>8s} | {'>FFM':>6s} {'>IDW':>6s} | "
              f"{'CRPS_F':>7s} {'CRPS_b':>7s} {'sp/eF':>6s} {'sp/eb':>6s} "
              f"{'cov90F':>7s} {'cov90b':>7s}")
        res_n = {}
        for f in FIELD_NAMES:
            w = wstar[n][f]
            m0 = eval_at_w(test, f, 0.0, w_grid)     # FFM alone
            m1 = eval_at_w(test, f, 1.0, w_grid)     # IDW alone
            mb = eval_at_w(test, f, w, w_grid)       # blend
            bF, bI, nt = per_snap_improvement(test, f, w)
            dF = 100.0 * (mb["rel_l2"] - m0["rel_l2"]) / m0["rel_l2"]
            dI = 100.0 * (mb["rel_l2"] - m1["rel_l2"]) / m1["rel_l2"]
            print(f"{f:>4s} {w:6.3f} | {m0['rel_l2']:7.4f} {m1['rel_l2']:7.4f} "
                  f"{mb['rel_l2']:7.4f} {dF:+8.2f} {dI:+8.2f} | "
                  f"{bF:3d}/{nt:<2d} {bI:3d}/{nt:<2d} | "
                  f"{m0['crps']:7.4f} {mb['crps']:7.4f} "
                  f"{m0['spread_error_ratio']:6.3f} "
                  f"{mb['spread_error_ratio']:6.3f} "
                  f"{m0['cov90']:7.4f} {mb['cov90']:7.4f}")
            res_n[f] = {"w_star": w, "ffm": m0, "idw": m1, "blend": mb,
                        "improved_vs_ffm": f"{bF}/{nt}",
                        "improved_vs_idw": f"{bI}/{nt}",
                        "delta_pct_vs_ffm": dF, "delta_pct_vs_idw": dI}
            # Nick's bar: consistent OBSERVED-channel improvement
            if f in OBSERVED:
                ch_pass = (mb["rel_l2"] < m0["rel_l2"]
                           and mb["rel_l2"] < m1["rel_l2"]
                           and bF >= int(0.9 * nt) and bI >= int(0.9 * nt))
                res_n[f]["pass"] = bool(ch_pass)
                overall_pass = overall_pass and ch_pass
        out["per_density"][str(n)] = res_n

    print()
    print("=" * 78)
    verdict = "PASS" if overall_pass else "FAIL"
    print(f"VERDICT vs Nick's bar (consistent observed-channel improvement "
          f"across untouched snapshots and densities): {verdict}")
    print("=" * 78)
    out["verdict"] = verdict

    if args.out:
        with open(args.out, "w") as fp:
            json.dump(out, fp, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
