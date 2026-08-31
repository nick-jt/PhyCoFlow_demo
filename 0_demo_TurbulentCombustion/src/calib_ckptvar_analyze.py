"""Two-way variance decomposition over an archived checkpoint window.

M_cs = mu + a_c + b_s + e_cs   (c = checkpoint, s = snapshot, one obs/cell)

naive sigma  = std of the checkpoint (row) means
corrected    = sqrt(max(0, Var(row means) - Var(e)/S)), with Var(e) the mean
               square of the two-way residual (unreplicated design: it carries
               any checkpoint x snapshot interaction, so it is an upper bound
               on pure measurement noise and the correction is conservative).
Also reports the trajectory across the window, so a monotone drift is not
reported as noise.
"""
import glob, json, os, re, sys
import numpy as np

E = sys.argv[1]
files = sorted(glob.glob(os.path.join(E, "ckptvar_ep*.json")))
if not files:
    raise SystemExit("no ckptvar_ep*.json found")

def metrics(d):
    """-> dict metric -> {snapshot: value}, for two aggregations."""
    out = {}
    for s in d["snapshots"]:
        pf = s["per_field"]
        obs = ("Ux", "Uz")
        vals = {
            "relL2_all": s["aggregate"]["rel_l2_mean"],
            "crps_all": s["aggregate"]["crps"],
            "spread_err_all": s["aggregate"]["spread_error_ratio"],
            "cov90_all": s["aggregate"]["coverage_90"],
            "relL2_obs": np.mean([pf[c]["rel_l2_mean"] for c in obs]),
            "crps_obs": np.mean([pf[c]["crps"] for c in obs]),
            "spread_obs": np.mean([pf[c]["spread"] for c in obs]),
            "rmse_obs": np.mean([pf[c]["rmse"] for c in obs]),
            "spread_err_obs": np.mean([pf[c]["spread"] for c in obs]) /
                              np.mean([pf[c]["rmse"] for c in obs]),
            "cov90_obs": np.mean([pf[c]["coverage_90"] for c in obs]),
        }
        for k, v in vals.items():
            out.setdefault(k, {})[int(s["snapshot"])] = float(v)
    return out

epochs, tabs = [], []
for f in files:
    ep = int(re.search(r"ckptvar_ep(\d+)", f).group(1))
    d = json.load(open(f))
    epochs.append(ep); tabs.append(metrics(d))
order = np.argsort(epochs)
epochs = [epochs[i] for i in order]; tabs = [tabs[i] for i in order]
snaps = sorted(set.intersection(*[set(t["relL2_all"]) for t in tabs]))
print(f"checkpoints={epochs}\nsnapshots({len(snaps)})={snaps}\n")

res = {"epochs": epochs, "snapshots": snaps, "metrics": {}}
GAPS = {"relL2_all": 0.019, "crps_all": 0.014}
for m in tabs[0]:
    M = np.array([[t[m][s] for s in snaps] for t in tabs])   # [C, S]
    C, S = M.shape
    row = M.mean(1); col = M.mean(0); mu = M.mean()
    resid = M - row[:, None] - col[None, :] + mu
    var_e = (resid ** 2).sum() / max((C - 1) * (S - 1), 1)
    var_row = row.var(ddof=1)
    corr = max(0.0, var_row - var_e / S)
    naive, sig = np.sqrt(var_row), np.sqrt(corr)
    sl = np.polyfit(epochs, row, 1)[0] * (epochs[-1] - epochs[0])
    rr = np.corrcoef(epochs, row)[0, 1]
    res["metrics"][m] = {"row_means": row.tolist(), "naive_sigma": float(naive),
                         "corrected_sigma": float(sig), "resid_sd": float(np.sqrt(var_e)),
                         "range": float(row.max() - row.min()),
                         "trend_over_window": float(sl), "trend_r": float(rr)}
    line = (f"{m:16s} mean={mu:.5f} naive_sigma={naive:.5f} corrected_sigma={sig:.5f} "
            f"range={row.max()-row.min():.5f} trend={sl:+.5f} r={rr:+.2f}")
    if m in GAPS and sig > 0:
        line += f"  | paper gap {GAPS[m]:.3f} = {GAPS[m]/sig:.2f} sigma"
    print(line)
    print("   per-checkpoint: " + " ".join(f"{e}:{v:.4f}" for e, v in zip(epochs, row)))
json.dump(res, open(os.path.join(E, "calib_ckptvar_summary.json"), "w"), indent=2)
print("\nwrote calib_ckptvar_summary.json")
