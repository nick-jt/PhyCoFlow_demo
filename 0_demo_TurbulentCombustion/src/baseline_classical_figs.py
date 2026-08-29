"""Reconstruction figures for the classical anchor baselines.

Two products per snapshot:

1. ``<method>_snapNN.png`` -- produced by ``ensemble_eval.save_ensemble_figure``,
   the *same* function that draws pointcloud_ffm's diagnostic panels, so the
   classical panels can sit directly beside
   ``Evaluation/sensor_sweep_n19531_snapNN.png`` with identical slice, colour
   scaling and layout.  For a deterministic estimator the "ens. mean" and
   "sample 0" rows are the same field and the spread row is exactly zero --
   that degeneracy is the point, and it should be visible.

2. ``<method>_snapNN_truth_pred_err.png`` -- truth / prediction / |error| per
   channel on the same z-midplane.

3. ``uy_limitation_snapNN.png`` -- the pairing the aggregate table cannot make:
   truth Uy beside our model's Uy and beside the constant (train-mean)
   predictor's Uy, which is a flat field.  The constant predictor scores
   relative L2 exactly 1.000 on Uy and our model scores 1.050, i.e. a
   parameter-free flat field beats a trained generative model on the
   unobserved channel.  ``--with-model`` samples our model for these snapshots
   so the two Uy fields are drawn from the same data in one figure.

Snapshots 11 and 33 are used because our model already has figures for exactly
those two at n_obs=19531.

Must run on a COMPUTE node: torch.randperm on CUDA is not portable between
H100 PCIe (login) and H100 SXM (compute), so the sensor layout differs.
Canonical check: snap=29 -> idx_sum=37987162596.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("JHU_SPLIT_MODE", "block")
os.environ.setdefault("JHU_SPLIT_GAP", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import baseline_classical_jhu as B
from ensemble_eval import save_ensemble_figure
from helpers import TurbulentCombustionH5Dataset

RUN = ("../Save_TrainedModel/JHU/pointcloud_ffm/"
       "iclr_jhu_xcube_spec02_DemoN29_20260822_140100")


def midplane(coords_raw):
    c = np.asarray(coords_raw)
    lv = np.unique(c[:, 2])
    m = c[:, 2] == lv[len(lv) // 2]
    xy = c[m][:, :2]
    return m, mtri.Triangulation(xy[:, 0], xy[:, 1])


def fig_truth_pred_err(pred, true, coords_raw, names, out, tag):
    m, tri = midplane(coords_raw)
    nC = len(names)
    fig, axs = plt.subplots(3, nC, figsize=(3.3 * nC, 9.4))
    axs = np.atleast_2d(axs)
    for j in range(nC):
        tj = true[m, j]
        lo, hi = (float(v) for v in np.percentile(tj, [1, 99]))
        rows = [("truth", tj, "coolwarm", (lo, hi)),
                ("prediction", pred[m, j], "coolwarm", (lo, hi)),
                ("|error|", np.abs(pred[m, j] - tj), "magma", (None, None))]
        for r, (lbl, a, cmap, vlim) in enumerate(rows):
            kw = {} if vlim[0] is None else {"vmin": vlim[0], "vmax": vlim[1]}
            im = axs[r, j].tripcolor(tri, a, cmap=cmap, shading="gouraud", **kw)
            fig.colorbar(im, ax=axs[r, j], fraction=0.046)
            axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
            if r == 0:
                axs[r, j].set_title(str(names[j]), fontsize=11)
            if j == 0:
                axs[r, j].set_ylabel(lbl, fontsize=10)
    fig.suptitle(f"z-midplane slice  {tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)


def fig_uy(true, panels, coords_raw, out, tag):
    """panels: list of (label, field[N], rel_l2)."""
    m, tri = midplane(coords_raw)
    tj = true[m, 1]
    lo, hi = (float(v) for v in np.percentile(tj, [1, 99]))
    cols = [("truth Uy", tj, None)] + [(f"{l}\nrelL2 {r:.3f}", p[m, 1], r)
                                       for l, p, r in panels]
    fig, axs = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 6.6))
    axs = np.atleast_2d(axs)
    for j, (lbl, a, _) in enumerate(cols):
        im = axs[0, j].tripcolor(tri, a, cmap="coolwarm", shading="gouraud",
                                 vmin=lo, vmax=hi)
        fig.colorbar(im, ax=axs[0, j], fraction=0.046)
        axs[0, j].set_title(lbl, fontsize=10)
        e = np.abs(a - tj)
        im = axs[1, j].tripcolor(tri, e, cmap="magma", shading="gouraud")
        fig.colorbar(im, ax=axs[1, j], fraction=0.046)
        axs[1, j].set_title(f"|error|  mean {e.mean():.3f}", fontsize=9)
        for r in (0, 1):
            axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
    axs[0, 0].set_ylabel("field", fontsize=10)
    axs[1, 0].set_ylabel("abs. error", fontsize=10)
    fig.suptitle("Unobserved channel Uy -- a flat train-mean field scores "
                 f"relative L2 exactly 1.000   ({tag})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)


def rel_l2(p, y):
    return float(np.linalg.norm(p - y) / (np.linalg.norm(y) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=int, nargs="+", default=[11, 33])
    ap.add_argument("--n-obs", type=int, default=19531)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pod-rank", type=int, default=80)
    ap.add_argument("--with-model", action="store_true",
                    help="Also sample pointcloud_ffm so the Uy figure carries "
                         "our model's field next to the constant predictor's.")
    ap.add_argument("--model-K", type=int, default=4)
    ap.add_argument("--model-nfe", type=int, default=4)
    ap.add_argument("--out-dir",
                    default="../Save_TrainedModel/JHU/baseline_classical/Figures")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    stats = str(Path(RUN) / "dataset_stats.pt")
    ds = TurbulentCombustionH5Dataset(B.DATA, split="val", train_ratio=0.75,
                                      field_names=B.FIELD_NAMES, seed=42,
                                      time_stride=1, stats_path=stats)
    ds_tr = TurbulentCombustionH5Dataset(B.DATA, split="train", train_ratio=0.75,
                                         field_names=B.FIELD_NAMES, seed=42,
                                         time_stride=1, stats_path=stats)
    N, C = ds.num_points, ds.num_fields
    names = list(B.FIELD_NAMES)
    craw = ds.coords_raw.numpy()

    cb = ds.coords_raw.numpy().astype(np.float64)
    side = int(round(N ** (1.0 / 3.0)))
    lo = cb.min(0); dx = (cb.max(0) - lo) / (side - 1)
    coords_box = np.ascontiguousarray((cb - lo) / (side * dx))

    print("[fig] fitting gappy POD basis on training cubes ...", flush=True)
    Xtr = B.load_split(ds_tr, ds_tr.indices).reshape(len(ds_tr), -1)
    mu = Xtr.mean(axis=0); Xtr -= mu
    pod = B.GappyPOD(Xtr, mu, rank=a.pod_rank)
    print(f"[fig] POD rank {pod.r}", flush=True)

    model = None
    if a.with_model:
        from ensemble_eval import load_run, sample_ensemble
        model, _ds_m, cfg = load_run(RUN, "best.pt", a.device)

    for snap in a.snapshots:
        Y = B.load_split(ds, ds.indices[snap:snap + 1])[0]          # [N, C]
        sens = B.draw_sensors(ds.coords, torch.from_numpy(Y), snap=snap,
                              n_obs=a.n_obs, seed=a.seed, device=a.device)
        fp = int(sum(int(v[0].sum()) for v in sens.values()))
        print(f"[seedcheck] snap={snap} sensors="
              f"{sum(len(v[0]) for v in sens.values())} idx_sum={fp}", flush=True)

        preds = {
            "constant_train_mean": np.zeros((N, C), np.float32),
            "kdtree": B.kd_predict(coords_box, sens, C, "nn", 8, None),
            "idw_k8": B.kd_predict(coords_box, sens, C, "idw", 8, None),
        }
        cols, vals = B.obs_columns(sens, C)
        preds["gappy_pod_r%d" % pod.r] = pod.reconstruct(cols, vals).reshape(N, C)

        if model is not None:
            from helpers import build_sparse_condition
            coords_t = ds.coords.unsqueeze(0).to(a.device)
            fields_t = torch.from_numpy(Y).unsqueeze(0).to(a.device)
            torch.manual_seed(a.seed * 777 + int(snap))
            oc, ov, om, oi, ofid = build_sparse_condition(
                coords_full=coords_t, fields_full=fields_t,
                cond_fields=[0, 2], n_obs_min=[a.n_obs] * 2,
                n_obs_max=[a.n_obs] * 2)
            obs = {"coords": oc, "values": ov, "mask": om, "indices": oi,
                   "field_ids": ofid}
            ens = sample_ensemble(model, coords_t, obs, K=a.model_K,
                                  n_steps=a.model_nfe, clamp_hard=False,
                                  seed=snap)
            preds["OURS_pointcloud_ffm_ensmean"] = ens.numpy().mean(0)
            del coords_t, fields_t, oc, ov, om, oi, ofid
            torch.cuda.empty_cache()

        for name, p in preds.items():
            r = rel_l2(p, Y)
            tag = (f"{name}  snap {snap}  n_obs={a.n_obs}/channel  relL2={r:.4f}")
            save_ensemble_figure(np.repeat(p[None], 2, axis=0), Y, craw, names,
                                 out / f"{name}_snap{snap}.png", tag=tag)
            fig_truth_pred_err(p, Y, craw, names,
                               out / f"{name}_snap{snap}_truth_pred_err.png", tag)
            print(f"[fig] {name} snap {snap}: relL2 {r:.4f} "
                  f"(Uy {rel_l2(p[:, 1], Y[:, 1]):.4f})", flush=True)

        panels = []
        if "OURS_pointcloud_ffm_ensmean" in preds:
            q = preds["OURS_pointcloud_ffm_ensmean"]
            panels.append(("ours (pointcloud_ffm)", q, rel_l2(q[:, 1], Y[:, 1])))
        q = preds["constant_train_mean"]
        panels.append(("constant = train mean", q, rel_l2(q[:, 1], Y[:, 1])))
        q = preds["kdtree"]
        panels.append(("KD-tree NN", q, rel_l2(q[:, 1], Y[:, 1])))
        fig_uy(Y, panels, craw, out / f"uy_limitation_snap{snap}.png",
               tag=f"snap {snap}, 1% sensors on Ux/Uz only")
        print(f"[fig] wrote uy_limitation_snap{snap}.png", flush=True)

    print(f"[fig] all figures in {out}", flush=True)


if __name__ == "__main__":
    main()
