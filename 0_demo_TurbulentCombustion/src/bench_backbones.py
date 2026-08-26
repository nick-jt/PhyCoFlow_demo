"""Apples-to-apples: our ENH vs the upstream CQ query side, with and without
our valid-key flash attention, at matched neighbour budget.

Identical shapes, identical sensor protocol, identical precision for every
row. Training rows time a full forward+backward+optimizer step at the
production batch; inference rows time a full 125^3 field reconstruction with
each stack's own caching enabled (our _ode_cache for ENH, the vendored
cached_streamed + geometry cache for CQ).
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

GRID = 125
NPTS = GRID ** 3
B, Q, M = 20, 39062, 39062


def batch(dev, n_fields=4):
    g = torch.Generator().manual_seed(0)
    counts = torch.randint(int(M * 0.1), M, (B,), generator=g)
    om = torch.zeros(B, M)
    for i, c in enumerate(counts):
        om[i, :c] = 1.0
    return dict(
        coords=torch.rand(B, Q, 3, generator=g).to(dev),
        x_t=torch.randn(B, Q, n_fields, generator=g).to(dev),
        t=torch.rand(B, generator=g).to(dev),
        obs_coords=torch.rand(B, M, 3, generator=g).to(dev),
        obs_values=torch.randn(B, M, 1, generator=g).to(dev),
        obs_mask=om.to(dev),
        obs_field_ids=torch.randint(0, 2, (B, M), generator=g).to(dev),
    )


def time_train(bb, bt, n=6):
    bb.train()
    opt = torch.optim.AdamW(bb.parameters(), lr=1e-9)
    for i in range(n):
        if i == 2:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = bb(bt["t"], bt["x_t"], bt["coords"], bt["obs_coords"],
                   bt["obs_values"], bt["obs_mask"], bt["obs_field_ids"])
            loss = v.float().pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (n - 2), torch.cuda.max_memory_allocated() / 1024 ** 2


@torch.no_grad()
def time_infer(model, dev, n_steps, kind, n_obs=19531):
    # eval mode is REQUIRED for our ODE cache: the reuse guard is
    # `isinstance(_ode_cache, dict) and not self.training`.
    model.eval()
    g = torch.Generator().manual_seed(1)
    lin = torch.linspace(0, 1, GRID)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    coords = torch.stack([X, Y, Z], -1).reshape(1, -1, 3).to(dev)
    oc = torch.rand(1, n_obs, 3, generator=g).to(dev)
    ov = torch.randn(1, n_obs, 1, generator=g).to(dev)
    om = torch.ones(1, n_obs).to(dev)
    ofid = torch.randint(0, 2, (1, n_obs), generator=g).to(dev)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if kind == "enh":
        # model.sample() is what installs our _ode_cache; ensemble_eval's own
        # stepping loop calls the backbone directly and therefore bypasses it,
        # so timing that path would understate our stack.
        model.sample(coords=coords, obs_coords=oc, obs_values=ov, obs_mask=om,
                     obs_field_ids=ofid, n_steps=n_steps,
                     obs_consistency_mode="none")
    else:
        model.sample(coords=coords, obs_coords=oc, obs_values=ov, obs_mask=om,
                     obs_field_ids=ofid, n_steps=n_steps,
                     obs_consistency_mode="none",
                     reconstruction_execution_mode="cached_streamed",
                     reconstruction_query_chunk_size=262144)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 1024 ** 2


def main():
    dev = torch.device("cuda:0")
    import glob
    from evaluate_ffm import _build_model as build_enh
    from model_cq import build_cq_model, load_cq_config
    import cq_flash_patch

    rows = []
    bt = batch(dev)
    ds = SimpleNamespace(num_fields=4, n_obs_field_types=None, num_points=NPTS)

    # ---------- ours (ENH, frozen config) ----------
    n29 = sorted(glob.glob("../Save_TrainedModel/JHU/pointcloud_ffm/"
                           "iclr_jhu_xcube_spec02_DemoN29_*"))[-1]
    cfg = json.load(open(Path(n29) / "args.json"))
    enh = build_enh(cfg, ds).to(dev)
    n_par = sum(p.numel() for p in enh.model.parameters())
    dt, mem = time_train(enh.model, bt)
    inf = {ns: time_infer(enh, dev, ns, "enh") for ns in (4, 16)}
    rows.append(("ours ENH (K=16, cached)", n_par, dt, mem, inf))
    from ensemble_eval import sample_ensemble as _se
    import time as _t

    @torch.no_grad()
    def _uncached(ns):
        g = torch.Generator().manual_seed(1)
        lin = torch.linspace(0, 1, GRID)
        X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
        c = torch.stack([X, Y, Z], -1).reshape(1, -1, 3).to(dev)
        n_obs = 19531
        obs = {"coords": torch.rand(1, n_obs, 3, generator=g).to(dev),
               "values": torch.randn(1, n_obs, 1, generator=g).to(dev),
               "mask": torch.ones(1, n_obs).to(dev),
               "field_ids": torch.randint(0, 2, (1, n_obs), generator=g).to(dev),
               "indices": torch.zeros(1, n_obs, dtype=torch.long, device=dev)}
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = _t.perf_counter()
        _se(enh, c, obs, K=1, n_steps=ns, chunk=262144, clamp_hard=False, seed=0)
        torch.cuda.synchronize()
        return _t.perf_counter() - t0, torch.cuda.max_memory_allocated() / 1024 ** 2
    rows.append(("ours ENH (K=16, NO cache: eval path)", n_par, dt, mem,
                 {ns: _uncached(ns) for ns in (4, 16)}))
    del enh; torch.cuda.empty_cache()

    # ---------- upstream CQ, as released ----------
    args = SimpleNamespace(prior="rff", rff_features=256, rff_lengthscale=0.15,
                           sigma_min=1e-4, neighbor_backend="keops",
                           gather_query_chunk_size=2048, n_obs_field_types=None,
                           t_sampling="logit_normal", seed=42)
    for tag, topk, flash in (("upstream CQ (K=32, masked)", 32, False),
                             ("CQ + our flash (K=32)", 32, True),
                             ("CQ + our flash (K=16)", 16, True)):
        cfg_cq = load_cq_config(args)
        cfg_cq["gather_topk"] = topk
        torch.manual_seed(0)
        from phycoflow_pointcloud import build_pointcloud_model, PointCloudFFM
        built = build_pointcloud_model(cfg_cq, n_fields=4, device="cpu")
        from model_cq import PointCloudFFM_Ours
        m = PointCloudFFM_Ours(built.model, built.prior, sigma_min=1e-4).to(dev)
        n_par = sum(p.numel() for p in m.model.parameters())
        if flash:
            cq_flash_patch.enable()
        dt, mem = time_train(m.model, bt)
        inf = {ns: time_infer(m, dev, ns, "cq") for ns in (4, 16)}
        cq_flash_patch.disable()
        rows.append((tag, n_par, dt, mem, inf))
        del m; torch.cuda.empty_cache()

    print("\n" + "=" * 104)
    print(f"{'configuration':32s} {'params':>10s} {'train s/step':>13s} "
          f"{'train GB':>9s} {'infer NFE4':>11s} {'infer NFE16':>12s} {'inf GB':>8s}")
    print("-" * 104)
    for tag, npar, dt, mem, inf in rows:
        print(f"{tag:32s} {npar/1e6:9.2f}M {dt:12.3f}s {mem/1024:8.1f} "
              f"{inf[4][0]:10.2f}s {inf[16][0]:11.2f}s {inf[4][1]/1024:7.2f}")
    print("=" * 104)
    base = rows[0]
    for tag, npar, dt, mem, inf in rows[1:]:
        print(f"{tag:32s} train {base[2]/dt:5.2f}x   "
              f"infer NFE4 {base[4][4][0]/inf[4][0]:5.2f}x   "
              f"infer NFE16 {base[4][16][0]/inf[16][0]:5.2f}x  (>1 = faster than ours)")


if __name__ == "__main__":
    main()
