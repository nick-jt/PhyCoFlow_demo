"""Partition the reconstruction solver's runtime into named sections.

Profiles a full 125^3 JHTDB reconstruction through the deployed path
(chunked Euler integration of the rectified flow at NFE=2) and attributes
every millisecond to a named stage of the solver. Timing uses CUDA events
stamped at STAGE BOUNDARIES inside the backbone forward, so the intervals
are contiguous by construction: each stage's interval runs from its entry
to the next stage's entry, which means inter-stage glue (concats, norms,
broadcasts) is charged to the stage that produced it and nothing is left
over for an "other" bucket. Time outside the velocity forward but inside
the sampler (prior draw, Euler state update, chunk assembly) is measured
directly and named for what it is.

Note: run with torch.compile OFF. Compiled mode fuses stages, so the
partition below is the algorithmic breakdown in eager mode; total eager
time will exceed the compiled production time accordingly.
"""

import json
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ensemble_eval import load_run
from helpers import build_sparse_condition

RUN = ("../Save_TrainedModel/JHU/pointcloud_ffm/"
       "iclr_jhu_xcube_aug_DemoN15_20260818_083446")
N_STEPS = 2
CHUNK = 131072

STAGES = [
    ("query_features", "Query featurization (pos-enc + point encoder)"),
    ("sensor_tokens", "Sensor tokenization"),
    ("latent_attn", "Global latent attention"),
    ("latent_readout", "Per-query latent readout"),
    ("sensor_refine", "Sensor back-attention + importance"),
    ("gather", "Top-K neighbor search + RBF gather"),
    ("head", "Fusion + output head"),
]


class SectionTimer:
    """Boundary events inside one forward; intervals are contiguous."""

    def __init__(self):
        self.events = []           # (stage_key, torch.cuda.Event) in call order
        self.totals = defaultdict(float)

    def stamp(self, key):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.append((key, ev))

    def flush_forward(self):
        """Convert one forward's boundary stamps into per-stage times."""
        torch.cuda.synchronize()
        for (k, e0), (_, e1) in zip(self.events, self.events[1:]):
            self.totals[k] += e0.elapsed_time(e1) / 1e3
        self.events.clear()


TIMER = SectionTimer()


def instrument(net):
    orig = net.forward

    def timed_forward(t, x_t, coords, obs_coords, obs_values, obs_mask,
                      obs_field_ids):
        bsz, n_pts, _ = x_t.shape
        TIMER.stamp("query_features")
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        coord_feat = net.pos_enc(coords) if net.pos_enc is not None else coords
        point_feat = net.point_encoder(
            torch.cat([coord_feat, x_t, t_feat], dim=-1))

        TIMER.stamp("sensor_tokens")
        sensor_tokens = net._build_sensor_tokens(
            obs_coords=obs_coords, obs_values=obs_values,
            obs_field_ids=obs_field_ids, obs_mask=obs_mask)

        TIMER.stamp("latent_attn")
        latents = net._encode_latents(sensor_tokens=sensor_tokens,
                                      obs_mask=obs_mask)
        global_feat = net._extract_global_summary(latents)

        TIMER.stamp("latent_readout")
        if net.use_query_latent_readout:
            query_global = net._readout_query_global_chunked(
                point_feat, coords, latents)
            global_for_head = (global_feat.unsqueeze(1)
                               + net.query_readout_scale * query_global)
        else:
            global_for_head = global_feat.unsqueeze(1).expand(bsz, n_pts, -1)

        TIMER.stamp("sensor_refine")
        refined = net.sensor_back_attn(q=sensor_tokens, kv=latents,
                                       kv_padding_mask=None)
        refined = refined * obs_mask.unsqueeze(-1)
        refined_sensor_feat = net.sensor_out_proj(refined)
        refined_sensor_feat = refined_sensor_feat * obs_mask.unsqueeze(-1)
        sensor_importance_bias = net._compute_sensor_importance_bias(
            refined_sensor_feat=refined_sensor_feat, obs_mask=obs_mask)

        TIMER.stamp("gather")
        local_cond = net.aggregate_sparse_obs(
            query_coords=coords, query_feat=point_feat,
            obs_coords=obs_coords, refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask, sensor_importance_bias=sensor_importance_bias)

        TIMER.stamp("head")
        coarse_pred = net.coarse_scale * net._predict_global_coarse(
            point_feat, global_feat)
        head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
        out = coarse_pred + net.head(net.head_in_norm(head_in))
        TIMER.stamp("_end")
        TIMER.flush_forward()
        return out

    net.forward = timed_forward
    return orig


def main():
    device = torch.device("cuda:0")
    model, dataset, cfg = load_run(RUN, "best.pt", str(device))
    model.eval()
    net = model.model

    # The instrumented path reproduces the topk_rbf_glres branch; guard
    # against silently profiling a different configuration.
    assert net.gather_mode == "topk_rbf_glres", net.gather_mode
    instrument(net)

    item = dataset[0]
    coords_full = item["coords"][None].to(device)
    fields_full = item["fields"][None].to(device)
    torch.manual_seed(0)
    oc, ov, om, oi, ofid = build_sparse_condition(
        coords_full=coords_full, fields_full=fields_full,
        cond_fields=[0, 2], n_obs_min=[19531], n_obs_max=[19531])

    prior_s = 0.0
    orig_src = model.sample_source

    def timed_source(coords):
        nonlocal prior_s
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig_src(coords)
        torch.cuda.synchronize(); prior_s += time.perf_counter() - t0
        return out

    model.sample_source = timed_source

    def run_once():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16,
                                             enabled=True):
            for c0 in range(0, coords_full.shape[1], CHUNK):
                cc = coords_full[:, c0:c0 + CHUNK]
                model.sample(coords=cc, obs_coords=oc, obs_values=ov,
                             obs_mask=om, obs_field_ids=ofid,
                             n_steps=N_STEPS, obs_consistency_mode="none")

    run_once()                                   # warmup (kernels, autotune)
    TIMER.totals.clear(); prior_s = 0.0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    run_once()
    torch.cuda.synchronize(); total = time.perf_counter() - t0

    sec = {k: TIMER.totals[k] for k, _ in STAGES}
    fwd = sum(sec.values())
    sec["prior"] = prior_s
    # Everything the sampler does outside the velocity forwards and the
    # prior draw: Euler state updates, per-chunk tensor setup and assembly.
    sec["ode_update"] = max(total - fwd - prior_s, 0.0)

    labels = {**dict(STAGES),
              "prior": "RFF prior sampling",
              "ode_update": "Euler update + chunk assembly"}
    order = sorted(sec, key=lambda k: -sec[k])
    print(f"total {total:.2f} s for 125^3 at NFE={N_STEPS} (eager)")
    for k in order:
        print(f"  {labels[k]:45s} {sec[k]:7.3f} s  {100*sec[k]/total:5.1f}%")

    json.dump({"total_s": total, "n_steps": N_STEPS, "chunk": CHUNK,
               "sections_s": sec},
              open("profile_solver_sections.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ys = range(len(order))
    vals = [sec[k] for k in order]
    ax.barh(list(ys), vals, color="#31688e")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([labels[k] for k in order], fontsize=10)
    ax.invert_yaxis()
    for y, v in zip(ys, vals):
        ax.text(v + 0.01 * max(vals), y, f"{v:.2f} s ({100*v/total:.0f}%)",
                va="center", fontsize=9)
    ax.set_xlabel(f"seconds per full 125$^3$ reconstruction "
                  f"(NFE={N_STEPS}, eager, H100)")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("profile_solver_sections.png", dpi=180)
    print("wrote profile_solver_sections.png / .json")





def profile_training_step():
    """Same section partition for one training step at the training protocol.

    The inference caching trick does not apply here: training draws FRESH
    sensor sets and query subsets every step (the amortization that makes the
    model robust), and each step runs exactly one velocity forward, so there
    is no repeated identical search to reuse. This measures where a training
    step actually spends its time instead.
    """
    device = torch.device("cuda:0")
    model, dataset, cfg = load_run(RUN, "best.pt", str(device))
    model.train()
    net = model.model
    instrument(net)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B, Q, M = 20, 39062, 19531
    items = [dataset[i % len(dataset)] for i in range(B)]
    coords = torch.stack([it["coords"] for it in items]).to(device)
    fields = torch.stack([it["fields"] for it in items]).to(device)

    import numpy as np
    from train_pointcloud_ffm import sample_query_subset

    reps, warm = 5, 2
    stage_tot = defaultdict(float)
    misc = {"obs_build": 0.0, "query_sample": 0.0, "prior": 0.0,
            "backward_opt": 0.0}
    total = 0.0
    for r in range(warm + reps):
        torch.cuda.synchronize(); ta = time.perf_counter()
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields, cond_fields=[0, 2],
            n_obs_min=[1953], n_obs_max=[M])
        torch.cuda.synchronize(); tb = time.perf_counter()
        cq, fq, _ = sample_query_subset(coords=coords, fields=fields,
                                        n_query=Q, mode="uniform")
        torch.cuda.synchronize(); tc = time.perf_counter()
        TIMER.totals.clear()
        t0p = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            x0 = model.sample_source(cq)
            torch.cuda.synchronize(); t1p = time.perf_counter()
            t = torch.rand(B, device=device, dtype=fq.dtype)
            x_t = model.simulate(t, x0, fq)
            target = model.target_vector_field(x0, fq)
            pred = net.forward(t, x_t, cq, oc, ov, om, ofid)
            loss = torch.nn.functional.mse_loss(pred, target)
        torch.cuda.synchronize(); td = time.perf_counter()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize(); te = time.perf_counter()
        if r < warm:
            continue
        for k, v in TIMER.totals.items():
            stage_tot[k] += v
        misc["obs_build"] += tb - ta
        misc["query_sample"] += tc - tb
        misc["prior"] += t1p - t0p
        misc["backward_opt"] += te - td
        total += te - ta

    sec = {k: stage_tot[k] / reps for k, _ in STAGES}
    for k in misc:
        misc[k] /= reps
    total /= reps
    fwd = sum(sec.values())
    # interpolation/target arithmetic inside the autocast block but outside
    # the instrumented forward and the prior draw
    sec["interp_target"] = max((td - t0p) / 1 - 0, 0)  # placeholder, computed below
    labels = {**dict(STAGES), "interp_target": "Path interpolation + target",
              "obs_build": "Sensor draw + masking", "query_sample": "Query subsampling",
              "prior": "RFF prior sampling", "backward_opt": "Backward + optimizer"}
    # recompute interp_target properly: forward-block wall minus sections minus prior
    sec["interp_target"] = max(misc["prior"] * 0 + (total - misc["obs_build"]
                               - misc["query_sample"] - misc["backward_opt"]
                               - misc["prior"] - fwd), 0.0)
    sec.update({k: misc[k] for k in ("obs_build", "query_sample", "prior",
                                     "backward_opt")})
    order = sorted(sec, key=lambda k: -sec[k])
    print(f"\nTRAINING step total {total:.3f} s (B={B}, Q={Q}, M<= {M}/field)")
    for k in order:
        print(f"  {labels[k]:42s} {sec[k]:7.3f} s  {100*sec[k]/total:5.1f}%")
    json.dump({"total_s": total, "sections_s": sec},
              open("profile_training_sections.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(9, 5.0))
    ys = range(len(order)); vals = [sec[k] for k in order]
    ax.barh(list(ys), vals, color="#35b779")
    ax.set_yticks(list(ys)); ax.set_yticklabels([labels[k] for k in order], fontsize=10)
    ax.invert_yaxis()
    for y, v in zip(ys, vals):
        ax.text(v + 0.01 * max(vals), y, f"{v:.3f} s ({100*v/total:.0f}%)",
                va="center", fontsize=9)
    ax.set_xlabel(f"seconds per training step (B={B}, {Q:,} queries, bf16, eager, H100)")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("profile_training_sections.png", dpi=180)
    print("wrote profile_training_sections.png / .json")





BW = SectionTimer()
BW_ACTIVE = [False]


def _bw_flush(order_labels):
    """Backward intervals, labeled by the boundary each span reaches."""
    torch.cuda.synchronize()
    spans = []
    for (k0, e0), (k1, e1) in zip(BW.events, BW.events[1:]):
        spans.append((k1, e0.elapsed_time(e1) / 1e3))
    BW.events.clear()
    return spans


def instrument_backward(net):
    """Forward identical to instrument(), plus grad hooks on boundaries."""
    def timed_forward(t, x_t, coords, obs_coords, obs_values, obs_mask,
                      obs_field_ids):
        bsz, n_pts, _ = x_t.shape
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        coord_feat = net.pos_enc(coords) if net.pos_enc is not None else coords
        point_feat = net.point_encoder(
            torch.cat([coord_feat, x_t, t_feat], dim=-1))
        sensor_tokens = net._build_sensor_tokens(
            obs_coords=obs_coords, obs_values=obs_values,
            obs_field_ids=obs_field_ids, obs_mask=obs_mask)
        latents = net._encode_latents(sensor_tokens=sensor_tokens,
                                      obs_mask=obs_mask)
        global_feat = net._extract_global_summary(latents)
        if net.use_query_latent_readout:
            query_global = net._readout_query_global_chunked(
                point_feat, coords, latents)
            global_for_head = (global_feat.unsqueeze(1)
                               + net.query_readout_scale * query_global)
        else:
            query_global = None
            global_for_head = global_feat.unsqueeze(1).expand(bsz, n_pts, -1)
        refined = net.sensor_back_attn(q=sensor_tokens, kv=latents,
                                       kv_padding_mask=None)
        refined = refined * obs_mask.unsqueeze(-1)
        refined_sensor_feat = net.sensor_out_proj(refined)
        refined_sensor_feat = refined_sensor_feat * obs_mask.unsqueeze(-1)
        sensor_importance_bias = net._compute_sensor_importance_bias(
            refined_sensor_feat=refined_sensor_feat, obs_mask=obs_mask)
        local_cond = net.aggregate_sparse_obs(
            query_coords=coords, query_feat=point_feat,
            obs_coords=obs_coords, refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask, sensor_importance_bias=sensor_importance_bias)
        coarse_pred = net.coarse_scale * net._predict_global_coarse(
            point_feat, global_feat)
        head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
        out = coarse_pred + net.head(net.head_in_norm(head_in))

        if BW_ACTIVE[0]:
            bounds = [("g_local_cond", local_cond),
                      ("g_refined_sensor", refined_sensor_feat),
                      ("g_latents", latents),
                      ("g_sensor_tokens", sensor_tokens),
                      ("g_point_feat", point_feat)]
            if query_global is not None:
                bounds.insert(1, ("g_query_global", query_global))
            for name, tensor in bounds:
                tensor.register_hook(
                    lambda g, n=name: (BW.stamp(n), None)[1])
        return out

    net.forward = timed_forward


def profile_backward():
    device = torch.device("cuda:0")
    model, dataset, cfg = load_run(RUN, "best.pt", str(device))
    model.train()
    net = model.model
    instrument_backward(net)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    from train_pointcloud_ffm import sample_query_subset

    B, Q, M = 20, 39062, 19531
    items = [dataset[i % len(dataset)] for i in range(B)]
    coords = torch.stack([it["coords"] for it in items]).to(device)
    fields = torch.stack([it["fields"] for it in items]).to(device)

    labels = {
        "g_local_cond": "Loss + fusion head + coarse",
        "g_query_global": "Latent readout (bwd)",
        "g_refined_sensor": "Top-K gather + importance (bwd)",
        "g_latents": "Sensor back-attention (bwd)",
        "g_sensor_tokens": "Global latent attention (bwd)",
        "g_point_feat": "Sensor tokenization (bwd)",
        "bw_end": "Query featurization + grad accum",
        "optimizer": "Optimizer step",
    }
    reps, warm = 5, 2
    agg = defaultdict(float)
    total = 0.0
    for r in range(warm + reps):
        oc, ov, om, oi, ofid = build_sparse_condition(
            coords_full=coords, fields_full=fields, cond_fields=[0, 2],
            n_obs_min=[1953], n_obs_max=[M])
        cq, fq, _ = sample_query_subset(coords=coords, fields=fields,
                                        n_query=Q, mode="uniform")
        BW_ACTIVE[0] = r >= warm
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            x0 = model.sample_source(cq)
            t = torch.rand(B, device=device, dtype=fq.dtype)
            x_t = model.simulate(t, x0, fq)
            target = model.target_vector_field(x0, fq)
            pred = net.forward(t, x_t, cq, oc, ov, om, ofid)
            loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        if r >= warm:
            BW.events.clear()
            BW.stamp("bw_start")
        torch.cuda.synchronize(); tb0 = time.perf_counter()
        loss.backward()
        if r >= warm:
            BW.stamp("bw_end")
        torch.cuda.synchronize(); tb1 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize(); tb2 = time.perf_counter()
        if r < warm:
            continue
        for k, dt in _bw_flush(None):
            agg[k] += dt
        agg["optimizer"] += tb2 - tb1
        total += tb2 - tb0

    for k in agg:
        agg[k] /= reps
    total /= reps
    order = sorted(agg, key=lambda k: -agg[k])
    print(f"\nBACKWARD partition, mean of {reps} steps "
          f"(B={B}, Q={Q}): total {total:.3f} s")
    for k in order:
        print(f"  {labels.get(k, k):40s} {agg[k]:7.3f} s  "
              f"{100*agg[k]/total:5.1f}%")
    json.dump({"total_s": total, "sections_s": dict(agg)},
              open("profile_backward_sections.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ys = range(len(order)); vals = [agg[k] for k in order]
    ax.barh(list(ys), vals, color="#d1495b")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([labels.get(k, k) for k in order], fontsize=10)
    ax.invert_yaxis()
    for y, v in zip(ys, vals):
        ax.text(v + 0.01 * max(vals), y, f"{v:.3f} s ({100*v/total:.0f}%)",
                va="center", fontsize=9)
    ax.set_xlabel(f"seconds, backward + optimizer per training step "
                  f"(B={B}, {Q:,} queries, bf16, eager, H100)")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("profile_backward_sections.png", dpi=180)
    print("wrote profile_backward_sections.png / .json")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "infer"
    if mode == "train":
        profile_training_step()
    elif mode == "backward":
        profile_backward()
    else:
        main()
