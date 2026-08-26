"""A/B the CQ backbone with and without our valid-key flash attention.

Measures a realistic training step (B=20, 39062 queries, variable sensor
counts drawn from the production protocol) and checks that the patched
attention is numerically equivalent to the masked version it replaces.
"""
import time
from types import SimpleNamespace

import torch

torch.backends.cuda.matmul.allow_tf32 = True


def make_batch(dev, B=20, Q=39062, M=39062, n_fields=4):
    g = torch.Generator(device="cpu").manual_seed(0)
    coords = torch.rand(B, Q, 3, generator=g).to(dev)
    x_t = torch.randn(B, Q, n_fields, generator=g).to(dev)
    t = torch.rand(B, generator=g).to(dev)
    obs_coords = torch.rand(B, M, 3, generator=g).to(dev)
    obs_values = torch.randn(B, M, 1, generator=g).to(dev)
    obs_field_ids = torch.randint(0, 2, (B, M), generator=g).to(dev)
    # production protocol: per-item sensor counts vary, so padding is present
    counts = torch.randint(int(M * 0.1), M, (B,), generator=g)
    obs_mask = torch.zeros(B, M)
    for i, c in enumerate(counts):
        obs_mask[i, :c] = 1.0
    return coords, x_t, t, obs_coords, obs_values, obs_mask.to(dev), obs_field_ids


def timed_step(model, batch, n=6, backward=True):
    coords, x_t, t, oc, ov, om, ofid = batch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-9)
    for i in range(n):
        if i == 2:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(t, x_t, coords, oc, ov, om, ofid)
            loss = v.float().pow(2).mean()
        if backward:
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / (n - 2)
    return dt, torch.cuda.max_memory_allocated() / 1024 ** 2


def main():
    dev = torch.device("cuda:0")
    from model_cq import build_cq_model
    import cq_flash_patch

    args = SimpleNamespace(prior="rff", rff_features=256, rff_lengthscale=0.15,
                           sigma_min=1e-4, neighbor_backend="keops",
                           gather_query_chunk_size=2048, n_obs_field_types=None,
                           t_sampling="logit_normal", seed=42)
    train_set = SimpleNamespace(num_fields=4, n_obs_field_types=None)

    torch.manual_seed(0)
    model, _ = build_cq_model(args, train_set, dev)
    batch = make_batch(dev)

    # --- equivalence check in eval mode (no dropout), fp32 for a clean compare
    model.eval()
    coords, x_t, t, oc, ov, om, ofid = batch
    with torch.no_grad():
        ref = model.model(t, x_t, coords, oc, ov, om, ofid).float()
        cq_flash_patch.enable()
        new = model.model(t, x_t, coords, oc, ov, om, ofid).float()
        cq_flash_patch.disable()
    d = (ref - new).abs().max().item()
    rel = d / ref.abs().max().item()
    print(f"[equivalence] max |masked - valid-key| = {d:.3e}  (rel {rel:.3e})")

    model.train()
    print("\n[training step, B=20, Q=39062, M=39062, bf16 autocast, eager]")
    dt_a, mem_a = timed_step(model.model, batch)
    print(f"  upstream cached_kv (masked SDPA):  {dt_a:.3f} s/step   peak {mem_a:.0f} MB")
    cq_flash_patch.enable()
    dt_b, mem_b = timed_step(model.model, batch)
    print(f"  + our valid-key flash attention:   {dt_b:.3f} s/step   peak {mem_b:.0f} MB")
    cq_flash_patch.disable()
    print(f"  speedup: {dt_a / dt_b:.2f}x   memory change: {mem_b - mem_a:+.0f} MB")


if __name__ == "__main__":
    main()
