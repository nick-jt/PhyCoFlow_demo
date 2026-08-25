"""Break down one training step: obs build / query sample / fwd / bwd, fp32 vs bf16, K=64 vs 32."""
import time, torch
from ensemble_eval import load_run
from helpers import build_sparse_condition
from train_pointcloud_ffm import sample_query_subset
model, dataset, cfg = load_run('../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_main_DemoN4_20260813_122006', 'last.pt')
device = torch.device('cuda:0'); model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
B = 20
items = [dataset[i] for i in range(B)]
coords = torch.stack([it['coords'] for it in items]).to(device)
fields = torch.stack([it['fields'] for it in items]).to(device)
def sync(): torch.cuda.synchronize()
def bench(tag, use_amp, topk=None):
    if topk is not None: model.model.gather_topk = topk
    for rep in range(3):
        sync(); t0=time.perf_counter()
        oc,ov,om,oi,ofid = build_sparse_condition(coords_full=coords, fields_full=fields,
            cond_fields=[0,2], n_obs_min=[1953], n_obs_max=[19531])
        sync(); t1=time.perf_counter()
        cq, fq, _ = sample_query_subset(coords=coords, fields=fields, n_query=19531, mode='uniform',
            obs_coords=oc, obs_mask=om, obs_counts=None)
        sync(); t2=time.perf_counter()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            loss,_ = model.training_loss(x1=fq, coords=cq, obs_coords=oc, obs_values=ov,
                obs_mask=om, obs_field_ids=ofid, compute_metrics=False)
        sync(); t3=time.perf_counter()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sync(); t4=time.perf_counter()
    print(f"{tag}: obs={t1-t0:.3f}s query={t2-t1:.3f}s fwd={t3-t2:.3f}s bwd+opt={t4-t3:.3f}s "
          f"total={t4-t0:.3f}s peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
    torch.cuda.reset_peak_memory_stats()
bench('bf16 K=64', True)
bench('fp32 K=64', False)
bench('bf16 K=32', True, topk=32)
bench('bf16 K=16', True, topk=16)
# torch.compile test
model.model = torch.compile(model.model, mode='max-autotune-no-cudagraphs')
try:
    bench('bf16 K=16 compiled (incl warmup)', True)
    bench('bf16 K=16 compiled (steady)', True)
except Exception as e:
    print('compile failed:', str(e)[:200])
