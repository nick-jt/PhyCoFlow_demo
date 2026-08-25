import numpy as np, torch, glob
from ensemble_eval import load_run, sample_ensemble
from helpers import build_sparse_condition
run = sorted(glob.glob('../Save_TrainedModel/firebench/pointcloud_ffm/iclr_firebench_v2_*'))[-1]
device = torch.device('cuda:0')
for ck in ['last.pt']:
    model, dataset, cfg = load_run(run, ck)
    item = dataset[len(dataset)-1]
    coords = item['coords'][None].to(device); fields = item['fields'][None].to(device)
    torch.manual_seed(3)
    oc,ov,om,oi,ofid = build_sparse_condition(coords_full=coords, fields_full=fields,
        cond_fields=[0,1,2], n_obs_min=[20000], n_obs_max=[20000])
    obs={"coords":oc,"values":ov,"mask":om,"indices":oi,"field_ids":ofid}
    ens = sample_ensemble(model, coords, obs, K=4, n_steps=16, chunk=262144, clamp_hard=False, seed=11)
    e = ens.numpy(); true = item['fields'].numpy()
    # quick 1D line-spectrum ratio for u at low height
    c = item['coords'].numpy()
    nx, nh, ny = 152, 126, 192
    tg = true[:,0].reshape(nx,nh,ny); sg = e[0][:,0].reshape(nx,nh,ny)
    def spec(v):
        rows = v[:,:,20] - v[:,:,20].mean(0, keepdims=True)
        return (np.abs(np.fft.rfft(rows, axis=0))**2).mean(1)
    st, ss = spec(tg), spec(sg)
    n=len(st)
    print(f"{ck}: u sample/truth 1D-spectrum ratio: large={ss[1:n//8].sum()/st[1:n//8].sum():.2f} "
          f"mid={ss[n//8:n//2].sum()/st[n//8:n//2].sum():.2f} small={ss[n//2:].sum()/st[n//2:].sum():.2f} | "
          f"relL2(mean)={np.linalg.norm(e.mean(0)-true)/np.linalg.norm(true):.3f}", flush=True)
    tag = ck.replace('.pt','')
    np.savez(f'{run}/Evaluation/firebench_field3d_{tag}.npz',
             truth=true, mean=e.mean(0), sample=e[0], coords=c)
    del model; torch.cuda.empty_cache()
print('done')
