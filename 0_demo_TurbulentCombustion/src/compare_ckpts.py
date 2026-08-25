import numpy as np, torch
from ensemble_eval import load_run, sample_ensemble
from helpers import build_sparse_condition
GRID=125
def shell(g):
    f=np.abs(np.fft.fftn(g))**2
    k=np.fft.fftfreq(GRID)*GRID
    KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij')
    kb=np.round(np.sqrt(KX**2+KY**2+KZ**2)).astype(int)
    return np.bincount(kb.ravel(),weights=f.ravel())
device=torch.device('cuda:0')
for tag,rd,ck in [("DemoN4(EMA)","../Save_TrainedModel/JHU/pointcloud_ffm/iclr_jhu_main_DemoN4_20260813_122006","best.pt"),
                  ("DemoN3(July)","../Save_TrainedModel/_legacy/ffm_tc_pointcloud_match_lfm_DemoN3_20260713_081738","best.pt")]:
    model,dataset,_=load_run(rd,ck)
    item=dataset[0]; coords=item['coords'][None].to(device); fields=item['fields'][None].to(device)
    true=item['fields'].numpy()
    torch.manual_seed(1)
    oc,ov,om,oi,ofid=build_sparse_condition(coords_full=coords,fields_full=fields,cond_fields=[0,2],n_obs_min=[19531],n_obs_max=[19531])
    obs={"coords":oc,"values":ov,"mask":om,"indices":oi,"field_ids":ofid}
    ens=sample_ensemble(model,coords,obs,K=1,n_steps=16,chunk=262144,clamp_hard=False,seed=5)
    e=ens[0].numpy()
    for j,nm in [(0,'Ux'),(1,'Uy')]:
        sp=shell(e[:,j].reshape(GRID,GRID,GRID)); st=shell(true[:,j].reshape(GRID,GRID,GRID))
        print(f"{tag} {nm}: relL2={float(np.linalg.norm(e[:,j]-true[:,j])/np.linalg.norm(true[:,j])):.3f} "
              f"inertial={sp[8:32].sum()/st[8:32].sum():.3f} dissip={sp[32:63].sum()/st[32:63].sum():.3f}", flush=True)
    del model; torch.cuda.empty_cache()
