import json, glob, os, sys
import numpy as np
E = ("/home/ntricard/generative_reconstruction/temp/PhyCoFlow_demo_forked_updated_fpe/"
     "0_demo_TurbulentCombustion/Save_TrainedModel/JHU/pointcloud_ffm/"
     "iclr_jhu_xcube_spec02_DemoN29_20260822_140100/Evaluation")
def agg(f):
    d = json.load(open(f)); S = d['snapshots']
    def per(k):
        return np.array([np.mean([s['per_field'][c][k] for c in ('Ux','Uz')]) for s in S])
    sp, rm = per('spread'), per('rmse')
    ratio = sp/rm
    return dict(n_obs=d['n_obs'][0], K=d['K'], nfe=d['n_steps'], n=len(S),
                spread=sp.mean(), rmse=rm.mean(), ratio=sp.mean()/rm.mean(),
                ratio_se=ratio.std(ddof=1)/np.sqrt(len(S)),
                spread_se=sp.std(ddof=1)/np.sqrt(len(S)),
                cov90=per('coverage_90').mean(), cov50=per('coverage_50').mean(),
                crps=per('crps').mean(), rel=per('rel_l2_mean').mean())
rows=[]
for f in sorted(glob.glob(os.path.join(E,'calib_sweep_*.json'))):
    try: rows.append(agg(f))
    except Exception as e: print('skip',f,e)
rows.sort(key=lambda r:(r['K'],r['nfe'],r['n_obs']))
print(f"{'K':>3} {'NFE':>4} {'n_obs':>7} {'%':>6} {'nsnap':>5} {'spread':>7} {'rmse':>7} {'s/e':>6} {'+-':>5} {'cov90':>6} {'cov50':>6} {'crps':>7} {'relL2':>6}")
for r in rows:
    print(f"{r['K']:3d} {r['nfe']:4d} {r['n_obs']:7d} {100*r['n_obs']/1953125:6.2f} {r['n']:5d} "
          f"{r['spread']:7.4f} {r['rmse']:7.4f} {r['ratio']:6.3f} {r['ratio_se']:5.3f} "
          f"{r['cov90']:6.3f} {r['cov50']:6.3f} {r['crps']:7.4f} {r['rel']:6.4f}")
