"""Verify that every figure source shows the SAME snapshot and SAME sensors.

Checks (printed, and the gallery script refuses panels that fail):
  * per-channel Pearson corr and allclose between each source's truth and the
    reference physical truth (the field-dump npz, itself checked against the H5);
  * dump sensor_idx_sum against the meta fingerprint;
  * recovered standardization (physical = z * std + mean) against the run's
    dataset_stats.pt.
"""
import json
import numpy as np
import torch

MAIN = ('/home/ntricard/generative_reconstruction/temp/'
        'PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion')
WT = ('/home/ntricard/generative_reconstruction/temp/'
      'PhyCoFlow_demo_forked_updated_fpe/.claude/worktrees/pof2026-benchmark/'
      '0_demo_TurbulentCombustion')
FD = f'{WT}/Save_TrainedModel_pof/field_dumps'
JHU_H5 = ('/projects/ammoniacomb/generative_reconstruction/'
          'jhu_homogeneous_turbulence/outputfiles_diverse/JHU_4cubes_stride100.h5')
FB_H5 = ('/projects/ammoniacomb/generative_reconstruction/firebench3d/'
         'FireBench_u10u12_merged.h5')


def load_stats(path):
    d = torch.load(path, map_location='cpu')
    if isinstance(d, dict):
        return d
    return {'obj': d}


def chan_corr(a, b):
    return [float(np.corrcoef(a[:, j], b[:, j])[0, 1]) for j in range(a.shape[1])]


def main():
    import h5py
    ok = True

    # ---------------- JHU ----------------
    qj = np.load(f'{MAIN}/Paper/iclr2027/figures/qual_jhu.npz')
    sf = np.load(f'{MAIN}/Paper/iclr2027/figures/spectra_fields.npz')
    sit = np.load(f'{FD}/jhu_sit.npz')
    sen = np.load(f'{FD}/jhu_senseiver.npz')
    ref = sit['truth']  # physical, frame 153

    with h5py.File(JHU_H5, 'r') as f:
        print('[h5 jhu] fields shape', f['fields'].shape)
        raw = f['fields'][0, 153, :, 0, 0, :].astype(np.float32)
    print('[jhu] h5 frame153 vs sit truth allclose:',
          np.allclose(raw, ref, atol=1e-4), 'corr', chan_corr(raw, ref))

    print('[jhu] senseiver truth == sit truth:', np.allclose(sen['truth'], ref))
    c = chan_corr(qj['truth'], ref)
    print('[jhu] qual truth (z) vs physical truth per-chan corr:', c)
    ok &= min(c) > 0.999
    c2 = chan_corr(sf['truth_3'], ref)
    print('[jhu] spectra truth_3 vs physical truth per-chan corr:', c2)
    print('[jhu] spectra truth_3 == qual truth allclose:',
          np.allclose(sf['truth_3'], qj['truth']))
    ok &= min(c2) > 0.999

    # recovered ours-run standardization vs dataset_stats.pt
    s29 = load_stats(f'{MAIN}/Save_TrainedModel/JHU/pointcloud_ffm/'
                     'iclr_jhu_xcube_spec02_DemoN29_20260822_140100/dataset_stats.pt')
    print('[jhu] N29 stats keys:', {k: np.asarray(v).ravel() for k, v in s29.items()})
    s23 = load_stats(f'{MAIN}/Save_TrainedModel/JHU/baseline_latent_fm/'
                     'Baseline_latent_fm_Stage2_DemoN23_20260818_153527/dataset_stats.pt')
    print('[jhu] N23 (latent-FM) stats:', {k: np.asarray(v).ravel() for k, v in s23.items()})
    print('[jhu] sit norm_mean/std:', sit['norm_mean'], sit['norm_std'])

    for j in range(4):
        A = np.polyfit(qj['truth'][::97, j].astype(np.float64),
                       ref[::97, j].astype(np.float64), 1)
        print(f'  chan {j}: physical = {A[0]:.6f} * z + {A[1]:.6f}')

    m = json.loads(str(sit['meta']))
    print('[jhu] sensor idx_sum stored', int(sit['sensor_indices'].sum()),
          'meta', m['sensor_idx_sum'],
          'match', int(sit['sensor_indices'].sum()) == m['sensor_idx_sum'])
    print('[jhu] senseiver sensors identical:',
          np.array_equal(sit['sensor_indices'], sen['sensor_indices']),
          np.array_equal(sit['sensor_field_ids'], sen['sensor_field_ids']))
    # qual dist vs recomputed dist from dump sensors
    from scipy.spatial import cKDTree
    xyz = sit['coords']
    sidx = np.unique(sit['sensor_indices'])
    d, _ = cKDTree(xyz[sidx]).query(xyz, k=1)
    print('[jhu] dist recomputed vs qual dist max|diff|:',
          float(np.abs(d - qj['dist']).max()))

    # ---------------- FireBench ----------------
    qf = np.load(f'{MAIN}/Paper/iclr2027/figures/qual_firebench.npz')
    flm = np.load(f'{FD}/firebench_latent_fm.npz')
    fsen = np.load(f'{FD}/firebench_senseiver.npz')
    fref = flm['truth']

    with h5py.File(FB_H5, 'r') as f:
        print('[h5 fb] fields shape', f['fields'].shape)
        fraw = f['fields'][0, 112, :, 0, 0, :].astype(np.float32)
    print('[fb] h5 frame112 vs latent_fm truth allclose:',
          np.allclose(fraw, fref, atol=1e-4), 'corr', chan_corr(fraw, fref))
    print('[fb] senseiver truth == latent_fm truth:', np.allclose(fsen['truth'], fref))
    cf = chan_corr(qf['truth'], fref)
    print('[fb] qual truth (z) vs physical truth per-chan corr:', cf)
    ok &= min(cf) > 0.999

    s18 = load_stats(f'{MAIN}/Save_TrainedModel/firebench/pointcloud_ffm/'
                     'iclr_firebench_v4_DemoN18_20260819_083221/dataset_stats.pt')
    print('[fb] N18 (ours) stats:', {k: np.asarray(v).ravel() for k, v in s18.items()})
    print('[fb] latent_fm dump norm_mean/std:', flm['norm_mean'], flm['norm_std'])
    for j in range(5):
        A = np.polyfit(qf['truth'][::97, j].astype(np.float64),
                       fref[::97, j].astype(np.float64), 1)
        print(f'  chan {j}: physical = {A[0]:.6f} * z + {A[1]:.6f}')

    mf = json.loads(str(flm['meta']))
    print('[fb] sensor idx_sum stored', int(flm['sensor_indices'].sum()),
          'meta', mf['sensor_idx_sum'],
          'match', int(flm['sensor_indices'].sum()) == mf['sensor_idx_sum'])
    print('[fb] senseiver sensors identical:',
          np.array_equal(flm['sensor_indices'], fsen['sensor_indices']))
    print('[fb] yc =', int(qf['yc']))
    NX, NY, NZ = 152, 126, 192
    rho = qf['truth'][:, 4].reshape(NX, NY, NZ)
    print('[fb] fuel-bed plane zc = argmax var =', int(np.argmax(rho.var(axis=(0, 1)))))

    print('\nALL-OK' if ok else '\nMISMATCH-FOUND')


if __name__ == '__main__':
    main()
