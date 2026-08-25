"""Gen4Turb cube-3 reconstruction eval at matched snapshots and sensors.

Their conditioning is a mask SHARED across all 4 channels, drawn U(0,1)
coverage during training. We evaluate in-distribution at 1% shared coverage
(all 4 fields observed at the sensors) - an INFORMATION ADVANTAGE over our
protocol (Ux+Uz only). A strict per-channel variant (u,w observed, v,p fully
masked) is also run, flagged as out-of-their-training-distribution.
Scores use our ensemble_metrics after converting to the same z-score units
as every other JHU number (per-field mean/std of our merged dataset, train
frames only).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

DM = ("/projects/ammoniacomb/generative_reconstruction/baselines/"
      "Gen4Turbulence/3_flow_reconstruction/dm")
DATA = Path(DM).parent / "data"
# our matched val snapshots (val idx -> global frame 150+idx)
VAL_IDS = [0, 1, 3, 12, 14, 23, 28, 36]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="best_model.pt")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--coverage", type=float, default=0.01)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sys.path.insert(0, DM)
    from utils.architecture import Unet
    from utils.diffusion import ElucidatedDiffusion
    from ensemble_eval import ensemble_metrics

    dev = torch.device("cuda:0")
    import pickle
    par = pickle.load(open(Path(DM) / "Par.pkl", "rb"))
    net = Unet(dim=16, dim_mults=(1, 2, 4, 8), channels=4,
               self_condition=True, flash_attn=True)
    model = ElucidatedDiffusion(net, channels=4, image_size_h=120,
                                image_size_w=120, image_size_d=120,
                                sigma_data=float(par["sigma_data"])).to(dev)
    print(f"sigma_data from Par.pkl: {par['sigma_data']:.6f}")
    sd = torch.load(str(Path(DM) / "models" / args.ckpt), map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    u = np.load(DATA / "u.npy")                      # [4, 200, 120,120,120]
    mn = np.load(DATA / "MIN_u.npy").reshape(1, 4, 1, 1, 1)
    mx = np.load(DATA / "MAX_u.npy").reshape(1, 4, 1, 1, 1)

    # z-score stats in OUR convention: per-field mean/std over train frames
    tr = u[:, :150].reshape(4, -1)
    z_mean = tr.mean(axis=1).reshape(1, 1, 4)
    z_std = tr.std(axis=1).reshape(1, 1, 4)

    field_names = ["Ux", "Uy", "Uz", "p"]
    n_vox = 120 ** 3
    n_obs = int(round(args.coverage * n_vox))
    results = {}
    for variant in ["shared", "strict"]:
        cases = []
        for vi in VAL_IDS:
            frame = 150 + vi
            x_phys = torch.from_numpy(u[:, frame][None].copy())          # [1,4,...]
            x_norm = ((x_phys - torch.from_numpy(mn)) /
                      torch.from_numpy(mx - mn)).float().to(dev)
            g = torch.Generator().manual_seed(1000 + vi)
            if variant == "shared":
                mask = torch.zeros(n_vox)
                mask[torch.randperm(n_vox, generator=g)[:n_obs]] = 1
                mask = mask.reshape(1, 1, 120, 120, 120).repeat(1, 4, 1, 1, 1)
            else:  # strict: own 1% sets on Ux,Uz; Uy,p unobserved
                mask = torch.zeros(1, 4, n_vox)
                for c in (0, 2):
                    mask[0, c, torch.randperm(n_vox, generator=g)[:n_obs]] = 1
                mask = mask.reshape(1, 4, 120, 120, 120)
            mask = mask.to(dev)
            cond = torch.cat([x_norm * mask, mask], dim=1)               # [1,8,...]
            ens = []
            t0 = time.time()
            with torch.no_grad():
                for k in range(args.K):
                    torch.manual_seed(9000 + 97 * vi + k)
                    s = model.sample(cond.float())                        # [1,4,...]
                    ens.append(s[0].float().cpu())
            ens = torch.stack(ens)                                        # [K,4,...]
            # -> physical -> our z-score units, [K, N, F]
            ens_p = ens * torch.from_numpy(mx - mn)[0].float() + torch.from_numpy(mn)[0].float()
            ens_z = (ens_p.reshape(args.K, 4, -1).permute(0, 2, 1).numpy() - z_mean) / z_std
            true_z = ((x_phys[0].reshape(4, -1).T.numpy()) - z_mean[0]) / z_std[0]
            m = ensemble_metrics(ens_z, true_z, field_names)
            m["snapshot"] = vi
            m["sample_seconds"] = round((time.time() - t0) / args.K, 2)
            cases.append(m)
            a = m["aggregate"]
            print(f"[{variant}] snap {vi}: relL2_mean={a['rel_l2_mean']:.4f} "
                  f"crps={a['crps']:.4f} spread_err={a['spread_error_ratio']:.3f} "
                  f"cov90={a['coverage_90']:.3f}", flush=True)
        keys = cases[0]["aggregate"].keys()
        results[variant] = {
            "aggregate": {k: float(np.mean([c["aggregate"][k] for c in cases])) for k in keys},
            "cases": cases,
        }
        print(f"== {variant} MEAN:", {k: round(v, 4) for k, v in results[variant]["aggregate"].items()}, flush=True)
    results["config"] = vars(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
