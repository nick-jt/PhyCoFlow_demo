#!/bin/bash
#SBATCH --job-name=kprior_smoke
#SBATCH --time=0:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --account=f2pde
#SBATCH --mem=96G
set -u
export JHU_SPLIT_MODE=block JHU_SPLIT_GAP=0
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
L=kprior_smoke_${SLURM_JOB_ID}.log
python - <<'PY' >> $L 2>&1
import numpy as np, torch, sys
sys.path.insert(0, ".")
from spectral_prior import PowerLawRFFPrior
import helpers as H
GRID = 125
dev = "cuda:0"
lin = torch.linspace(0, 1, GRID, device=dev)
X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
coords = torch.stack([X, Y, Z], -1).reshape(1, -1, 3)
def shell(g):
    f = np.abs(np.fft.fftn(g))**2
    k = np.fft.fftfreq(GRID)*GRID
    KX,KY,KZ = np.meshgrid(k,k,k,indexing='ij')
    kb = np.round(np.sqrt(KX**2+KY**2+KZ**2)).astype(int)
    return np.bincount(kb.ravel(), weights=f.ravel())[1:63]
torch.cuda.reset_peak_memory_stats()
p = PowerLawRFFPrior(n_features=1024, slope=5/3, k_min=1.0, k_max=48.0).to(dev)
x = p(coords, 4)
print("draw shape", tuple(x.shape), "peak MB",
      torch.cuda.max_memory_allocated()/1024**2)
s = shell(x[0,:,0].cpu().numpy().reshape(GRID,GRID,GRID))
ks = np.arange(1,63)
sl = np.polyfit(np.log(ks[3:40]), np.log(s[3:40]+1e-30), 1)[0]
print(f"prior E(k) fitted slope {sl:+.2f} (target -1.67); frac energy k>8 = {s[7:].sum()/s.sum():.3f}, k>31 = {s[30:].sum()/s.sum():.3f}")
PY
echo "exit status: $?" >> $L
