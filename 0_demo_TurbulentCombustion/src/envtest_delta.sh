#!/bin/bash
#SBATCH --job-name=envtest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/envtest_%j.log
# DeltaAI environment preflight: torch+CUDA, KeOps compile, h5py on the 2D datasets.
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python - <<'PY'
import torch, time, h5py, numpy as np
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t=time.time()
    for _ in range(10): y = x @ x
    torch.cuda.synchronize(); print("matmul 10x4096^3: %.3fs" % (time.time()-t))
    torch.manual_seed(0); print("randperm fingerprint", torch.randperm(65536, device="cuda")[:5].tolist())
try:
    from pykeops.torch import LazyTensor
    a = torch.randn(10000, 3, device="cuda"); b = torch.randn(20000, 3, device="cuda")
    t=time.time(); d = ((LazyTensor(a[:,None,:]) - LazyTensor(b[None,:,:]))**2).sum(-1); i = d.argKmin(16, dim=1); torch.cuda.synchronize()
    print("keops argKmin ok", i.shape, "%.1fs (incl. compile)" % (time.time()-t))
except Exception as e:
    print("KEOPS FAIL", repr(e))
for f in ["/work/hdd/bilr/ntricard/datasets/kolmogorov2d/Kolmogorov2D_shu_stride4.h5",
          "/work/hdd/bilr/ntricard/datasets/cylinder2d/Cylinder2D_mesh.h5",
          "/work/hdd/bilr/ntricard/datasets/cylinder2d/Cylinder2D_grid.h5"]:
    with h5py.File(f) as h:
        print(f.split('/')[-1], {k: h[k].shape for k in h.keys()})
PY
echo "=== envtest done ==="
