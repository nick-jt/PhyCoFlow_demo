#!/bin/bash
#SBATCH --job-name=keopstest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ghx4
#SBATCH --account=bilr-dtai-gh
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/keopstest_%j.log
set -u
source ~/envs/jhtdb
cd $SLURM_SUBMIT_DIR
python - <<'PY'
import torch, traceback
from pykeops.torch import LazyTensor
def knn(q, o, mask, k=16):
    x_i = LazyTensor(q[:, :, None, :]); y_j = LazyTensor(o[:, None, :, :])
    d = ((x_i - y_j)**2).sum(-1)
    m_j = LazyTensor(mask[:, None, :, None].to(q.dtype).contiguous())
    d = d + (1.0 - m_j) * 1e6
    return d.Kmin_argKmin(K=k, dim=2)
B,N,M,D = 20,1310,655,3
def run(tag, **kw):
    try:
        q = torch.randn(B,N,D, device="cuda", **kw); o = torch.randn(B,M,D, device="cuda", **kw); mask=(torch.rand(B,M,device="cuda")>0.3).float()
        v,i = knn(q,o,mask); torch.cuda.synchronize(); print(tag, "ok", v.dtype, i.dtype, i.shape, flush=True)
    except Exception as e:
        print(tag, "FAIL", type(e).__name__, str(e)[:200], flush=True)
run("fp32", dtype=torch.float32)
run("fp16", dtype=torch.float16)
run("bf16", dtype=torch.bfloat16)
try:
    q = torch.randn(B,N,D, device="cuda"); o = torch.randn(B,M,D, device="cuda"); mask=(torch.rand(B,M,device="cuda")>0.3).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v,i = knn(q,o,mask)
    print("autocast-bf16 ok", v.dtype, flush=True)
except Exception as e:
    print("autocast FAIL", type(e).__name__, str(e)[:200], flush=True)
try:
    f = torch.compile(lambda q,o,m: knn(q,o,m))
    q = torch.randn(B,N,D, device="cuda"); o = torch.randn(B,M,D, device="cuda"); mask=(torch.rand(B,M,device="cuda")>0.3).float()
    v,i = f(q,o,mask); torch.cuda.synchronize(); print("compiled ok", i.shape, flush=True)
except Exception as e:
    print("compiled FAIL", type(e).__name__, str(e)[:300], flush=True)
    traceback.print_exc()
try:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v,i = f(q,o,mask)
    print("compiled+autocast ok", flush=True)
except Exception as e:
    print("compiled+autocast FAIL", type(e).__name__, str(e)[:300], flush=True)
PY
