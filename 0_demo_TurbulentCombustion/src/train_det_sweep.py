"""Wrapper trainer: apply sen_sweep_fixes patches, then run the unified
deterministic trainer unchanged.

Import order matters: sen_sweep_fixes must patch model_baseline BEFORE
train_Det_Baseline is imported, because train_Det_Baseline binds
`from model_baseline import build_dataset` at import time.
"""
import os
import sys

SRC = ("/home/ntricard/generative_reconstruction/temp/"
       "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (SRC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import sen_sweep_fixes  # noqa: F401  (applies env-gated patches)
if os.environ.get("SEN_LOCAL_XATTN", "0") == "1":
    import sen_local_xattn  # noqa: F401  (secondary variant, env-gated)
import train_Det_Baseline

if __name__ == "__main__":
    train_Det_Baseline.main()
