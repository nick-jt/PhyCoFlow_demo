"""Wrapper trainer: apply sen_sweep_fixes patches, then run the unified
deterministic trainer unchanged.

Import order matters: sen_sweep_fixes must patch model_baseline BEFORE
train_Det_Baseline is imported, because train_Det_Baseline binds
`from model_baseline import build_dataset` at import time.
"""

# --- src/ path bootstrap (2026-09-27 reorganisation; see src/_srcpaths.py) ---
import os as _os, sys as _sys
_SRC_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
if _SRC_ROOT not in _sys.path:
    _sys.path.insert(0, _SRC_ROOT)
import _srcpaths  # noqa: E402,F401  (puts every src/ library dir on sys.path)
# ------------------------------------------------------------------------------
import os
import sys

SRC = ("/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion/src")
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
