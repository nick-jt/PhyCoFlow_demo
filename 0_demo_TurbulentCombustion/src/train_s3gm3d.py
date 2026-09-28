"""Launcher: swap the 3-D S3GM adapter into model_baseline's registry, then run
the shared generative trainer unchanged.

model_baseline.py's built-in S3GMAdapter is 2-D only (it calls
validate_regular_grid_compatibility(num_x, num_y), which raises on a 125^3
grid).  get_baseline_adapter() resolves `S3GMAdapter` from module globals at
call time, so rebinding that name is enough -- no shared file is edited.
"""

# --- src/ path bootstrap (2026-09-27 reorganisation; see src/_srcpaths.py) ---
import os as _os, sys as _sys
_SRC_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _SRC_ROOT not in _sys.path:
    _sys.path.insert(0, _SRC_ROOT)
import _srcpaths  # noqa: E402,F401  (puts every src/ library dir on sys.path)
# ------------------------------------------------------------------------------
import sys
import model_baseline as MB
from s3gm3d import S3GM3DAdapter

MB.S3GMAdapter = S3GM3DAdapter

import train_Gen_Baseline as T

if __name__ == "__main__":
    T.main()
