"""Back-compat shim: the symmetry augmentations now live in augment_symmetry.

Kept so already-launched runs that import this name keep working.
"""

# --- src/ path bootstrap (2026-09-27 reorganisation; see src/_srcpaths.py) ---
import os as _os, sys as _sys
_SRC_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
if _SRC_ROOT not in _sys.path:
    _sys.path.insert(0, _SRC_ROOT)
import _srcpaths  # noqa: E402,F401  (puts every src/ library dir on sys.path)
# ------------------------------------------------------------------------------

from augment_symmetry import octahedral_augment  # noqa: F401

__all__ = ["octahedral_augment"]
