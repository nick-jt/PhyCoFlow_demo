"""Historical import shim for the portable persistent Top-K cache."""

# --- src/ path bootstrap (2026-09-27 reorganisation; see src/_srcpaths.py) ---
import os as _os, sys as _sys
_SRC_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
if _SRC_ROOT not in _sys.path:
    _sys.path.insert(0, _SRC_ROOT)
import _srcpaths  # noqa: E402,F401  (puts every src/ library dir on sys.path)
# ------------------------------------------------------------------------------

from phycoflow_pointcloud.cache.geometry import (
    PersistentTopKGeometryCache,
    build_persistent_topk_geometry_cache,
    cache_tensors,
    validate_persistent_topk_geometry_cache,
)

__all__ = [
    "PersistentTopKGeometryCache",
    "build_persistent_topk_geometry_cache",
    "cache_tensors",
    "validate_persistent_topk_geometry_cache",
]
