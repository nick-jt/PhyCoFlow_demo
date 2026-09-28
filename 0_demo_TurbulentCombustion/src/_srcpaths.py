"""Register the src/ library directories on sys.path.

Since the 2026-09-27 reorganisation the flat modules that used to sit next to
each other at src/ (``helpers``, ``Model``, ``model_baseline`` ...) live in
sub-directories by purpose (see src/README.md).  Module NAMES are unchanged,
so every ``import helpers`` in the tree keeps working -- provided the
directories below are on ``sys.path``.  Importing this module does that.

Entry points do::

    import _srcpaths          # when src/ itself is already on sys.path
                              # (scripts that live at src/ root)

or, from a sub-directory script, first put src/ on sys.path and then import
this module (the bootstrap block at the top of every moved script).

Nothing here changes numerical behaviour: it only makes the same module
files importable from their new locations.
"""
import os
import sys

# Every directory that holds importable flat modules, relative to src/.
LIB_DIRS = (
    "core",
    "baselines",
    "baselines/confild",
    "baselines/s3gm",
    "baselines/deeponet",
    "baselines/senseiver",
    "data_prep",
    "calibration",
    "spectra",
    "figures",
    "analysis",
    "experiments",
    "benchmarks",
    "ops",
)

_registered = set()


def register(root):
    """Put ``root`` and each of its LIB_DIRS on sys.path (front), once."""
    root = os.path.abspath(root)
    if root in _registered:
        return
    _registered.add(root)
    for rel in ("",) + LIB_DIRS:
        d = os.path.join(root, rel) if rel else root
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


def is_src_root(path):
    """True for this layout (has _srcpaths.py) or the old flat layout
    (model_baseline.py directly inside)."""
    return (os.path.isfile(os.path.join(path, "_srcpaths.py"))
            or os.path.isfile(os.path.join(path, "model_baseline.py")))


def find_module_file(root, filename):
    """Path of ``filename`` inside ``root`` or one of its LIB_DIRS; falls back
    to ``root/filename`` (which then simply does not exist)."""
    for rel in ("",) + LIB_DIRS:
        cand = os.path.join(root, rel, filename) if rel else os.path.join(root, filename)
        if os.path.isfile(cand):
            return cand
    return os.path.join(root, filename)


SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
register(SRC_ROOT)
