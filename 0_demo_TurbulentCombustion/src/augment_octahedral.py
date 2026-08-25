"""Back-compat shim: the symmetry augmentations now live in augment_symmetry.

Kept so already-launched runs that import this name keep working.
"""

from augment_symmetry import octahedral_augment  # noqa: F401

__all__ = ["octahedral_augment"]
