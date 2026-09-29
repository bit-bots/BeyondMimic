"""Single entry point for loading a BVH clip, whichever dataset it came from.

LAFAN1 and CMU clips need different loaders -- CMU's skeleton is calibrated
onto LAFAN1's bone convention first, see cmu_bvh.py -- but every consumer
wants the same thing out: per-frame dicts in LAFAN1 naming, MixedRoot
included. Use this instead of picking a loader by hand, so both datasets stay
processable by the same scripts.
"""

from __future__ import annotations

from .cmu_bvh import is_cmu_bvh, load_cmu_bvh_file
from .lafan1 import load_lafan1_file
from .mixed_root import DEFAULT_ROOT_BLEND


def load_bvh_file(bvh_file, root_blend: float = DEFAULT_ROOT_BLEND):
    """Load a LAFAN1 or CMU BVH clip -> (list of per-frame dicts, human height).

    The dataset is auto-detected from the skeleton. See mixed_root.py for what
    root_blend does.
    """
    loader = load_cmu_bvh_file if is_cmu_bvh(bvh_file) else load_lafan1_file
    return loader(bvh_file, root_blend=root_blend)
