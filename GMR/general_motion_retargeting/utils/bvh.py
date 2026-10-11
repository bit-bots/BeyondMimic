"""Single entry point for loading a BVH clip, whichever dataset it came from.

LAFAN1, CMU and 100STYLE clips need different loaders -- CMU's and 100STYLE's
skeletons are calibrated onto LAFAN1's bone convention first, see cmu_bvh.py
and style100_bvh.py -- but every consumer
wants the same thing out: per-frame dicts in LAFAN1 naming, MixedRoot
included. Use this instead of picking a loader by hand, so all datasets stay
processable by the same scripts.
"""

from __future__ import annotations

from .cmu_bvh import is_cmu_bvh, load_cmu_bvh_file
from .lafan1 import load_lafan1_file
from .mixed_root import DEFAULT_ROOT_BLEND
from .style100_bvh import is_100style_bvh, load_style100_bvh_file


def load_bvh_file(bvh_file, root_blend: float = DEFAULT_ROOT_BLEND):
    """Load a LAFAN1, CMU or 100STYLE BVH clip -> (list of per-frame dicts, human height).

    The dataset is auto-detected from the skeleton. See mixed_root.py for what
    root_blend does.
    """
    if is_cmu_bvh(bvh_file):
        loader = load_cmu_bvh_file
    elif is_100style_bvh(bvh_file):
        loader = load_style100_bvh_file
    else:
        loader = load_lafan1_file
    return loader(bvh_file, root_blend=root_blend)
