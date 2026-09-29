"""The synthetic "MixedRoot" body the bvh_to_pi_football retarget is rooted at.

bvh_to_pi_football.json used to anchor base_link on Spine2, the UPPER spine,
which tipped the robot's torso horizontal during running. MixedRoot replaces
it with the pelvis position plus an orientation slerped from the pelvis toward
the upper spine, so the base follows the pelvis while still picking up part of
the torso lean. root_blend 0 = pure pelvis (what GMR's other bvh_to_*.json
configs use, as a plain "Hips" root), 1 = the old Spine2 orientation.

Both BVH loaders inject it, the same way they inject Left/RightFootMod, so
every consumer of their frames has it: the IK config names MixedRoot both as
human_root_name (the pivot scale_human_data scales all other bodies around)
and as base_link's IK target, and GMR raises KeyError without it.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

DEFAULT_ROOT_BLEND = 0.5


def inject_mixed_root(frame: dict, blend: float = DEFAULT_ROOT_BLEND) -> dict:
    """Add frame["MixedRoot"]: Hips position, slerp(Hips, Spine2, blend) rotation.

    Mutates and returns `frame`. Quats are wxyz (scalar-first), as both loaders
    emit them. Sources that emit no "Hips" at all fall back to Spine2 alone --
    for those, blend has no effect.
    """
    if "Hips" not in frame:
        frame["MixedRoot"] = frame["Spine2"]
        return frame
    hips_pos, hips_q = frame["Hips"]
    _, spine_q = frame["Spine2"]
    key = R.from_quat(np.array([hips_q, spine_q]), scalar_first=True)
    mixed_q = Slerp([0.0, 1.0], key)([blend])[0].as_quat(scalar_first=True)
    frame["MixedRoot"] = (hips_pos, mixed_q)
    return frame
