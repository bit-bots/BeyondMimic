"""Load 100STYLE-format BVH files (https://zenodo.org/records/8127870) in LAFAN1
convention, so bvh_to_pi_football.json applies unchanged -- same contract as
cmu_bvh.py's load_cmu_bvh_file.

100STYLE is an XSens motion-capture export, the same source format upstream
GMR (github.com/YanjieZe/GMR) already has dedicated support for via its own
ik_configs/bvh_xsens_to_g1.json, used alongside bvh_lafan1_to_g1.json for the
same G1 robot body links. Since both configs target identical robot bodies
from the same class of human motion, comparing their offset_quat values
directly yields the fixed per-bone rotation that converts an XSens/100STYLE
raw bone orientation into LAFAN1 convention:

  G_xsens(f) * offset_xsens == target == G_lafan1_equiv(f) * offset_lafan1
  => G_xsens(f) * (offset_xsens * offset_lafan1^-1) == G_lafan1_equiv(f)

STYLE100_CORRECTIONS below is exactly (offset_xsens * offset_lafan1^-1) per
bone, computed once from those two reference configs -- no T-pose scanning or
per-clip calibration needed, unlike cmu_bvh.py's approach for CMU data.

Per upstream's own xsens.py loader, "FootMod" bones there use the ANKLE's own
raw orientation (not the toe's, unlike load_lafan1_file/load_cmu_bvh_file) --
so LeftAnkle/RightAnkle map directly to LeftFootMod/RightFootMod here, and
LeftToe/RightToe are unused (bvh_to_pi_football.json never references them
directly, only via FootMod).
"""

import re

import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

# Raw 100STYLE/XSens bone name -> LAFAN1-equivalent bone name emitted for
# bvh_to_pi_football.json's ik_match_table. LeftAnkle/RightAnkle map straight
# to the FootMod bones (see module docstring); Hips has no LAFAN1-name change
# but still needs its own correction below.
STYLE100_TO_LAFAN1 = {
    "Chest4": "Spine2",
    "LeftShoulder": "LeftArm",
    "LeftElbow": "LeftForeArm",
    "LeftWrist": "LeftHand",
    "RightShoulder": "RightArm",
    "RightElbow": "RightForeArm",
    "RightWrist": "RightHand",
    "LeftHip": "LeftUpLeg",
    "LeftKnee": "LeftLeg",
    "LeftAnkle": "LeftFootMod",
    "RightHip": "RightUpLeg",
    "RightKnee": "RightLeg",
    "RightAnkle": "RightFootMod",
}

# Fixed per-bone correction (raw 100STYLE/XSens bone name -> quat, wxyz),
# derived from bvh_xsens_to_g1.json vs bvh_lafan1_to_g1.json -- see module
# docstring. G_style100(f) -> world_rot * G_style100(f) * correction puts
# frame f into LAFAN1's convention, same formula cmu_bvh.py uses for CMU.
STYLE100_CORRECTIONS = {
    "Hips": (0.5, -0.5, -0.5, -0.5),
    "Chest4": (0.5, -0.5, -0.5, -0.5),
    "LeftHip": (-0.5, -0.5, -0.5, 0.5),
    "LeftKnee": (-0.5, -0.5, -0.5, 0.5),
    "LeftAnkle": (-0.70710678, -0.70710678, 0.0, 0.0),
    "RightHip": (-0.5, -0.5, -0.5, 0.5),
    "RightKnee": (-0.5, -0.5, -0.5, 0.5),
    "RightAnkle": (-0.70710678, -0.70710678, 0.0, 0.0),
    "LeftShoulder": (0.5, -0.5, 0.5, 0.5),
    "LeftElbow": (0.5, -0.5, 0.5, 0.5),
    "LeftWrist": (0.5, -0.5, 0.5, 0.5),
    "RightShoulder": (0.5, -0.5, 0.5, 0.5),
    "RightElbow": (0.5, -0.5, 0.5, 0.5),
    "RightWrist": (0.5, -0.5, 0.5, 0.5),
}

# Same height/scale assumption as load_lafan1_file/load_cmu_bvh_file --
# human_height_assumption in the IK configs is calibrated to this nominal
# value regardless of the actual subject.
_HUMAN_HEIGHT_M = 1.75

_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

TARGET_FPS = 30


def is_100style_bvh(bvh_file):
    """True for a 100STYLE-style skeleton, detected from bones neither LAFAN1
    nor CMU ever define (100STYLE numbers its spine Chest/Chest2/Chest3/Chest4
    and uses Shoulder/Elbow/Wrist + Hip/Knee/Ankle instead of
    Arm/ForeArm/Hand + UpLeg/Leg/Foot)."""
    with open(bvh_file, "r") as f:
        for line in f:
            if "MOTION" in line:
                break
            if re.match(r"\s*JOINT\s+(Chest2|RightCollar|RightWrist)\b", line):
                return True
    return False


def _read_frame_time(bvh_file):
    with open(bvh_file, "r") as f:
        for line in f:
            m = re.match(r"\s*Frame Time:\s+([\d.]+)", line)
            if m:
                return float(m.group(1))
    raise ValueError(f"no 'Frame Time' in {bvh_file}")


def load_style100_bvh_file(bvh_file, verbose=True):
    """Same contract as load_lafan1_file/load_cmu_bvh_file: (list of per-frame
    dicts, human height)."""
    data = read_bvh(bvh_file)
    global_rot, global_pos = utils.quat_fk(data.quats, data.pos, data.parents)
    bones = list(data.bones)

    world_rot = R.from_matrix(_YUP_TO_ZUP)
    corrections = {
        raw_bone: R.from_quat(q, scalar_first=True)
        for raw_bone, q in STYLE100_CORRECTIONS.items()
        if raw_bone in bones
    }

    src_fps = 1.0 / _read_frame_time(bvh_file)
    stride = max(1, int(round(src_fps / TARGET_FPS)))
    frame_indices = range(0, global_pos.shape[0], stride)

    if verbose:
        print(
            f"[style100_bvh] {bvh_file}: {src_fps:.0f} -> {TARGET_FPS} fps "
            f"(stride {stride}), xsens/lafan1-g1-config-derived correction"
        )

    frames = []
    for frame in frame_indices:
        result = {}
        for raw_bone, correction in corrections.items():
            lafan_bone = STYLE100_TO_LAFAN1.get(raw_bone, raw_bone)
            g_raw = R.from_quat(global_rot[frame, bones.index(raw_bone)], scalar_first=True)
            orientation = (world_rot * g_raw * correction).as_quat(scalar_first=True)
            position = global_pos[frame, bones.index(raw_bone)] @ _YUP_TO_ZUP.T / 100.0
            result[lafan_bone] = (position, orientation)

        frames.append(result)

    return frames, _HUMAN_HEIGHT_M
