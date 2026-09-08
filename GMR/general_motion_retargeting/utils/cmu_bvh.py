"""Load CMU-style BVH files (e.g. the soccer_kicks clips) in LAFAN1 convention.

The CMU mocap skeleton differs from LAFAN1 in three ways, all of which have to
be undone before the existing bvh_to_*.json IK configs mean anything:

  * scale     - CMU units are ~15.9 per metre, LAFAN1 is centimetres (100/m)
  * topology  - extra LHipJoint/RHipJoint, LowerBack, Neck1 and finger bones,
                no Spine2, and the toes are called LeftToeBase/RightToeBase
  * bone axes - the local frames differ per bone by up to ~170 degrees

Nothing here rewrites the .bvh on disk; the conversion happens in memory so the
loader is a drop-in alternative to load_lafan1_file.

The bone-axis difference is recovered rather than guessed: both datasets record
a T-pose (LAFAN1 in frame 0 of most clips, CMU in frame 0 of every clip), so

    C_b = G_cmu_b(T)^-1 @ G_lafan_b(T)

is a constant per-bone correction, and G_cmu_b(f) @ C_b puts frame f into
LAFAN1's convention. That keeps bvh_to_pi_football.json applicable unchanged.

LAFAN1_TPOSE_QUAT below is the mean over the 40 LAFAN1 clips whose frame 0 is
within 12 degrees of a textbook T-pose (spread vs. that mean: 2.9 deg for
Spine2, ~5 deg for the legs).
"""

import re

import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

# CMU bone name -> LAFAN1 bone name. CMU's Spine1 is the chest: it parents Neck
# and both shoulders, which is LAFAN1's Spine2 (CMU has no bone of that name, so
# the rename cannot collide).
CMU_TO_LAFAN1 = {
    "Spine1": "Spine2",
    "LeftToeBase": "LeftToe",
    "RightToeBase": "RightToe",
}

# Child bone used to read a bone's true direction from joint positions, which is
# independent of either skeleton's frame convention. The child must sit at a fixed
# offset from the bone, so that the direction is rigidly attached to it and the
# correction derived at frame 0 stays valid for every later frame.
#
# Spine1 is deliberately absent: all of its children (Neck, LeftShoulder,
# RightShoulder) sit at zero offset, so no rigid direction exists, and reaching
# past them to Neck1 would drag the neck joint's own rotation in. It keeps
# posture = identity, i.e. pure convention matching -- which is also the safe
# choice for the bone the IK uses as its root. Toes and hands are leaves and are
# likewise absent.
CMU_BONE_CHILD = {
    "LeftUpLeg": "LeftLeg",
    "RightUpLeg": "RightLeg",
    "LeftLeg": "LeftFoot",
    "RightLeg": "RightFoot",
    "LeftFoot": "LeftToeBase",
    "RightFoot": "RightToeBase",
    "LeftArm": "LeftForeArm",
    "RightArm": "RightForeArm",
    "LeftForeArm": "LeftHand",
    "RightForeArm": "RightHand",
}

# Global rotations (raw BVH space, wxyz) of a LAFAN1 T-pose.
LAFAN1_TPOSE_QUAT = {
    "Spine2": (0.46783835, 0.52775842, 0.46302757, 0.53684616),
    "LeftUpLeg": (0.51883616, 0.50691629, -0.48092433, -0.49250046),
    "RightUpLeg": (0.49996445, 0.48963527, -0.51154895, -0.49860859),
    "LeftLeg": (0.47225235, 0.53279265, -0.42476168, -0.55918443),
    "RightLeg": (0.40360111, 0.57620069, -0.47321477, -0.53025153),
    "LeftFoot": (0.73213104, 0.11869716, -0.65840819, -0.12803824),
    "RightFoot": (0.62539965, 0.13845710, -0.76001391, -0.10992622),
    "LeftToe": (0.74318690, -0.00592984, -0.66897561, 0.01047392),
    "RightToe": (0.63491044, -0.00542716, -0.77252098, 0.00840325),
    "LeftArm": (0.68642255, -0.72710500, -0.00953441, 0.00717604),
    "RightArm": (0.02672458, 0.00699290, 0.68457874, -0.72841530),
    "LeftForeArm": (0.67293929, -0.72220607, -0.15925125, 0.01449608),
    "RightForeArm": (0.16519840, -0.01125778, -0.66770784, 0.72577475),
    "LeftHand": (0.73965058, -0.66482407, -0.06230890, 0.08392608),
    "RightHand": (0.07498428, -0.09259737, -0.71467217, 0.68923638),
}

# Head height above the lowest joint in that same LAFAN1 T-pose, in metres after
# the loader's /100. Mean 159.04 raw units, std 0.96 over the 40 clips.
LAFAN1_TPOSE_HEAD_HEIGHT_M = 1.5904

# Bone directions a T-pose must exhibit (world, Z-up), used to sanity check that
# frame 0 really is the calibration pose.
_TPOSE_CHECKS = [
    ("LeftArm", "LeftForeArm", (1.0, 0.0, 0.0)),
    ("RightArm", "RightForeArm", (-1.0, 0.0, 0.0)),
    ("LeftUpLeg", "LeftLeg", (0.0, 0.0, -1.0)),
    ("RightUpLeg", "RightLeg", (0.0, 0.0, -1.0)),
    ("Spine", "Neck", (0.0, 0.0, 1.0)),
]
_TPOSE_TOL_DEG = 15.0

# Y-up (BVH) -> Z-up (MuJoCo), same matrix the LAFAN1 loader uses.
_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

TARGET_FPS = 30


def is_cmu_bvh(bvh_file):
    """True for a CMU-style skeleton, detected from bones LAFAN1 never defines."""
    with open(bvh_file, "r") as f:
        for line in f:
            if "MOTION" in line:
                break
            if re.match(r"\s*JOINT\s+(LowerBack|LeftToeBase|LHipJoint)\b", line):
                return True
    return False


def _read_frame_time(bvh_file):
    """read_bvh parses Frame Time but drops it, so pull it out of the header."""
    with open(bvh_file, "r") as f:
        for line in f:
            m = re.match(r"\s*Frame Time:\s+([\d.]+)", line)
            if m:
                return float(m.group(1))
    raise ValueError(f"no 'Frame Time' in {bvh_file}")


def _min_rotation(a, b):
    """Smallest rotation taking unit-ish vector a onto b (identity if degenerate)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:  # zero-length bone, e.g. CMU's Spine1 -> Neck
        return R.identity()
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return R.identity()
    return R.from_rotvec(v / s * np.arctan2(s, float(a @ b)))


def _tpose_error_deg(pos_zup, bones):
    """Worst angular deviation of frame 0 from a textbook T-pose, in degrees."""
    worst = 0.0
    for parent, child, ideal in _TPOSE_CHECKS:
        v = pos_zup[bones.index(child)] - pos_zup[bones.index(parent)]
        v = v / np.linalg.norm(v)
        cos = np.clip(v @ np.array(ideal), -1.0, 1.0)
        worst = max(worst, np.degrees(np.arccos(cos)))
    return worst


def load_cmu_bvh_file(bvh_file, verbose=True):
    """Same contract as load_lafan1_file: (list of per-frame dicts, human height).

    Frame 0 is consumed as the calibration T-pose and is not part of the output,
    and the 120 fps source is decimated to 30 fps to match the rest of the
    pipeline (scripts/bvh_to_robot.py hardcodes motion_fps = 30).
    """
    data = read_bvh(bvh_file)
    global_rot, global_pos = utils.quat_fk(data.quats, data.pos, data.parents)
    bones = list(data.bones)

    # --- validate + calibrate against frame 0 -------------------------------
    tpose_err = _tpose_error_deg(global_pos[0] @ _YUP_TO_ZUP.T, bones)
    if tpose_err > _TPOSE_TOL_DEG:
        raise ValueError(
            f"{bvh_file}: frame 0 deviates {tpose_err:.1f} deg from a T-pose "
            f"(tolerance {_TPOSE_TOL_DEG}). The frame-axis calibration needs a "
            f"T-pose there; check whether this clip really is CMU-style."
        )

    # Metric scale from the subject's own T-pose height, so per-subject skeleton
    # differences are absorbed instead of relying on a fixed divisor.
    pos_zup_t = global_pos[0] @ _YUP_TO_ZUP.T
    head_height = pos_zup_t[bones.index("Head")][2] - pos_zup_t[:, 2].min()
    scale = LAFAN1_TPOSE_HEAD_HEIGHT_M / head_height

    # C_b, per bone, in raw BVH space.
    #
    # Matching frame 0 straight onto the LAFAN1 reference would absorb the pose the
    # CMU subject actually held while being calibrated -- their arms hang ~8 deg
    # below horizontal -- and silently raise the arms in every output frame. So the
    # frame-0 difference is split: `posture` is the minimal world rotation carrying
    # the reference bone direction onto the direction CMU's joint positions really
    # show, and only what is left is treated as a frame-convention difference. Bone
    # directions come from positions, which no convention can distort; the axial
    # roll, which a T-pose cannot pin down, still follows LAFAN1.
    world_rot = R.from_matrix(_YUP_TO_ZUP)
    corrections = {}
    for cmu_bone in bones:
        lafan_bone = CMU_TO_LAFAN1.get(cmu_bone, cmu_bone)
        if lafan_bone not in LAFAN1_TPOSE_QUAT:
            continue
        q_cmu = global_rot[0, bones.index(cmu_bone)]  # wxyz
        g_cmu = R.from_quat(q_cmu, scalar_first=True)
        g_lafan = R.from_quat(LAFAN1_TPOSE_QUAT[lafan_bone], scalar_first=True)

        child = CMU_BONE_CHILD.get(cmu_bone)
        if child is None:
            posture = R.identity()
        else:
            # LAFAN1's local +X is the bone axis, so this is where the uncorrected
            # output would point that bone.
            ref_dir = (world_rot * g_lafan).apply([1.0, 0.0, 0.0])
            cmu_dir = pos_zup_t[bones.index(child)] - pos_zup_t[bones.index(cmu_bone)]
            posture = _min_rotation(ref_dir, cmu_dir)

        corrections[cmu_bone] = (
            g_cmu.inv() * world_rot.inv() * posture * world_rot * g_lafan
        )

        # Pi Plus fix: LeftArm's axial roll (twist about the bone's own +X axis)
        # is unobservable from a T-pose by construction -- position data alone
        # can't distinguish it, so it silently follows LAFAN1's own convention
        # (see the "posture" comment above). For this CMU dataset that leaves a
        # ~180 deg twist error, confirmed constant across every CMU clip checked
        # (CMU's frame 0 bakes a fixed bind pose independent of the captured
        # motion, so the derived correction -- and this error -- is identical
        # for all of them).
        #
        # The whole LeftArm/LeftForeArm/LeftHand chain needs the SAME twist
        # applied, not just LeftArm: each bone's correction is independently
        # derived as an absolute-world target, so twisting only the shoulder
        # leaves the elbow bridging a now-mismatched gap between a rotated
        # upper-arm target and untouched forearm/hand targets -- visibly
        # breaking the arm. Rotating all three together instead leaves their
        # RELATIVE orientation (elbow bend, wrist pose) unchanged and just
        # reorients the whole chain, matching a real T-pose calibration fix.
        # Verified on 69_34.bvh: with only LeftArm twisted, l_upper_arm_joint
        # stops pinning at its range limit but the arm looks broken in the
        # viewer; with the whole chain twisted, l_upper_arm settles near 0 deg
        # (matching bitbots_main's real walkready pose) and l_shoulder_pitch
        # (~93 deg) / l_shoulder_roll (~73 deg) land close to walkready's 90/70
        # deg too, with the elbow bend symmetric to the (already-correct)
        # right arm -- no pinning anywhere.
        if cmu_bone in ("LeftArm", "LeftForeArm", "LeftHand"):
            corrections[cmu_bone] = corrections[cmu_bone] * R.from_rotvec([np.pi, 0.0, 0.0])

    # --- resample ----------------------------------------------------------
    src_fps = 1.0 / _read_frame_time(bvh_file)
    stride = max(1, int(round(src_fps / TARGET_FPS)))
    frame_indices = range(1, global_pos.shape[0], stride)  # skip the T-pose

    if verbose:
        print(
            f"[cmu_bvh] {bvh_file}: {src_fps:.0f} -> {TARGET_FPS} fps (stride {stride}), "
            f"scale 1/{1 / scale:.2f}, T-pose residual {tpose_err:.1f} deg"
        )

    frames = []
    for frame in frame_indices:
        result = {}
        for cmu_bone, correction in corrections.items():
            lafan_bone = CMU_TO_LAFAN1.get(cmu_bone, cmu_bone)
            g_cmu = R.from_quat(global_rot[frame, bones.index(cmu_bone)], scalar_first=True)
            orientation = (world_rot * g_cmu * correction).as_quat(scalar_first=True)
            position = global_pos[frame, bones.index(cmu_bone)] @ _YUP_TO_ZUP.T * scale
            result[lafan_bone] = (position, orientation)

        # Ankle position with toe orientation, as the IK configs expect.
        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftToe"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightToe"][1])

        frames.append(result)

    # Matches load_lafan1_file, which also reports a nominal 1.75 m regardless of
    # the subject: human_height_assumption in the IK configs is calibrated to it.
    human_height = 1.75

    return frames, human_height
