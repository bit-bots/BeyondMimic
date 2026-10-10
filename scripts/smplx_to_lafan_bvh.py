"""Convert GVHMR output (hmr4d_results.pt) or an AMASS-style SMPL-X .npz into a .bvh
in LAFAN1 convention, so GVHMR motions go through exactly the same retargeting as
LAFAN1 clips (scripts/bvh_to_robot.py + ik_configs/bvh_to_pi_football.json).

    # straight from GVHMR (y-up world -> rotated to z-up with the defaults below)
    pixi run -e gmr python scripts/smplx_to_lafan_bvh.py \
        GVHMR/outputs/demo/tennis/hmr4d_results.pt --fps 60 -o RetargetData/gvhmr/bvh/tennis.bvh
    # or from an already converted npz (scripts/gvhmr_to_smplx.py output)
    pixi run -e gmr python scripts/smplx_to_lafan_bvh.py \
        RetargetData/gvhmr/smplx/tennis.npz -o RetargetData/gvhmr/bvh/tennis.bvh
    pixi run -e gmr python scripts/bvh_to_robot.py --robot pi_football \
        --bvh_file RetargetData/gvhmr/bvh/tennis.bvh --save_path ...csv

What "LAFAN1 convention" means here, and how each part is matched
-----------------------------------------------------------------
  * topology  - SMPL's 22 body joints map 1:1 onto LAFAN1's 22 bones (SMPL_TO_LAFAN1);
                the file has LAFAN1's hierarchy, names and channel layout (root 6
                channels, every other joint 3, Euler ZYX), Y-up, centimetres, 30 fps.
  * scale     - the subject is rescaled so its T-pose head height equals LAFAN1's
                (1.5904 m), like cmu_bvh.py does. load_lafan1_file reports a fixed
                1.75 m whatever the subject, and the IK scale table is tuned to that.
  * bone axes - LAFAN1's local frames differ from SMPL's. SMPL's zero pose is a
                T-pose with every global rotation = identity, and its canonical frame
                (Y-up, facing +Z, left = +X) is LAFAN1's raw BVH frame, so the
                per-bone correction is simply C_b = posture_b * G_lafan_b(T-pose).
                posture_b is the minimal rotation carrying LAFAN1's bone axis (+X)
                onto the subject's actual rest bone direction -- the same split as in
                cmu_bvh.py, so the subject's own rest geometry (e.g. GVHMR arms below
                horizontal) is not mistaken for a frame-convention difference. Root
                and spine keep posture = identity (pure convention matching).

For a .pt the gvhmr_to_smplx.py conversion runs in memory (same function), with
--rot_axis x --rot_deg 90 as default because GVHMR's world is y-up. --fps must be the
SOURCE video's fps (phone clips are often 60): GVHMR keeps every frame, and the
resampling to 30 fps here relies on it.

Joint positions are reproduced exactly: offsets are the subject's own rest bone
vectors expressed in the corrected parent frames, so BVH forward kinematics gives
back the (scaled) SMPL-X joint positions. --verify (on by default) checks that by
reading the written file with GMR's load_bvh_file.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import smplx
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.utils.cmu_bvh import LAFAN1_TPOSE_HEAD_HEIGHT_M, _min_rotation

HERE = pathlib.Path(__file__).parent

# LAFAN1 bone order/hierarchy, exactly as in the dataset's files.
LAFAN1_BONES = [
    "Hips",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
]
LAFAN1_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

SMPL_TO_LAFAN1 = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg", "left_knee": "LeftLeg", "left_ankle": "LeftFoot", "left_foot": "LeftToe",
    "right_hip": "RightUpLeg", "right_knee": "RightLeg", "right_ankle": "RightFoot", "right_foot": "RightToe",
    "spine1": "Spine", "spine2": "Spine1", "spine3": "Spine2", "neck": "Neck", "head": "Head",
    "left_collar": "LeftShoulder", "left_shoulder": "LeftArm", "left_elbow": "LeftForeArm", "left_wrist": "LeftHand",
    "right_collar": "RightShoulder", "right_shoulder": "RightArm", "right_elbow": "RightForeArm",
    "right_wrist": "RightHand",
}
LAFAN1_TO_SMPL = {v: k for k, v in SMPL_TO_LAFAN1.items()}

# Global rotations (raw BVH space, wxyz) of the LAFAN1 T-pose for all 22 bones: the
# mean frame 0 of the 40 LAFAN1 clips within 12 deg of a textbook T-pose -- the same
# selection as cmu_bvh.LAFAN1_TPOSE_QUAT, whose 15 bones this reproduces to 0.00 deg.
LAFAN1_TPOSE_QUAT = {
    "Hips": (0.51656293, 0.48026828, 0.51092124, 0.49139048),
    "LeftUpLeg": (-0.51883616, -0.50691629, 0.48092433, 0.49250046),
    "LeftLeg": (-0.47225235, -0.53279265, 0.42476168, 0.55918443),
    "LeftFoot": (0.73213104, 0.11869716, -0.65840819, -0.12803824),
    "LeftToe": (-0.74318690, 0.00592984, 0.66897561, -0.01047392),
    "RightUpLeg": (-0.49996445, -0.48963527, 0.51154895, 0.49860859),
    "RightLeg": (-0.40360111, -0.57620069, 0.47321477, 0.53025153),
    "RightFoot": (-0.62539965, -0.13845710, 0.76001391, 0.10992622),
    "RightToe": (-0.63491044, 0.00542716, 0.77252098, -0.00840325),
    "Spine": (0.48906396, 0.50763596, 0.48397600, 0.51854547),
    "Spine1": (0.47781797, 0.51845972, 0.47287643, 0.52846702),
    "Spine2": (0.46783835, 0.52775842, 0.46302757, 0.53684616),
    "Neck": (-0.44218925, -0.56123682, -0.41999353, -0.55954207),
    "Head": (-0.50180025, -0.50061568, -0.47579175, -0.52077122),
    "LeftShoulder": (-0.73733706, 0.67310799, -0.02630601, -0.05067224),
    "LeftArm": (0.68642255, -0.72710500, -0.00953441, 0.00717604),
    "LeftForeArm": (0.67293929, -0.72220607, -0.15925125, 0.01449608),
    "LeftHand": (0.73965058, -0.66482407, -0.06230890, 0.08392608),
    "RightShoulder": (-0.01649611, -0.03152747, -0.74055013, 0.67105842),
    "RightArm": (0.02672458, 0.00699290, 0.68457874, -0.72841530),
    "RightForeArm": (-0.16519840, 0.01125778, 0.66770784, -0.72577475),
    "RightHand": (0.07498428, -0.09259737, -0.71467217, 0.68923638),
}

# Bones whose posture is matched to the subject's rest bone direction (bone -> child),
# the same set cmu_bvh.py uses. Root/spine/neck/head/collars and the leaves keep
# posture = identity.
BONE_CHILD = {
    "LeftUpLeg": "LeftLeg", "RightUpLeg": "RightLeg",
    "LeftLeg": "LeftFoot", "RightLeg": "RightFoot",
    "LeftFoot": "LeftToe", "RightFoot": "RightToe",
    "LeftArm": "LeftForeArm", "RightArm": "RightForeArm",
    "LeftForeArm": "LeftHand", "RightForeArm": "RightHand",
}

# Z-up (MuJoCo / AMASS world) <-> Y-up (BVH): p_zup = YUP_TO_ZUP @ p_bvh, same matrix
# as load_lafan1_file.
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

TARGET_FPS = 30


def rest_joints(smplx_data, body_model):
    """Subject's zero-pose (T-pose) joint positions in SMPL's canonical frame, (22, 3)."""
    with torch.no_grad():
        out = body_model(betas=torch.tensor(smplx_data["betas"]).float().view(1, -1))
    return out.joints[0, :22].numpy().astype(np.float64)


def corrections(J_rest_by_bone):
    """C_b per LAFAN1 bone: SMPL global rotation @ C_b = LAFAN1-convention global rotation."""
    C = {}
    for bone in LAFAN1_BONES:
        g_lafan = R.from_quat(LAFAN1_TPOSE_QUAT[bone], scalar_first=True)
        child = BONE_CHILD.get(bone)
        if child is None:
            posture = R.identity()
        else:
            ref_dir = g_lafan.apply([1.0, 0.0, 0.0])  # LAFAN1's local +X is the bone axis
            posture = _min_rotation(ref_dir, J_rest_by_bone[child] - J_rest_by_bone[bone])
        C[bone] = posture * g_lafan
    return C


def write_bvh(path, offsets_cm, root_pos_cm, local_euler_deg, frame_time):
    """offsets_cm (22, 3); root_pos_cm (N, 3); local_euler_deg (N, 22, 3) as Z, Y, X."""
    children = {i: [] for i in range(len(LAFAN1_BONES))}
    for i, p in enumerate(LAFAN1_PARENTS):
        if p >= 0:
            children[p].append(i)

    lines = ["HIERARCHY"]

    def emit(i, depth):
        ind = "\t" * depth
        o = offsets_cm[i]
        if LAFAN1_PARENTS[i] < 0:
            lines.append(f"ROOT {LAFAN1_BONES[i]}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
            lines.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation ")
        else:
            lines.append(f"{ind}JOINT {LAFAN1_BONES[i]}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
            lines.append(f"{ind}\tCHANNELS 3 Zrotation Yrotation Xrotation")
        if children[i]:
            for c in children[i]:
                emit(c, depth + 1)
        else:
            lines.extend([f"{ind}\tEnd Site", f"{ind}\t{{", f"{ind}\t\tOFFSET 0.000000 0.000000 0.000000", f"{ind}\t}}"])
        lines.append(f"{ind}}}")

    emit(0, 0)
    n = root_pos_cm.shape[0]
    lines += ["MOTION", f"Frames: {n}", f"Frame Time: {frame_time:.6f}"]
    for f in range(n):
        vals = list(root_pos_cm[f]) + list(local_euler_deg[f].reshape(-1))
        lines.append(" ".join(f"{v:.6f}" for v in vals))
    path.write_text("\n".join(lines) + "\n")


def evaluate_smplx(smplx_data, body_model_path):
    """load_smplx_file for an in-memory dict: (body_model, smplx_output), same model and inputs."""
    body_model = smplx.create(body_model_path, "smplx", gender=str(smplx_data["gender"]), use_pca=False)
    n = smplx_data["pose_body"].shape[0]
    zeros = lambda k: torch.zeros(n, k).float()
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=zeros(45), right_hand_pose=zeros(45),
        jaw_pose=zeros(3), leye_pose=zeros(3), reye_pose=zeros(3),
        return_full_pose=True,
    )
    return body_model, smplx_output


def convert(smplx_file, out_file, gvhmr_args=None):
    """smplx_file: .npz, or GVHMR .pt together with gvhmr_args for gvhmr_to_smplx.load_gvhmr."""
    body_model_path = HERE / ".." / "assets" / "body_models"
    if str(smplx_file).endswith(".pt"):
        from gvhmr_to_smplx import load_gvhmr
        smplx_data = load_gvhmr(smplx_file, **(gvhmr_args or {}))
        body_model, smplx_output = evaluate_smplx(smplx_data, body_model_path)
    else:
        smplx_data, body_model, smplx_output, _ = load_smplx_file(smplx_file, body_model_path)
    frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=TARGET_FPS)

    J = rest_joints(smplx_data, body_model)
    from smplx.joint_names import JOINT_NAMES
    J_by_bone = {SMPL_TO_LAFAN1[JOINT_NAMES[i]]: J[i] for i in range(22)}
    head_height = J_by_bone["Head"][1] - J[:, 1].min()  # canonical frame is Y-up
    scale = LAFAN1_TPOSE_HEAD_HEIGHT_M / head_height
    C = corrections(J_by_bone)

    # Offsets: rest bone vector in the corrected parent frame. At the T-pose the parent's
    # global rotation is C_parent, so C_parent^-1 maps the canonical vector into it.
    offsets_cm = np.zeros((22, 3))
    for i, bone in enumerate(LAFAN1_BONES):
        p = LAFAN1_PARENTS[i]
        if p >= 0:
            parent = LAFAN1_BONES[p]
            offsets_cm[i] = C[parent].inv().apply(J_by_bone[bone] - J_by_bone[parent]) * scale * 100

    zup_to_yup = R.from_matrix(YUP_TO_ZUP.T)
    n = len(frames)
    root_pos_cm = np.zeros((n, 3))
    local_euler = np.zeros((n, 22, 3))
    for f, fr in enumerate(frames):
        G = [zup_to_yup * R.from_quat(fr[LAFAN1_TO_SMPL[b]][1], scalar_first=True) * C[b] for b in LAFAN1_BONES]
        root_pos_cm[f] = YUP_TO_ZUP.T @ np.asarray(fr["pelvis"][0]) * scale * 100
        for i, p in enumerate(LAFAN1_PARENTS):
            local = G[i] if p < 0 else G[p].inv() * G[i]
            # read_bvh composes Z, then Y, then X about the moving axes = scipy intrinsic "ZYX".
            local_euler[f, i] = local.as_euler("ZYX", degrees=True)
    root_offset = root_pos_cm[0].copy()
    offsets_cm[0] = root_offset  # cosmetic: the root's position channels override it

    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_bvh(out_file, offsets_cm, root_pos_cm, local_euler, 1.0 / fps)
    print(f"[smplx_to_lafan_bvh] {smplx_file} -> {out_file}: {n} frames @ {fps:.1f} fps, "
          f"subject head height {head_height:.3f} m -> scale {scale:.3f}")
    return frames, C, scale


def verify(out_file, frames, C, scale, tol_mm=1.0, tol_deg=0.1):
    """Read the file back through the BVH retarget loader and compare against SMPL-X."""
    from general_motion_retargeting.utils.bvh import load_bvh_file
    from general_motion_retargeting.utils.cmu_bvh import is_cmu_bvh

    if is_cmu_bvh(str(out_file)):
        sys.exit("verify: load_bvh_file would treat this file as CMU")
    loaded, _ = load_bvh_file(str(out_file))
    if len(loaded) != len(frames):
        sys.exit(f"verify: frame count {len(loaded)} != {len(frames)}")
    pos_err, rot_err = 0.0, 0.0
    for fl, fs in zip(loaded, frames):
        for bone in LAFAN1_BONES:
            ps, qs = fs[LAFAN1_TO_SMPL[bone]]
            pos_err = max(pos_err, np.linalg.norm(fl[bone][0] - np.asarray(ps) * scale))
            want = R.from_quat(qs, scalar_first=True) * C[bone]
            got = R.from_quat(fl[bone][1], scalar_first=True)
            rot_err = max(rot_err, np.degrees((want.inv() * got).magnitude()))
    ok = pos_err * 1000 < tol_mm and rot_err < tol_deg
    print(f"[verify] {'PASS' if ok else 'FAIL'}: max joint position error {pos_err * 1000:.3f} mm, "
          f"max orientation error {rot_err:.4f} deg over {len(loaded)} frames x 22 bones")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smplx_file", help="GVHMR hmr4d_results.pt, or AMASS-style SMPL-X .npz")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .bvh (default: input with .bvh suffix; for hmr4d_results.pt "
                             "<clip dir>/<clip dir name>.bvh)")
    parser.add_argument("--no_verify", action="store_true", help="Skip reading the file back for a check.")
    g = parser.add_argument_group(".pt input only (same meaning as in gvhmr_to_smplx.py)")
    g.add_argument("--frame", choices=["global", "incam"], default="global")
    g.add_argument("--fps", type=float, default=30.0, help="SOURCE video fps (e.g. 60 for phone clips).")
    g.add_argument("--gender", default="neutral")
    g.add_argument("--rot_axis", choices=["x", "y", "z"], default="x")
    g.add_argument("--rot_deg", type=float, default=90.0, help="World up-alignment; 90 about x = GVHMR y-up -> z-up.")
    args = parser.parse_args()

    src = pathlib.Path(args.smplx_file)
    if args.output:
        out = pathlib.Path(args.output)
    elif src.suffix == ".pt":
        out = src.parent / f"{src.parent.name}.bvh"
    else:
        out = src.with_suffix(".bvh")
    gvhmr_args = dict(frame=args.frame, fps=args.fps, gender=args.gender,
                      rot_axis=args.rot_axis, rot_deg=args.rot_deg)
    frames, C, scale = convert(src, out, gvhmr_args)
    if not args.no_verify:
        verify(out, frames, C, scale)
