"""Unified sim2sim script supporting multiple robot configurations.

Usage Examples:
    # For HI robot:
    python sim2sim.py --robot hi --motion_file source/motion/hightorque/hi/npz/dance1_subject2.npz \
    --xml_path /path/to/hi.xml --policy_path /path/to/hi_policy.onnx --save_json
    
    # For PI Plus robot:
    python sim2sim.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/dance1_subject2.npz \
    --xml_path /path/to/pi_plus.xml --policy_path /path/to/pi_plus_policy.onnx --save_json
"""

import argparse
import contextlib
import json
import os
import sys
import time

# Offscreen video recording (--video) renders headlessly and needs a non-GLFW GL
# backend. MUJOCO_GL is read once, at `import mujoco` below, so it must be selected
# here BEFORE the import — setting it later (e.g. in main()) is too late and MuJoCo
# falls back to GLFW, which fails on a headless host (no DISPLAY). Default to EGL
# (GPU); override with MUJOCO_GL=osmesa for CPU-only / no-GPU remote hosts.
if "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime
import torch
import yaml
from scipy.spatial.transform import Rotation as R

# Simulation parameters
simulation_duration = 300.0
simulation_dt = 0.002
control_decimation = 10

# Robot configurations
ROBOT_CONFIGS = {
    
    "hi": {
        "num_actions": 23,
        "num_obs": 124,
        "reference_body": "base_link",
        "default_xml": None,  # Must be provided
        "joint_names": [
            "l_hip_pitch_joint",
            "l_hip_roll_joint",
            "l_hip_thigh_joint",
            "l_hip_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_hip_thigh_joint",
            "r_hip_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "waist_yaw_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "l_wrist_joint",
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "r_wrist_joint",
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 46,
            "motion_ref_ori_b": 6,
            "base_ang_vel": 3,
            "joint_pos": 23,
            "joint_vel": 23,
            "actions": 23
        }
    },
    "pi_plus": {
        "num_actions": 20,
        "num_obs": 115,  # FULL variant (Tracking-Flat-PI-Plus-v0): incl. motion_anchor_pos_b + base_lin_vel
        "reference_body": "base_link",
        "default_xml": None,  # Must be provided
        # bitbots pi_plus: 20 DoF (no wrists -> fixed, no head -> fixed).
        # Order MUST match the MJCF qpos order of
        # assets/hightorque/pi_plus_bitbots/mjcf/pi_plus_22dof.xml: R-arm, L-arm, R-leg, L-leg.
        # sim2sim indexes d.qpos[7:]/d.ctrl via this list, so the order is load-bearing.
        "joint_names": [
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_thigh_joint",
            "r_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "l_hip_pitch_joint",
            "l_hip_roll_joint",
            "l_thigh_joint",
            "l_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
        ],
        "motion_body_index": 0,
        "observation_structure": {
            "command": 40,
            "motion_anchor_pos_b": 3,
            "motion_anchor_ori_b": 6,
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "joint_pos": 20,
            "joint_vel": 20,
            "actions": 20
        }
    }
}


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[0, 1, 2, 3]].astype(np.double)
    
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)

    return (qpos, dq, quat, v, omega, gvec, state_tau)


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v (NumPy version)."""
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    
    if q_vec.ndim == 2:
        dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
        c = q_vec * dot_product * 2.0
    else:
        dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
        c = q_vec * dot_product * 2.0
    
    return a - b + c


def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):
    """Calculate relative transformation from frame A to frame B (MuJoCo version)."""
    rotm_a = np.zeros(9)
    mujoco.mju_quat2Mat(rotm_a, quat_a)
    rotm_a = rotm_a.reshape(3, 3)
    
    rel_pos = rotm_a.T @ (pos_b - pos_a)
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), quat_b)
    rel_quat = rel_quat / np.linalg.norm(rel_quat)
    
    return rel_pos, rel_quat


def quaternion_conjugate(q):
    """Quaternion conjugate: [w, x, y, z] -> [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """Quaternion multiplication: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together."""
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion."""
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)


def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Computes the inverse of a quaternion."""
    return quat_conjugate_np(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def create_observation_hi_pi(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions, motion_anchor_pos_b=None, base_lin_vel=None):
    """Create observation for HI and PI Plus robots.

    Layout follows the policy's `observation_names`:
        command, [motion_anchor_pos_b], motion_anchor_ori_b, [base_lin_vel],
        base_ang_vel, joint_pos, joint_vel, actions
    The bracketed terms are only written when provided. They belong to the FULL
    obs variant (e.g. Tracking-Flat-PI-Plus-v0, 115). Pass None for the deploy/Wo
    variant (109) and for the `hi` config, preserving the original layout.
    """
    cmd_size = len(motioninput)
    obs[offset:offset + cmd_size] = motioninput
    offset += cmd_size
    if motion_anchor_pos_b is not None:
        obs[offset:offset + 3] = motion_anchor_pos_b
        offset += 3
    obs[offset:offset + 6] = motion_ref_ori_b
    offset += 6
    if base_lin_vel is not None:
        obs[offset:offset + 3] = base_lin_vel
        offset += 3
    obs[offset:offset + 3] = omega
    offset += 3
    obs[offset:offset + num_actions] = qpos_seq - joint_pos_array_seq
    offset += num_actions
    obs[offset:offset + num_actions] = qvel_seq
    offset += num_actions
    obs[offset:offset + num_actions] = action_buffer
    return obs


def run_simulation(robot_type: str, motion_file: str, xml_path: str, policy_path: str, save_json: bool = False, loop: bool = False,
                   headless: bool = False, video: bool = False, video_length: int = 0):
    """Run the sim2sim simulation."""
    config = ROBOT_CONFIGS[robot_type]
    print(f"[INFO]: Using robot configuration: {robot_type}")
    print(f"[INFO]: Actions: {config['num_actions']}, Observations: {config['num_obs']}")
    
    # Load motion data
    motion = np.load(motion_file)
    motionpos = motion["body_pos_w"]
    motionquat = motion["body_quat_w"]
    motioninputpos = motion["joint_pos"]
    motioninputvel = motion["joint_vel"]
    # number of frames available across all sequences
    num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0], motionpos.shape[0], motionquat.shape[0])
    # safe index helper (supports looping)
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1
    
    # Save motion data to JSON if requested
    if save_json:
        motion_dict = {
            "body_pos_w": motionpos.tolist(),
            "body_quat_w": motionquat.tolist(),
            "joint_pos": motioninputpos.tolist(),
            "joint_vel": motioninputvel.tolist()
        }
        # Convert npz path to json path: npz/file.npz -> json/file.json
        motion_dir = os.path.dirname(motion_file)
        motion_basename = os.path.basename(motion_file)
        
        # Replace 'npz' directory with 'json' directory
        if motion_dir.endswith('/npz') or motion_dir.endswith('\\npz'):
            json_dir = motion_dir[:-3] + 'json'  # Replace last 3 characters 'npz' with 'json'
        else:
            json_dir = motion_dir  # If not in npz directory, use same directory
            
        # Create json directory if it doesn't exist
        os.makedirs(json_dir, exist_ok=True)
        
        # Create json filename
        json_basename = motion_basename.replace('.npz', '.json')
        json_filename = os.path.join(json_dir, json_basename)
        with open(json_filename, 'w') as f:
            json.dump(motion_dict, f, indent=2)
        print(f"[INFO]: Motion data saved to: {json_filename}")
    
    # Load ONNX model and extract metadata
    model = onnx.load(policy_path)
    joint_seq = None
    joint_pos_array_seq = None
    stiffness_array_seq = None
    damping_array_seq = None
    action_scale = None
    
    for prop in model.metadata_props:
        if prop.key == "joint_names":
            joint_seq = prop.value.split(",")
        elif prop.key == "default_joint_pos":   
            joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_stiffness":
            stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "joint_damping":
            damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
        elif prop.key == "action_scale":
            action_scale = np.array([float(x) for x in prop.value.split(",")])
        print(f"{prop.key}: {prop.value}")
    
    # Remap to XML joint order
    joint_xml = config["joint_names"]
    joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    
    print("stiffness_array", stiffness_array)
    print("damping_array", damping_array)
    print("action_scale", action_scale)
    
    # Initialize variables
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # Load policy
    policy = onnxruntime.InferenceSession(policy_path)
    
    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0 
    motioninput = np.concatenate((motioninputpos[frame_idx(timestep), :], motioninputvel[frame_idx(timestep), :]), axis=0)
    
    motion_body_idx = config["motion_body_index"]
    motionposcurrent = motionpos[frame_idx(timestep), motion_body_idx, :]
    motionquatcurrent = motionquat[frame_idx(timestep), motion_body_idx, :]
    
    # Spawn the robot at the motion's first frame instead of the default standing
    # pose. Without this the robot always starts at z=0.385 with identity orientation,
    # so motions that do not start standing (e.g. supine get-up) are impossible to track.
    # The base pose comes from the anchor body (base_link); the motion joint_pos is stored
    # in the policy (joint_seq) order, so it is remapped to the MJCF qpos order.
    start_idx = frame_idx(0)
    if robot_type == "hi":
        d.qpos[2] = 0.68
        target_dof_pos = joint_pos_array.copy()
    else:
        d.qpos[0:3] = motionpos[start_idx, motion_body_idx, :]
        d.qpos[3:7] = motionquat[start_idx, motion_body_idx, :]
        start_joint_seq = motioninputpos[start_idx]
        d.qpos[7:7 + num_actions] = np.array(
            [start_joint_seq[joint_seq.index(j)] for j in joint_xml]
        )
        mujoco.mj_forward(m, d)
        target_dof_pos = d.qpos[7:7 + num_actions].copy()
    
    # Set reference body
    body_name = config["reference_body"]
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")

    # --- optional headless video recording ------------------------------------
    # Mirrors the convention in play.py / replay_npz.py: the mp4 lands at
    #   <run_dir>/videos/sim2sim/<model_name>_sim2sim.mp4
    # where <run_dir> is the training run (parent of the policy's exported/ dir) and
    # <model_name> is the onnx basename (model_4000.onnx -> model_4000_sim2sim.mp4).
    record_video = video
    headless = headless or video  # video is always rendered offscreen / headless
    renderer = None
    mp4_writer = None
    rec_cam = None
    video_path = None
    frames_to_record = 0
    frames_written = 0
    if record_video:
        import imageio

        policy_dir = os.path.dirname(os.path.abspath(policy_path))
        run_dir = os.path.dirname(policy_dir) if os.path.basename(policy_dir) == "exported" else policy_dir
        video_dir = os.path.join(run_dir, "videos", "sim2sim")
        os.makedirs(video_dir, exist_ok=True)
        model_name = os.path.splitext(os.path.basename(policy_path))[0]
        video_path = os.path.join(video_dir, f"{model_name}_sim2sim.mp4")

        # ensure the model's offscreen framebuffer is large enough for the requested size
        width, height = 1280, 720
        m.vis.global_.offwidth = max(width, m.vis.global_.offwidth)
        m.vis.global_.offheight = max(height, m.vis.global_.offheight)
        renderer = mujoco.Renderer(m, height=height, width=width)
        # follow camera tracking the robot base (works for standing and ground-start motions)
        rec_cam = mujoco.MjvCamera()
        rec_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        rec_cam.trackbodyid = body_id
        rec_cam.distance = 2.5
        rec_cam.azimuth = 90.0
        rec_cam.elevation = -20.0

        fps = int(round(1.0 / (simulation_dt * control_decimation)))  # one frame per control step
        mp4_writer = imageio.get_writer(video_path, fps=fps)
        # default (0): record exactly one full motion loop
        frames_to_record = video_length if video_length > 0 else num_frames
        print(f"[INFO]: Recording {frames_to_record} frames @ {fps} fps to {video_path}", flush=True)

    # In headless mode there is no interactive viewer; step/render as fast as possible.
    viewer_cm = contextlib.nullcontext() if headless else mujoco.viewer.launch_passive(m, d)
    with viewer_cm as viewer:
        start = time.time()
        while (viewer.is_running() if viewer is not None else True) and time.time() - start < simulation_duration:
            step_start = time.time()

            mujoco.mj_step(m, d)
            qpos, dq, quat, v, omega, gvec, state_tau = get_obs(d)
            tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)

            d.ctrl[:] = tau
            counter += 1
            
            if counter % control_decimation == 0:
                # Update motion data
                idx = frame_idx(timestep)
                motioninput = np.concatenate((motioninputpos[idx, :], motioninputvel[idx, :]), axis=0)
                motionposcurrent = motionpos[idx, motion_body_idx, :]
                motionquatcurrent = motionquat[idx, motion_body_idx, :]

                # Create observations based on robot type
                offset = 0
                if robot_type in ["hi", "pi_plus"]:
                    # HI and PI Plus observation creation
                    robot_quat_w = torch.from_numpy(quat).unsqueeze(0)
                    q01 = quat
                    q02 = motionquatcurrent
                    q10 = quat_inv_np(q01)
                    if q02 is not None:
                        q12 = quat_mul_np(q10, q02)
                    else:
                        q12 = q10
                    mat = matrix_from_quat(torch.from_numpy(q12))
                    motion_ref_ori_b = mat[..., :2].reshape(6)

                    qpos_xml = d.qpos[7:7 + num_actions]
                    qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                    qvel_xml = d.qvel[6:6 + num_actions]
                    qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])

                    # Privileged terms for the FULL obs variant. Config-driven so the
                    # deploy/Wo layout and the `hi` robot keep their original 6-group obs.
                    obs_struct = config["observation_structure"]
                    motion_anchor_pos_b = None
                    if "motion_anchor_pos_b" in obs_struct:
                        # motion anchor pos expressed in the robot anchor (base_link) frame:
                        # R_robot^T (motion_anchor_pos_w - robot_anchor_pos_w)
                        motion_anchor_pos_b = quat_rotate_inverse_np(quat, motionposcurrent - d.qpos[:3])
                    base_lin_vel = v if "base_lin_vel" in obs_struct else None

                    obs = create_observation_hi_pi(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions, motion_anchor_pos_b=motion_anchor_pos_b, base_lin_vel=base_lin_vel)
                
                # Run policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy.run(['actions'], {
                    'obs': obs_tensor.numpy(),
                    'time_step': np.array([frame_idx(timestep)], dtype=np.float32).reshape(1, 1)
                })[0]
                
                action = np.asarray(action).reshape(-1)
                action_buffer = action.copy()
                target_dof_pos = action * action_scale + joint_pos_array_seq
                target_dof_pos = target_dof_pos.reshape(-1,)
                target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
                
                # advance time step; if not looping and超过序列则保持在末帧
                if loop or timestep + 1 < num_frames:
                    timestep += 1

                # one rendered frame per control step (~50 fps) when recording
                if record_video:
                    renderer.update_scene(d, camera=rec_cam)
                    mp4_writer.append_data(renderer.render())
                    frames_written += 1
                    if frames_written >= frames_to_record:
                        break

            if viewer is not None:
                viewer.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    if record_video:
        mp4_writer.close()
        print(f"[INFO]: Saved video to {video_path} ({frames_written} frames)", flush=True)
        # offscreen GL contexts can keep the process alive; the mp4 is flushed, so exit hard.
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Unified sim2sim script for multiple robots.")
    parser.add_argument("--robot", type=str, choices=["hi", "pi_plus"], required=True,
                        help="Robot type:  hi (Hi), pi_plus (PI Plus)")
    parser.add_argument("--motion_file", type=str, required=True, 
                        help="Path to the motion NPZ file")
    parser.add_argument("--xml_path", type=str, required=True,
                        help="Path to the robot XML file")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to the ONNX policy file")
    parser.add_argument("--save_json", action="store_true",
                        help="Save motion data to JSON file")
    parser.add_argument("--loop", action="store_true",
                        help="Loop motion/policy when reaching the end of sequence")
    parser.add_argument("--headless", action="store_true",
                        help="Run without the interactive viewer (no window). Implied by --video.")
    parser.add_argument("--video", action="store_true",
                        help="Render the run to an mp4 headlessly, saved to "
                             "<run_dir>/videos/sim2sim/<model>_sim2sim.mp4. Remote/headless friendly.")
    parser.add_argument("--video_length", type=int, default=0,
                        help="Frames (control steps) to record. 0 (default) = one full motion loop.")

    args = parser.parse_args()

    # All parameters are now required, so no additional validation needed
    
    print(f"[INFO]: Robot: {args.robot}")
    print(f"[INFO]: Motion file: {args.motion_file}")
    print(f"[INFO]: XML path: {args.xml_path}")
    print(f"[INFO]: Policy path: {args.policy_path}")
    
    run_simulation(args.robot, args.motion_file, args.xml_path, args.policy_path, args.save_json, args.loop,
                   headless=args.headless, video=args.video, video_length=args.video_length)


if __name__ == "__main__":
    main()