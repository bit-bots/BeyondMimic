import argparse
import pathlib
import csv
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from general_motion_retargeting.utils.cmu_bvh import is_cmu_bvh, load_cmu_bvh_file
from smplx_to_pi_plus import PI_FOOTBALL_WALKREADY
from rich import print
from tqdm import tqdm
import os
import numpy as np

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pi_football", "hightorque_hi"],
        default="unitree_g1",
    )

    parser.add_argument(
        "--save_path",
        required=True,
        help="Path to save the robot motion (CSV).",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
        help="Ignored in headless mode (no viewer to rate-limit). Accepted for command compatibility.",
    )

    parser.add_argument(
        "--keep_wrist",
        action="store_true",
        default=False,
        help="Keep l_wrist and r_wrist joint columns in CSV output. Default behavior is to drop them (bitbots pi_plus has no wrist joints).",
    )

    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir:  # Only create directory if it's not empty
        os.makedirs(save_dir, exist_ok=True)
    qpos_list = []

    # Load SMPLX trajectory. CMU-style skeletons (soccer_kicks) carry extra bones,
    # a ~15.9-units-per-metre scale and different bone axes, so they go through a
    # loader that converts them to LAFAN1 convention first.
    if is_cmu_bvh(args.bvh_file):
        lafan1_data_frames, actual_human_height = load_cmu_bvh_file(args.bvh_file)
    else:
        lafan1_data_frames, actual_human_height = load_lafan1_file(args.bvh_file)

    # Initialize the retargeting system. pi_football needs the walkready seed: the
    # shoulder rolls' ref=∓1.5708 lies outside the bitbots-derived limits, so the
    # default qpos0 seed makes mink raise on the first solve.
    retargeter = GMR(
        src_human="bvh",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        init_qpos=PI_FOOTBALL_WALKREADY if args.robot == "pi_football" else None,
    )

    motion_fps = 30
    print(f"mocap_frame_rate: {motion_fps}")

    # Retarget every frame (no viewer / no rendering -> works headless)
    for i in tqdm(range(len(lafan1_data_frames)), desc="Retargeting"):
        smplx_data = lafan1_data_frames[i]
        qpos = retargeter.retarget(smplx_data)
        qpos_list.append(qpos)

    import pickle
    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    # save from wxyz to xyzw
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    # GMR output order: l_leg(0-5), l_arm(6-9), r_leg(10-15), r_arm(16-19).
    # The wrist joints were commented out of the pi XML in 64bef26, so the model
    # went 22 -> 20 DoF and everything from index 10 on shifted down by two.
    # Target CSV order: l_leg → r_leg → l_arm → r_arm
    reorder_indices = (
        [0, 1, 2, 3, 4, 5]
        + [10, 11, 12, 13, 14, 15]
        + [6, 7, 8, 9]
        + [16, 17, 18, 19]
    )
    dof_pos = dof_pos[:, reorder_indices]
    # Save as CSV
    with open(args.save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # header
        header = ['frame']
        header.extend([f'root pos {i}' for i in ['x', 'y', 'z']])
        header.extend([f'root rot {i}' for i in ['x', 'y', 'z', 'w']])

        # Joint names matching reorder_indices order
        l_arm_names = ['l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow']
        r_arm_names = ['r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow']
        if args.keep_wrist:
            l_arm_names.append('l_wrist')
            r_arm_names.append('r_wrist')
        joint_names = (
            ['l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll']
            + ['r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll']
            + l_arm_names
            + r_arm_names
        )
        header.extend(joint_names)
        writer.writerow(header)

        # write data per frame
        for frame_idx in range(len(qpos_list)):
            row = [frame_idx]
            row.extend(root_pos[frame_idx].tolist())
            row.extend(root_rot[frame_idx].tolist())
            row.extend(dof_pos[frame_idx].tolist())
            writer.writerow(row)
    print(f"Saved to {args.save_path}")
