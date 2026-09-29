import argparse
import pathlib
import time
import csv
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.bvh import load_bvh_file
from general_motion_retargeting.utils.mixed_root import DEFAULT_ROOT_BLEND
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
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01","pi_football","hightorque_hi"],
        default="unitree_g1",
    )
        
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )

    parser.add_argument(
        "--keep_wrist",
        action="store_true",
        default=False,
        help="Keep l_wrist and r_wrist joint columns in CSV output. Default behavior is to drop them (bitbots pi_plus has no wrist joints).",
    )

    parser.add_argument(
        "--root_blend",
        type=float,
        default=DEFAULT_ROOT_BLEND,
        help="MixedRoot orientation = slerp(Hips, Spine2, blend); 0=pelvis, 1=upper spine.",
    )

    args = parser.parse_args()
    

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load the BVH trajectory. load_bvh_file auto-detects the dataset: CMU-style
    # skeletons (soccer_kicks) carry extra bones, a ~15.9-units-per-metre scale and
    # different bone axes, so they go through a loader that converts them to LAFAN1
    # convention first. Either way the frames come back with MixedRoot injected,
    # which the pi_football IK config roots at.
    lafan1_data_frames, actual_human_height = load_bvh_file(
        args.bvh_file, root_blend=args.root_blend
    )


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
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0

    while i < len(lafan1_data_frames):
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        i += 1

        if args.save_path is not None:
            qpos_list.append(qpos)
    
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        
        # Reorder GMR model dof order -> target CSV order (l_leg -> r_leg -> l_arm -> r_arm)
        # by joint NAME, so it stays correct regardless of the model's joint set/order.
        # GMR's pi_football model is this repo's bitbots MJCF, whose joint order differs
        # entirely from the bundled demo model's (it starts with the right arm). The
        # dof_pos column for a joint is its dof index minus 6 (the free base occupies
        # dof 0-5 / qpos 0-6).
        csv_joint_order = (
            ['l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll']
            + ['r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll']
            + ['l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow']
            + ['r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow']
        )
        if args.keep_wrist:
            print("[warn] --keep_wrist ignored: the pi_plus model has no wrist joints")
        reorder_indices = [retargeter.robot_dof_names[f"{j}_joint"] - 6 for j in csv_joint_order]
        dof_pos = dof_pos[:, reorder_indices]
        # 保存为CSV格式
        with open(args.save_path, 'w',newline='') as f:
            writer = csv.writer(f)
            # 写入表头 
            header = ['frame']
            header.extend([f'root pos {i}' for i in ['x', 'y','z']])
            header.extend([f'root rot {i}' for i in ['x','y','z','w']])
            
            # Joint names matching reorder_indices order
            header.extend(csv_joint_order)
            writer.writerow(header)

            # 按帧写入数据
            for frame_idx in range(len(qpos_list)):
                row = [frame_idx]
                row.extend(root_pos[frame_idx].tolist())
                row.extend(root_rot[frame_idx].tolist())
                row.extend(dof_pos[frame_idx].tolist())
                writer.writerow(row)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
