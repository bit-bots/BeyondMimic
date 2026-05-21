import argparse
import pathlib
import time
import csv
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
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

    args = parser.parse_args()
    

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_lafan1_file(args.bvh_file)
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human="bvh",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
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
        
        # GMR output order: l_leg(0-5), l_arm(6-10, last=l_wrist), r_leg(11-16), r_arm(17-21, last=r_wrist)
        # Target CSV order: l_leg → r_leg → l_arm → r_arm
        l_arm_src = [6, 7, 8, 9, 10] if args.keep_wrist else [6, 7, 8, 9]
        r_arm_src = [17, 18, 19, 20, 21] if args.keep_wrist else [17, 18, 19, 20]
        reorder_indices = (
            [0, 1, 2, 3, 4, 5]
            + [11, 12, 13, 14, 15, 16]
            + l_arm_src
            + r_arm_src
        )
        dof_pos = dof_pos[:, reorder_indices]
        # 保存为CSV格式
        with open(args.save_path, 'w',newline='') as f:
            writer = csv.writer(f)
            # 写入表头 
            header = ['frame']
            header.extend([f'root pos {i}' for i in ['x', 'y','z']])
            header.extend([f'root rot {i}' for i in ['x','y','z','w']])
            
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
       
