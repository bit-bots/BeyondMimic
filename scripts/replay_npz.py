"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage Examples:
    # For HI robot with local file:
    python scripts/replay_npz.py --robot hi --motion_file source/motion/hightorque/hi/npz/hi_dance1_subject2.npz
    
    # For PI Plus robot with local file:
    python scripts/replay_npz.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/pi_plus_dance1_subject2.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--robot", type=str, choices=["hi", "pi_plus"], required=True,
                   help="Robot type: hi (Hi), pi_plus (PI Plus)")
parser.add_argument("--registry_name", type=str, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, help="Local motion NPZ file path")
parser.add_argument("--video", action="store_true", default=False,
                    help="Render the motion to an mp4 file instead of (or in addition to) the live viewer. "
                         "Works headless when combined with --headless --enable_cameras.")
parser.add_argument("--video_path", type=str, default=None,
                    help="Output mp4 path. Defaults to <motion_file>.mp4 next to the npz.")
parser.add_argument("--video_length", type=int, default=0,
                    help="Number of frames to record. 0 (default) records exactly one full motion loop.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# recording needs the rendering pipeline / cameras enabled
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.hi import HI_CFG
from whole_body_tracking.robots.pi_plus import PI_PLUS_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader

# Robot configurations
ROBOT_CONFIGS = {
    "hi": {
        "cfg": HI_CFG,
        "name": "Hi"
    },
    "pi_plus": {
        "cfg": PI_PLUS_CFG,
        "name": "PI Plus"
    }
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation (will be set dynamically based on robot type)
    robot: ArticulationCfg = None

    # camera for video recording (only set when --video is given; see main())
    camera: CameraCfg = None


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Support both local file and WandB registry
    if args_cli.motion_file:
        motion_file = args_cli.motion_file
        print(f"[INFO]: Using local motion file: {motion_file}")
    else:
        if not args_cli.registry_name:
            raise ValueError("Either --motion_file or --registry_name must be provided")
        
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO]: Using WandB motion file: {registry_name}")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # video recording setup
    record_video = args_cli.video
    mp4_writer = None
    frames_to_record = 0
    frames_written = 0
    if record_video:
        import imageio

        video_path = args_cli.video_path
        if video_path is None:
            video_path = os.path.splitext(motion_file)[0] + ".mp4"
        video_dir = os.path.dirname(os.path.abspath(video_path))
        os.makedirs(video_dir, exist_ok=True)
        fps = int(round(1.0 / sim_dt))
        mp4_writer = imageio.get_writer(video_path, fps=fps)
        # default: exactly one full motion loop
        frames_to_record = args_cli.video_length if args_cli.video_length > 0 else int(motion.time_step_total)
        print(f"[INFO]: Recording {frames_to_record} frames @ {fps} fps to {video_path}")

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()

        pos_lookat = root_states[0, :3].cpu().numpy()
        eye = pos_lookat + np.array([2.0, 2.0, 0.5])
        sim.set_camera_view(eye, pos_lookat)
        if record_video:
            # keep the recording camera following the robot, same view as the live camera
            cam = scene["camera"]
            cam.set_world_poses_from_view(
                torch.tensor([eye], dtype=torch.float32, device=sim.device),
                torch.tensor([pos_lookat], dtype=torch.float32, device=sim.device),
            )

        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        if record_video:
            rgb = scene["camera"].data.output["rgb"][0, ..., :3]
            mp4_writer.append_data(rgb.detach().cpu().numpy().astype(np.uint8))
            frames_written += 1
            if frames_written >= frames_to_record:
                mp4_writer.close()
                print(f"[INFO]: Saved video to {video_path} ({frames_written} frames)")
                break


def main():
    # Get robot configuration
    robot_config = ROBOT_CONFIGS[args_cli.robot]
    print(f"[INFO]: Using robot configuration: {args_cli.robot} ({robot_config['name']})")
    
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    # Design scene with robot-specific configuration
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_config["cfg"].replace(prim_path="{ENV_REGEX_NS}/Robot")
    if args_cli.video:
        # standalone follow-camera; its pose is driven each step in run_simulator()
        scene_cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/record_cam",
            update_period=0.0,
            width=1280,
            height=720,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, clipping_range=(0.01, 1.0e5)
            ),
        )
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print(f"[INFO]: Setup complete for {robot_config['name']} robot...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
