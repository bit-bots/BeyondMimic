"""BeyondMimic Full-Body Motion Tracking Environment Configuration File

This file defines the complete configuration for a humanoid robot motion tracking task, including:
- Scene setup (terrain, lighting, sensors)
- MDP components (observations, actions, rewards, termination conditions)
- Training environment parameters and randomization strategies

Based on the Isaac Lab framework, this uses reinforcement learning to train the robot to track a reference motion
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Predefined configurations
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp

##
# Scene Definition
##

# Robot Speed Perturbation Range (for Domain Randomization)
# Units: Linear velocity (m/s), Angular velocity (rad/s)
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),      # Forward and backward linear velocity
    "y": (-0.5, 0.5),      # Lateral linear velocity
    "z": (-0.2, 0.2),      # Vertical linear velocity
    "roll": (-0.52, 0.52), # Roll angular velocity (approximately 30 degrees per second)
    "pitch": (-0.52, 0.52),# Pitch rate (approx. 30 degrees per second)
    "yaw": (-0.78, 0.78),  # Yaw angular velocity (approximately 45 degrees per second)
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene Configuration for Motion Tracking Tasks

    A complete scene setup that includes terrain, robots, lighting, and sensors.
    """

    # Ground Terrain Configuration
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",          # Paths in the USD scene
        terrain_type="plane",              # Types of Flat Terrain
        collision_group=-1,                # Collision group ID (-1 indicates collision with all groups)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # Friction Combination Mode
            restitution_combine_mode="multiply", # Elastic Recovery Combination Mode
            static_friction=1.0,           # Coefficient of static friction
            dynamic_friction=1.0,          # Coefficient of kinetic friction
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,              # Enable UV Projection
        ),
    )
    # Robot Configuration (to be specified in the specific task)
    robot: ArticulationCfg = MISSING
    
    # Lighting Setup
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),      # Light Source Color (RGB)
            intensity=3000.0               # Light intensity
        ),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.13, 0.13, 0.13),      # Ambient light color
            intensity=1000.0               # Ambient light intensity
        ),
    )
    
    # Contact Force Sensor Configuration
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", # Monitor the contact of all robot components
        history_length=3,                   # Save the last 3 frames of data
        track_air_time=True,                # Track flight time
        force_threshold=10.0,               # Contact force threshold (N)
        debug_vis=True                      # Enable debug visualization
    )


##
# MDP Settings (Markov Decision Process)
##


@configclass
class CommandsCfg:
    """MDP Command Specification Configuration
    
    Define the motion commands the robot needs to follow, including sampling and randomization parameters for the reference motion.
    """

    motion = mdp.MotionCommandCfg(
        asset_name="robot",                    # Name of Target Asset
        resampling_time_range=(1.0e9, 1.0e9), # Resampling time range (s) - The maximum value indicates no resampling
        debug_vis=True,                       # Enable debug visualization
        pose_range={                          # Range of pose randomization
            "x": (-0.05, 0.05),              # X-axis position offset (m)
            "y": (-0.05, 0.05),              # Y-axis position offset (m)
            "z": (-0.01, 0.01),              # Z-axis position offset (m)
            "roll": (-0.1, 0.1),             # Roll angle offset (rad)
            "pitch": (-0.1, 0.1),            # Pitch angle offset (rad)
            "yaw": (-0.2, 0.2),              # Yaw angle offset (rad)
        },
        velocity_range=VELOCITY_RANGE,        # Speed randomization range
        joint_position_range=(-0.2, 0.2),    # Range of joint position randomization (rad)
    )


@configclass
class ActionsCfg:
    """MDP Motion Specification Configuration
    
    Defines the robot's control motion space; joint position control is used here
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",           # Target Robotic Assets
        joint_names=[".*"],          # Control all joints (regular expression)
        use_default_offset=True      # Use the default joint positions as offsets
    )


@configclass
class ObservationsCfg:
    """MDP Observation Specification Configuration
    
    Define the observation spaces for the policy network and the critic network. Policy observations include noise to improve the sim-to-real transfer performance.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy Network Observation Set Configuration
        
        Contains noisy observations used to train robust policies. The order of the observations is preserved.
        """

        # Observation item definitions (keep order)
        command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "motion"}
        )  # Movement Commands
        
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, 
            params={"command_name": "motion"}, 
            noise=Unoise(n_min=-0.25, n_max=0.25)  # Anchor point position noise
        )
        
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, 
            params={"command_name": "motion"}, 
            noise=Unoise(n_min=-0.05, n_max=0.05)  # Anchor-directional noise
        )
        
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
            noise=Unoise(n_min=-0.5, n_max=0.5)    # Base linear velocity noise
        )
        
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            noise=Unoise(n_min=-0.2, n_max=0.2)    # Base angular velocity noise
        )
        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=Unoise(n_min=-0.01, n_max=0.01)  # Joint Position Noise
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            noise=Unoise(n_min=-0.5, n_max=0.5)    # Joint velocity noise
        )
        
        actions = ObsTerm(func=mdp.last_action)  # Previous step

        def __post_init__(self):
            self.enable_corruption = True      # Enable observation of disruptions (noise)
            self.concatenate_terms = True      # Combine all observations into a single vector

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Privileged Observation Set Configuration (Critic Network)
        
        Contains noise-free, precise observations, as well as additional privileged information (such as body position/orientation).
        Used to train the critic network for value function estimation.
        """
        
        command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "motion"}
        )  # Motion command (silent)
        
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, 
            params={"command_name": "motion"}
        )  # Anchor position (noise-free)
        
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, 
            params={"command_name": "motion"}
        )  # Anchor Direction (No Noise)
        
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b, 
            params={"command_name": "motion"}
        )  # Robot Body Position (Privileged Information)
        
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b, 
            params={"command_name": "motion"}
        )  # Robot Body Orientation (Privileged Information)
        
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)     # Base linear velocity (noise-free)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)     # Base angular velocity (noise-free)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)       # Joint position (no noise)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)       # Joint speed (noise-free)
        actions = ObsTerm(func=mdp.last_action)           # Previous step

    # Observation Group Examples
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg() # Critics' Online Observations


@configclass
class EventCfg:
    """Event Configuration
    
    Define domain randomization events to improve the policy's generalization ability and sim-to-real transfer performance.
    This includes randomization events that occur at startup and during training.
    """

    # On startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",                    # Execute at the beginning of each episode
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),    # Static friction
            "dynamic_friction_range": (0.3, 1.2),   # Dynamic friction
            "restitution_range": (0.0, 0.5),        # Restitution coefficient range
            "num_buckets": 64,                      # Number of randomization buckets
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",                              # Execute at startup
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01), # Joint default position randomization range (rad)
            "operation": "add",                       # Addition operation
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",                              # Execute at startup
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {                          # Center of mass randomization range (m)
                "x": (-0.025, 0.025),               # Center of mass offset in the X direction
                "y": (-0.05, 0.05),                 # Center of mass offset in the Y direction
                "z": (-0.05, 0.05)                  # Center of mass offset in the Z direction
            },
        },
    )

    # Interval Event - Traditional Push (for all environments)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",                        # Interval execution mode
        interval_range_s=(1.0, 3.0),            # Execution interval range (s)
        params={"velocity_range": VELOCITY_RANGE}, # Thrust velocity range
    )

@configclass
class RewardsCfg:
    """MDP Reward Configuration
    
    Define the reward function for the motion tracking task, including rewards for position, orientation, and velocity tracking, as well as penalties for behavior.
    Use an exponentially decaying reward function to achieve precise tracking.
    """

    # DeepMimic-style motion tracking reward
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,                           # Reward weight
        params={
            "command_name": "motion", 
            "std": 0.3                        # Standard deviation parameter, controls the rate of reward decay
        },
    )  #  Global anchor position tracking reward
    
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={
            "command_name": "motion", 
            "std": 0.4
        },
    )  # Global Anchor Direction Tracking Reward
    
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion", 
            "std": 0.3
        },
    )  # Relative Body Position Tracking Reward
    
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "command_name": "motion", 
            "std": 0.4
        },
    )  # Relative Body Orientation Tracking Reward
    
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion", 
            "std": 1.0
        },
    )  # Global Body Velocity Tracking Reward
    
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion", 
            "std": 3.14
        },
    )  # Global body angular velocity tracking reward
    # Behavior regularization penalty term
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-1e-1                          # Negative weight indicates a penalty
    )  # L2 penalty on action rate, encouraging smooth actions
    
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,                         # severe punishment
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])
        },
    )  # Joint limit detection to prevent joints from moving beyond their safe range
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,                          # Contact penalty weight
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    # Regular expression: Exclude all “body” elements from the ankles and wrists
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,                 # Contact force threshold (N)
        },
    )  # Avoid contact with the penalty zone; ensure that non-end effectors do not touch the ground

@configclass
class TerminationsCfg:
    """MDP Termination Condition Configuration
    
    Define the conditions under which an episode ends early to avoid dangerous states and invalid training data.
    """

    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True                         # Marked as terminated due to timeout
    )  # Terminated due to timeout
    
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={
            "command_name": "motion", 
            "threshold": 0.25                 # Z-axis position deviation threshold (m)
        },
    )  # Terminate due to excessive anchor point deviation (check Z-axis only)
    
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "command_name": "motion", 
            "threshold": 0.8                  # Directional deviation threshold
        },
    )  # Terminate due to excessive deviation in anchor point direction (to prevent the robot from tipping over)
    
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,                # End-effector position deviation threshold (m)
            "body_names": [                   # End-effectors to monitor
                "left_ankle_roll_link",       # Left ankle
                "right_ankle_roll_link",      # Right ankle
                "left_wrist_yaw_link",        # Left wrist
                "right_wrist_yaw_link",       # Right wrist
            ],
        },
    )  # Terminate if end-effector position deviation is too large

@configclass
class CurriculumCfg:
    """MDP Policy Learning Configuration
    
    Define policy learning strategies for the training process, allowing you to gradually increase the difficulty of the task.
    Includes adaptive training strategies such as force-based policy learning.
    """

    # Force-based Learning Module - Dynamically Adjusting Assistive Force Based on Robot Performance
    pass

##
# Environment Configuration
##

@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Motion Tracking Environment Configuration
    
    A complete environment configuration that integrates all MDP components for training humanoid robots to track reference motions.
    Based on the ManagerBasedRLEnv framework from Isaac Labs.
    """

    # Scene Configuration
    scene: MySceneCfg = MySceneCfg(
        num_envs=4096,                        # Number of parallel environments
        env_spacing=2.5                      # Distance between environments (m)
    )
    
    # Basic MDP Components
    observations: ObservationsCfg = ObservationsCfg()  #  Observation space configuration
    actions: ActionsCfg = ActionsCfg()                  # Action space configuration
    commands: CommandsCfg = CommandsCfg()               # Command configuration
    
    #  MDP Behavior Definitions
    rewards: RewardsCfg = RewardsCfg()                  # Reward function configuration
    terminations: TerminationsCfg = TerminationsCfg()   # Termination condition configuration
    events: EventCfg = EventCfg()                       # Randomized event configuration
    curriculum: CurriculumCfg = CurriculumCfg()         # Curriculum learning configuration

    def __post_init__(self):
        """Post-initialization configuration
        
        Set simulation parameters, rendering settings, and viewer configuration.
        """
        # General settings
        self.decimation = 4                   # Control frequency decimation rate (simulation 50Hz → control 12.5Hz)
        self.episode_length_s = 10.0          # Episode duration (s)
        
        # Simulation settings
        self.sim.dt = 0.005                   # Simulation time step (s) = 200Hz
        self.sim.render_interval = self.decimation  # Render interval
        self.sim.physics_material = self.scene.terrain.physics_material  # Physics material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # Maximum number of GPU rigid body patches
        
        # Viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)    #  Camera position
        self.viewer.origin_type = "asset_root" # Camera origin type
        self.viewer.asset_name = "robot"      # Name of the asset to follow
