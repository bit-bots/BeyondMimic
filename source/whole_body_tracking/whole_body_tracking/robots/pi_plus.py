import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE_4438 = 0.01317  # bitbots amature
ARMATURE_5047 = 0.01316  # bitbots amature

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_4438 = 30
STIFFNESS_5047 = 80

DAMPING_4438 = 0.6
DAMPING_5047 = 1.1


PI_PLUS_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,  # keep head_*/wrist_* as separate bodies for per-link contact penalties
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/hightorque/pi_plus_bitbots/pi_plus_22dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3173),  # walkready pelvis height (miro_rad_walkready.npz frame 0)
        joint_pos={
            # Pitch joints have opposite signs L/R because the bitbots URDF uses
            # mirrored rotation axes (see joint_inversions in csv_to_npz.py). The
            # right hip/ankle pitch and the left calf are the inverted ones.
            "l_hip_pitch_joint": -0.6,
            "r_hip_pitch_joint": 0.6,
            "l_calf_joint": -1.2,
            "r_calf_joint": 1.2,
            "l_ankle_pitch_joint": -0.6,
            "r_ankle_pitch_joint": 0.6,
            # Arms = walkready frame-0 from miro_rad_walkready.npz (sim convention,
            # i.e. CSV frame 0 after csv_to_npz joint_inversions). NOT 0 — arms-down
            # rest sits near the shoulder ref offset, not at 0.
            "l_shoulder_pitch_joint": -1.663410,
            "l_shoulder_roll_joint": -1.180453,
            "l_upper_arm_joint": 0.222445,
            "l_elbow_joint": -0.288728,
            "r_shoulder_pitch_joint": 1.697383,
            "r_shoulder_roll_joint": 1.166996,
            "r_upper_arm_joint": -0.305506,
            "r_elbow_joint": 0.315797,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_thigh_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim={
                ".*_thigh_joint": 30.0,
                ".*_hip_roll_joint": 30.0,
                ".*_hip_pitch_joint": 30.0,
                ".*_calf_joint": 30.0,
            },
            velocity_limit_sim={
                ".*_thigh_joint": 8.0,
                ".*_hip_roll_joint": 8.0,
                ".*_hip_pitch_joint": 8.0,
                ".*_calf_joint": 8.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_5047,
                ".*_hip_roll_joint": STIFFNESS_5047,
                ".*_thigh_joint": STIFFNESS_5047,
                ".*_calf_joint": STIFFNESS_5047,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_5047,
                ".*_hip_roll_joint": DAMPING_5047,
                ".*_thigh_joint": DAMPING_5047,
                ".*_calf_joint": DAMPING_5047,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_5047,
                ".*_hip_roll_joint": ARMATURE_5047,
                ".*_thigh_joint": ARMATURE_5047,
                ".*_calf_joint": ARMATURE_5047,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=30.0,
            velocity_limit_sim=17.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=STIFFNESS_5047,
            damping=DAMPING_5047,
            armature=ARMATURE_5047,
        ),
        
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_upper_arm_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 20.0,
                ".*_shoulder_roll_joint": 20.0,
                ".*_upper_arm_joint": 20.0,
                ".*_elbow_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 17.0,
                ".*_shoulder_roll_joint": 17.0,
                ".*_upper_arm_joint": 17.0,
                ".*_elbow_joint": 17.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_4438,
                ".*_shoulder_roll_joint": STIFFNESS_4438,
                ".*_upper_arm_joint": STIFFNESS_4438,
                ".*_elbow_joint": STIFFNESS_4438,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_4438,
                ".*_shoulder_roll_joint": DAMPING_4438,
                ".*_upper_arm_joint": DAMPING_4438,
                ".*_elbow_joint": DAMPING_4438,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_4438,
                ".*_shoulder_roll_joint": ARMATURE_4438,
                ".*_upper_arm_joint": ARMATURE_4438,
                ".*_elbow_joint": ARMATURE_4438,
            },
        ),
    },
)

PI_PLUS_ACTION_SCALE = {}
for a in PI_PLUS_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            PI_PLUS_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
