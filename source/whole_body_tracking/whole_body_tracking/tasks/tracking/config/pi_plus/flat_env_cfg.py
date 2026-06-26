
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.robots.pi_plus import PI_PLUS_CFG, PI_PLUS_ACTION_SCALE

from whole_body_tracking.tasks.tracking.config.pi_plus.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm

@configclass
class PIPLUSFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PI_PLUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = PI_PLUS_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "l_hip_roll_link",
            "l_calf_link",
            "l_ankle_roll_link",
            "r_hip_roll_link",
            "r_calf_link",
            "r_ankle_roll_link",

            "l_shoulder_roll_link",
            "l_elbow_link",
            "r_shoulder_roll_link",
            "r_elbow_link",
        ]
        
        # 相机设置：跟随机器人，使其在画面中央
        self.viewer.eye = (3.0, 3.0, 2.0)  # 相机位置（相对机器人的偏移）
        self.viewer.lookat = (0.0, 0.0, 0.3)  # 看向位置（相对机器人的偏移）
        self.viewer.origin_type = "asset_root"  # 跟随机器人根节点
        self.viewer.asset_name = "robot"  # 绑定到机器人资产
        
        # 关闭调试可视化显示
        # self.commands.motion.debug_vis = False  # 关闭motion命令的调试可视化
        self.scene.contact_forces.debug_vis = False  # 关闭接触力可视化
        
        # 保持训练时的默认设置（自适应采样等）\
        self.rewards.motion_body_pos.weight = 1.5  # 从1.0增加到1.5
        self.rewards.motion_body_pos.params["std"] = 0.15  # 从1.0增加到1.5

        self.terminations.ee_body_pos.params["body_names"] = [
            "l_ankle_roll_link",
            "r_ankle_roll_link",
            "l_elbow_link",      # bitbots: wrist_link merged into elbow (fixed joint)
            "r_elbow_link",
        ]
        # Penalize ground contact on everything EXCEPT the legs (hip/thigh/calf/ankle)
        # and the forearms/hands (wrist). => torso, head, shoulders and upper arms are
        # penalized; the whole legs incl. feet and the wrists are allowed to touch.
        # (elbow_link has no collider -> no-op; the lower-arm collider sits on the
        # now-allowed wrist_link.)
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[r"^(?!.*(?:hip|thigh|calf|ankle|wrist)).*$"],
        )
        self.rewards.undesired_contacts.weight = -2.5
        # 如需演示模式，请使用 Tracking-Flat-PI-Plus-Play-v0
        
        # 修复base_com事件配置，使用base_link而不是torso_link
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )

@configclass
class PIPLUSFlatWoEnvCfg(PIPLUSFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.5, 3),
                "dynamic_friction_range": (0.5, 3),
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        )

