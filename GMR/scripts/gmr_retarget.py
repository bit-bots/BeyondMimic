"""Retarget a LAFAN1 or CMU BVH clip to the Pi Plus (pi_football) via GMR.

CMU-style clips (e.g. the CMU soccer kicks) are auto-detected and loaded via
GMR's cmu_bvh module (per-bone T-pose-calibrated correction to LAFAN1
convention, see general_motion_retargeting/utils/cmu_bvh.py). That loader
only emits bones it can calibrate against a T-pose, which excludes "Hips"
(no reference T-pose orientation for it was captured) -- see
inject_mixed_root below for how the root injection handles that.

Runs in the GMR venv. Saves qpos + xml path + fps for playback/preview.

Full pipeline for a new BVH clip (LAFAN1 or CMU, this script auto-detects
which):
  1. gmr_retarget.py -- this script. Produces a qpos + xml + fps npz.
  2. (optional) preview the retargeted robot from that npz in a viewer.
  3. Convert to whatever downstream training format you need (e.g.
     AMP_mjlab's scripts/piplus_gmr_to_amp.py for AMP-style motion imitation).

Usage (GMR venv, from this repo's root):
  gmrvenv/bin/python GMR/scripts/gmr_retarget.py --start 900 --end 3900
  gmrvenv/bin/python GMR/scripts/gmr_retarget.py --bvh /path/to/cmu_kick.bvh --out out.npz

ROBOT_XML below points at a specific downstream project's Pi Plus MJCF,
verified (via a 1000-random-config FK comparison) to exactly match the real
robot's URDF -- unlike general_motion_retargeting's own bundled pi_football
demo XML, which uses a different, non-matching axis/ref convention on
several joints. If you're retargeting to a different Pi Plus MJCF, point
ROBOT_XML at your own model instead; the IK config's offset_quat values
(ik_configs/bvh_to_pi_football.json) assume this project's mirrored
shoulder-pitch/elbow convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import general_motion_retargeting.params as gmr_params
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Point GMR's pi_football entry at the verified-correct Pi Plus MJCF. Must
# happen before GMR reads ROBOT_XML_DICT (mutating the shared dict is enough).
# See the module docstring above for why this isn't general_motion_retargeting's
# own bundled demo XML.
ROBOT_XML = "/homes/17vahl/smp/smp/src/smp/robot/piplus/piplus.xml"
gmr_params.ROBOT_XML_DICT["pi_football"] = ROBOT_XML

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.cmu_bvh import (  # noqa: E402
  is_cmu_bvh,
  load_cmu_bvh_file,
)
from general_motion_retargeting.utils.lafan1 import load_lafan1_file  # noqa: E402

# Standing seed in ROBOT_XML's qpos convention (mirrored, in-range). Seeds the
# IK away from the out-of-range shoulder-roll refs.
#
# Arm values (shoulder_pitch/roll) match bitbots_main's real walkready pose
# directly, no sign correction -- verified with a 1000-random-config FK
# comparison between ROBOT_XML and the real robot's URDF: every one of the 20
# shared joints has IDENTICAL axis and zero-convention between the two models
# (0.0 rotation-matrix error at every sampled angle).
SEED_QPOS = {
  "r_hip_pitch_joint": 0.5, "l_hip_pitch_joint": -0.5,
  "r_hip_roll_joint": 0.0, "l_hip_roll_joint": 0.0,
  "r_thigh_joint": 0.0, "l_thigh_joint": 0.0,
  "r_calf_joint": 1.0, "l_calf_joint": -1.0,
  "r_ankle_pitch_joint": 0.5, "l_ankle_pitch_joint": -0.5,
  "r_ankle_roll_joint": 0.0, "l_ankle_roll_joint": 0.0,
  "r_shoulder_pitch_joint": -1.5708, "l_shoulder_pitch_joint": 1.5708,
  "r_shoulder_roll_joint": -1.2217, "l_shoulder_roll_joint": 1.2217,
  "r_upper_arm_joint": 0.0, "l_upper_arm_joint": 0.0,
  "r_elbow_joint": 0.0, "l_elbow_joint": 0.0,
}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument(
    "--bvh",
    default=str(Path(__file__).resolve().parent.parent / "MotionData/lafan1/run1_subject2.bvh"),
  )
  ap.add_argument("--out", default="gmr_retarget_output.npz")
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--end", type=int, default=-1, help="-1 = end")
  ap.add_argument(
    "--foot-target-z", type=float, default=0.0,
    help="shift root so the lowest FOOT geom center over all frames reaches this z",
  )
  ap.add_argument(
    "--root-blend", type=float, default=0.5,
    help="root orientation = slerp(Hips, Spine2, blend); 0=pelvis, 1=upper spine",
  )
  ap.add_argument(
    "--ik-config-path", default=None,
    help="override the bvh->pi_football IK config file (default: "
         "ik_configs/bvh_to_pi_football.json)",
  )
  args = ap.parse_args()

  if args.ik_config_path is not None:
    gmr_params.IK_CONFIG_DICT["bvh"]["pi_football"] = Path(args.ik_config_path)

  cmu = is_cmu_bvh(args.bvh)
  frames, human_height = (load_cmu_bvh_file if cmu else load_lafan1_file)(args.bvh)
  end = len(frames) if args.end < 0 else min(args.end, len(frames))
  print(f"bvh {len(frames)} frames, h {human_height:.3f}; retargeting [{args.start}:{end}] "
        f"to {ROBOT_XML} (root_blend={args.root_blend})")

  retarget = GMR(
    src_human="bvh",
    tgt_robot="pi_football",
    actual_human_height=human_height,
    init_qpos=SEED_QPOS,
    # Motor names keyed without the "_joint" suffix mismatch GMR's
    # motor-name-keyed velocity limit lookup against mink's joint lookup.
    # Disable it (not needed for offline retargeting).
    use_velocity_limit=False,
  )

  def inject_mixed_root(frame: dict) -> dict:
    """Add a synthetic 'MixedRoot' body: Hips position, slerp(Hips, Spine2) rot.

    load_cmu_bvh_file doesn't emit "Hips" (its per-bone correction is only derived
    where a LAFAN1 T-pose reference orientation was captured, which excludes the
    root). Rather than invent an uncalibrated Hips correction, CMU frames fall back
    to Spine2 alone for MixedRoot (root_blend has no effect on those frames).
    """
    if "Hips" not in frame:
      frame["MixedRoot"] = frame["Spine2"]
      return frame
    hips_pos, hips_q = frame["Hips"]  # quats are wxyz (scalar-first)
    _, spine_q = frame["Spine2"]
    key = R.from_quat(np.array([hips_q, spine_q]), scalar_first=True)
    mixed_q = Slerp([0.0, 1.0], key)([args.root_blend])[0].as_quat(scalar_first=True)
    frame["MixedRoot"] = (hips_pos, mixed_q)
    return frame

  qpos = []
  for i in range(args.start, end):
    qpos.append(retarget.retarget(inject_mixed_root(frames[i])).copy())
    if (i - args.start) % 300 == 0:
      print(f"  {i - args.start}/{end - args.start}")
  qpos = np.asarray(qpos)

  # Ground-height fix based on FOOT geoms only (ankle_roll bodies), so an outlier
  # geom elsewhere doesn't over-lift the robot.
  model = mj.MjModel.from_xml_path(retarget.xml_file)
  data = mj.MjData(model)
  foot_geoms = [
    g for g in range(model.ngeom)
    if "ankle_roll"
    in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "")
  ]
  per_frame_min = np.empty(len(qpos))
  for k, q in enumerate(qpos):
    data.qpos[:] = q
    mj.mj_forward(model, data)
    per_frame_min[k] = float(data.geom_xpos[foot_geoms, 2].min())
  # Robust reference: 25th percentile of per-frame lowest foot (the typical
  # stance level), not the single deepest frame — avoids over-lifting.
  ref = float(np.percentile(per_frame_min, 25))
  shift = args.foot_target_z - ref
  qpos[:, 2] += shift
  print(f"height adjust (feet p25): ref {ref:.3f} (min {per_frame_min.min():.3f}) "
        f"-> shift {shift:+.3f}")

  np.savez(args.out, qpos=qpos, xml=retarget.xml_file, fps=30)
  print(f"saved {args.out}: qpos {qpos.shape}")


if __name__ == "__main__":
  main()
