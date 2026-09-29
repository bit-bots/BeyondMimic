"""Viser preview of a raw BVH skeleton (pre-retarget), using the SAME GMR FK the
retarget consumes, so we can eyeball the source motion itself.

Draws joints as a point cloud and bones as line segments, in the Z-up, human-scaled
frame the retarget loader builds (Y-up->Z-up, scaled to ~1.75 m, feet on ground).

Usage (GMR venv; run from the directory containing both gmrvenv/ and BeyondMimic/):
  gmrvenv/bin/python BeyondMimic/GMR/scripts/bvh_viz.py --bvh /path/to/clip.bvh --port 8080
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import viser

import general_motion_retargeting.utils.lafan_vendor.utils as u
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

_Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--bvh", required=True)
  ap.add_argument("--port", type=int, default=8080)
  ap.add_argument("--fps", type=float, default=120.0, help="BVH native fps (CMU=120)")
  args = ap.parse_args()

  d = read_bvh(args.bvh)
  _, gpos = u.quat_fk(d.quats, d.pos, d.parents)  # (T, J, 3) native, Y-up
  bones = list(d.bones)
  parents = np.asarray(d.parents)

  # Match the retarget loader: Y-up -> Z-up, scale to ~1.75 m standing, feet at z=0.
  pos = gpos @ _Y_UP_TO_Z_UP.T
  hi = bones.index("Head")
  lf, rf = bones.index("LeftFoot"), bones.index("RightFoot")
  native_h = float(np.max(pos[:, hi, 2] - np.minimum(pos[:, lf, 2], pos[:, rf, 2])))
  pos *= 1.75 / native_h
  pos[:, :, 2] -= pos[:, :, 2].min()  # drop onto the ground

  T = pos.shape[0]
  # Bone index pairs (child, parent) for line segments.
  pairs = [(i, int(p)) for i, p in enumerate(parents) if p >= 0]
  print(f"{args.bvh}: {T} frames, {len(bones)} joints, {len(pairs)} bones @ {args.fps}fps")

  server = viser.ViserServer(port=args.port)
  server.scene.add_grid("/grid", width=4.0, height=4.0)

  def frame_segments(f: int) -> np.ndarray:
    return np.stack([np.stack([pos[f, c], pos[f, p]]) for c, p in pairs]).astype(np.float32)

  # Initial draw.
  server.scene.add_point_cloud(
    "/joints", points=pos[0].astype(np.float32), colors=(255, 80, 80), point_size=0.02
  )
  server.scene.add_line_segments(
    "/bones", points=frame_segments(0), colors=(80, 160, 255), line_width=3.0
  )
  print(f"Viser on http://localhost:{args.port}  (playing {T} frames)")

  dt = 1.0 / args.fps
  f = 0
  while True:
    server.scene.add_point_cloud(
      "/joints", points=pos[f].astype(np.float32), colors=(255, 80, 80), point_size=0.02
    )
    server.scene.add_line_segments(
      "/bones", points=frame_segments(f), colors=(80, 160, 255), line_width=3.0
    )
    f = (f + 1) % T
    time.sleep(dt)


if __name__ == "__main__":
  main()
