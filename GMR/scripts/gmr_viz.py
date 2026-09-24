"""Viser preview of a GMR-retargeted Pi Plus qpos sequence (raw mujoco model).

Runs in the GMR venv (mujoco + viser). Renders the pi_football visual mesh geoms
and plays back the retargeted qpos so the retarget can be eyeballed for the torso
orientation and left/right symmetry.

Usage (GMR venv; run from the directory containing both gmrvenv/ and BeyondMimic/):
  gmrvenv/bin/python BeyondMimic/GMR/scripts/gmr_viz.py --npz /path/to/retarget_output.npz --port 8080
"""

from __future__ import annotations

import argparse
import time

import mujoco as mj
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R


def _geom_mesh(model, gid: int):
  did = int(model.geom_dataid[gid])
  va, vn = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
  fa, fn = int(model.mesh_faceadr[did]), int(model.mesh_facenum[did])
  verts = np.asarray(model.mesh_vert[va : va + vn]).reshape(-1, 3).astype(np.float32)
  faces = np.asarray(model.mesh_face[fa : fa + fn]).reshape(-1, 3).astype(np.int32)
  return verts, faces


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", required=True)
  ap.add_argument("--port", type=int, default=8080)
  args = ap.parse_args()

  data = np.load(args.npz, allow_pickle=True)
  qpos = data["qpos"]
  xml = str(data["xml"])
  fps = float(data["fps"])
  print(f"qpos {qpos.shape}, xml {xml}, fps {fps}")

  model = mj.MjModel.from_xml_path(xml)
  mjdata = mj.MjData(model)

  server = viser.ViserServer(port=args.port)
  server.scene.add_grid("/grid", width=4.0, height=4.0)

  handles = []
  for gid in range(model.ngeom):
    if int(model.geom_type[gid]) != int(mj.mjtGeom.mjGEOM_MESH):
      continue
    rgba = np.asarray(model.geom_rgba[gid])
    if rgba[3] == 0:
      continue
    verts, faces = _geom_mesh(model, gid)
    h = server.scene.add_mesh_simple(
      f"/geom_{gid}",
      vertices=verts,
      faces=faces,
      color=tuple((rgba[:3] * 255).astype(np.uint8).tolist()),
      opacity=float(rgba[3]),
    )
    handles.append((gid, h))
  print(f"rendering {len(handles)} mesh geoms")

  playing = {"v": True}
  frame_slider = server.gui.add_slider(
    "Frame", min=0, max=len(qpos) - 1, step=1, initial_value=0
  )
  speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
  play = server.gui.add_button("Play / Pause")

  @play.on_click
  def _(_e) -> None:
    playing["v"] = not playing["v"]

  def render(frame: int) -> None:
    mjdata.qpos[:] = qpos[frame]
    mj.mj_forward(model, mjdata)
    for gid, h in handles:
      p = np.asarray(mjdata.geom_xpos[gid])
      quat_xyzw = R.from_matrix(np.asarray(mjdata.geom_xmat[gid]).reshape(3, 3)).as_quat()
      h.position = (float(p[0]), float(p[1]), float(p[2]))
      h.wxyz = (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
      )

  print(f"Viser on http://localhost:{args.port}")
  while True:
    render(int(frame_slider.value))
    if playing["v"]:
      frame_slider.value = (int(frame_slider.value) + 1) % len(qpos)
    time.sleep(1.0 / (fps * speed.value))


if __name__ == "__main__":
  main()
