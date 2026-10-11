"""Viser preview of a GMR-retargeted Pi Plus qpos sequence (raw mujoco model).

Runs in the pixi gmr env (mujoco + viser). Renders the pi_football visual mesh geoms
and plays back the retargeted qpos so the retarget can be eyeballed for the torso
orientation and left/right symmetry.

Usage (from the repo root):
  pixi run -e gmr python GMR/scripts/gmr_viz.py --npz /path/to/retarget_output.npz --port 8080
  pixi run -e gmr python GMR/scripts/gmr_viz.py --npz /path/to/dir_of_npz --port 8080

If --npz is a directory, the viewer shows one button per clip. All clips
must use the same robot XML.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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

  src = Path(args.npz)
  paths = sorted(src.glob("*.npz")) if src.is_dir() else [src]
  clips = {}
  for path in paths:
    d = np.load(path, allow_pickle=True)
    clips[path.stem] = (d["qpos"], str(d["xml"]), float(d["fps"]))
  names = list(clips)
  qpos, xml, fps = clips[names[0]]
  print(f"{len(clips)} clip(s); first: qpos {qpos.shape}, xml {xml}, fps {fps}")

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
  # One button per clip: a dropdown is unusable on mobile (small tap target,
  # scroll-vs-open conflicts).
  if len(names) > 1:
    current_label = server.gui.add_markdown(f"**Playing:** {names[0]}")

    def select_clip(name: str) -> None:
      nonlocal qpos, fps
      qpos, _xml, fps = clips[name]
      frame_slider.max = len(qpos) - 1
      frame_slider.value = 0
      current_label.content = f"**Playing:** {name}"

    for name in names:
      btn = server.gui.add_button(name)

      def _make_handler(n: str):
        def _(_e) -> None:
          select_clip(n)

        return _

      btn.on_click(_make_handler(name))

  @play.on_click
  def _(_e) -> None:
    playing["v"] = not playing["v"]

  def render(frame: int) -> None:
    mjdata.qpos[:] = qpos[min(frame, len(qpos) - 1)]
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
