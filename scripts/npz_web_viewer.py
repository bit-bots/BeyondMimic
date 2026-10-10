"""Browser viewer for pi_plus motion NPZs (csv_to_npz.py output), for headless hosts.

Draws the robot from the NPZ's recorded world pose of every body (body_pos_w / body_quat_w),
so what you see is exactly what the tracking env will use as reference. If a source video
with the same name exists (GVHMR clip folders), it plays in sync next to the robot.

Needs only numpy (no Isaac, no display). Serves on localhost; open it through an SSH tunnel:

    pixi run python scripts/npz_web_viewer.py                    # on the server
    ssh -L 8080:localhost:8080 23grabic@cl05                     # on your machine
    -> http://localhost:8080

Keys: space play/pause, left/right step a frame, up/down previous/next file.
"""

import argparse
import http.server
import json
import os
import pathlib
import re
import socketserver
import struct
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
URDF = REPO / "source/whole_body_tracking/whole_body_tracking/assets/hightorque/pi_plus_bitbots/pi_plus_22dof.urdf"

# Isaac Lab body order of PI_PLUS_CFG (robot.body_names), i.e. the second axis of body_pos_w /
# body_quat_w in the NPZ. It is not the URDF link order. Derived from the NPZs themselves: in every
# frame each child body's origin sits exactly at its URDF joint origin in the parent body's frame,
# which pins down a unique assignment (checked on 13 NPZs, < 0.1 mm).
BODY_NAMES = [
    "base_link", "l_hip_pitch_link", "r_hip_pitch_link", "torso_link", "l_hip_roll_link", "r_hip_roll_link",
    "head_yaw_link", "l_shoulder_pitch_link", "r_shoulder_pitch_link", "l_thigh_link", "r_thigh_link",
    "head_pitch_link", "l_shoulder_roll_link", "r_shoulder_roll_link", "l_calf_link", "r_calf_link",
    "l_upper_arm_link", "r_upper_arm_link", "l_ankle_pitch_link", "r_ankle_pitch_link", "l_elbow_link",
    "r_elbow_link", "l_ankle_roll_link", "r_ankle_roll_link", "l_wrist_link", "r_wrist_link",
]

STL_DTYPE = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def rpy_to_quat_xyzw(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]


def load_stl_decimated(path, grid):
    """Binary STL -> indexed mesh, vertices clustered on a `grid` (m) lattice to cut the size."""
    raw = path.read_bytes()
    n = struct.unpack_from("<I", raw, 80)[0]
    v = np.frombuffer(raw, STL_DTYPE, n, 84)["v"].reshape(-1, 3).astype(np.float64)
    _, first, inv = np.unique(np.round(v / grid).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    counts = np.bincount(inv)
    pos = np.stack([np.bincount(inv, weights=v[:, k]) for k in range(3)], 1) / counts[:, None]
    faces = inv.reshape(-1, 3)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    faces = np.unique(faces, axis=0)
    return struct.pack("<II", len(pos), len(faces)) + pos.astype("<f4").tobytes() + faces.astype("<u4").tobytes()


def load_robot(grid):
    root = ET.parse(URDF).getroot()
    links, meshes = {}, {}
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None or vis.find("geometry/mesh") is None:
            continue
        name = link.get("name")
        mesh_file = URDF.parent / re.sub(r"^package://[^/]+/", "", vis.find("geometry/mesh").get("filename"))
        origin = vis.find("origin")
        xyz = [float(x) for x in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(x) for x in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        color = vis.find("material/color")
        rgba = [float(x) for x in color.get("rgba").split()] if color is not None else [0.75, 0.75, 0.75, 1]
        links[name] = {"xyz": xyz, "quat": rpy_to_quat_xyzw(*rpy), "rgba": rgba}
        meshes[name] = load_stl_decimated(mesh_file, grid)
    return links, meshes


def find_videos(video_dirs):
    videos = {}
    for d in video_dirs:
        for p in sorted(pathlib.Path(d).rglob("*")):
            if p.suffix.lower() in (".mp4", ".mov", ".webm"):
                videos.setdefault(p.stem, p)
    return videos


PAGE = r"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><title>NPZ Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#16181d; --panel:#1f2229; --line:#2e323b; --text:#e6e8ec; --mute:#9aa0aa; --acc:#5aa9ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:13px/1.4 system-ui, sans-serif; height:100vh; display:flex; overflow:hidden; }
  #side { width:300px; min-width:220px; background:var(--panel); border-right:1px solid var(--line); display:flex; flex-direction:column; }
  #side input { margin:8px; padding:6px 8px; background:var(--bg); color:var(--text); border:1px solid var(--line); border-radius:4px; }
  #count { padding:0 10px 6px; color:var(--mute); font-size:12px; }
  #list { flex:1; overflow:auto; }
  #list div { padding:4px 10px; cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:12px; }
  #list div:hover { background:#2a2e37; }
  #list div.sel { background:#24406a; }
  #list .dir { color:var(--mute); }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #views { flex:1; display:flex; min-height:0; }
  #c3d { flex:1; min-width:0; position:relative; }
  #vid { flex:1; min-width:0; background:#000; display:none; }
  #vid video { width:100%; height:100%; object-fit:contain; }
  #bar { display:flex; gap:10px; align-items:center; padding:8px 10px; background:var(--panel); border-top:1px solid var(--line); flex-wrap:wrap; }
  #bar button, #bar select { background:var(--bg); color:var(--text); border:1px solid var(--line); border-radius:4px; padding:4px 10px; cursor:pointer; }
  #slider { flex:1; min-width:150px; accent-color:var(--acc); }
  #info { color:var(--mute); font-variant-numeric:tabular-nums; min-width:170px; }
  #title { position:absolute; top:8px; left:10px; color:var(--mute); font-size:12px; pointer-events:none; }
  label { color:var(--mute); display:flex; gap:4px; align-items:center; }
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="side"><input id="filter" placeholder="Filter …"><div id="count"></div><div id="list"></div></div>
<div id="main">
  <div id="views"><div id="c3d"><div id="title"></div></div><div id="vid"><video id="video" muted playsinline preload="auto"></video></div></div>
  <div id="bar">
    <button id="play">▶</button>
    <input id="slider" type="range" min="0" max="0" value="0">
    <span id="info">–</span>
    <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
    <label><input id="loop" type="checkbox" checked>Loop</label>
    <label><input id="follow" type="checkbox" checked>Kamera folgt</label>
    <label><input id="showvid" type="checkbox" checked>Video</label>
  </div>
</div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const robot = await (await fetch("api/robot")).json();
let files = await (await fetch("api/list")).json();

// --- scene (Z-up like Isaac) ---
const host = $("c3d");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1e24);
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(1.4, -1.4, 0.8);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.3);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(2, -3, 5); scene.add(sun);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color: 0x2a2e36, roughness: 1 }));
floor.position.z = -0.001; scene.add(floor);
const grid = new THREE.GridHelper(40, 160, 0x7a8190, 0x484e5a); grid.rotation.x = Math.PI / 2; scene.add(grid);
const axes = new THREE.AxesHelper(0.2); scene.add(axes);

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);

// --- robot bodies ---
const bodies = [];
for (const name of robot.bodies) {
  const g = new THREE.Group(); scene.add(g); bodies.push(g);
  const link = robot.links[name];
  if (!link) continue;
  fetch("api/mesh/" + name).then(r => r.arrayBuffer()).then(buf => {
    const [nv, nf] = new Uint32Array(buf, 0, 2);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buf, 8, nv * 3), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(buf, 8 + nv * 12, nf * 3), 1));
    geo.computeVertexNormals();
    const [r, gg, b] = link.rgba;
    const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: new THREE.Color(r, gg, b), roughness: 0.6, metalness: 0.1 }));
    m.position.fromArray(link.xyz); m.quaternion.fromArray(link.quat);
    g.add(m);
  });
}

// --- motion state ---
let motion = null, frame = 0, playing = false, tPlay = 0, last = performance.now(), current = -1;
const video = $("video");

function setFrame(f) {
  if (!motion) return;
  frame = Math.max(0, Math.min(motion.n - 1, f));
  const nb = motion.nb, o = frame * nb;
  for (let i = 0; i < nb && i < bodies.length; i++) {
    const p = motion.pos, q = motion.quat, k = (o + i);
    bodies[i].position.set(p[k * 3], p[k * 3 + 1], p[k * 3 + 2]);
    bodies[i].quaternion.set(q[k * 4 + 1], q[k * 4 + 2], q[k * 4 + 3], q[k * 4]);  // wxyz -> xyzw
  }
  $("slider").value = frame;
  $("info").textContent = `${frame + 1}/${motion.n}  ${(frame / motion.fps).toFixed(2)}/${((motion.n - 1) / motion.fps).toFixed(2)} s`;
  if ($("follow").checked) {
    const root = bodies[0].position, d = new THREE.Vector3().subVectors(root, controls.target);
    d.z = 0; controls.target.add(d); camera.position.add(d);
  }
}

async function load(i) {
  if (i < 0 || i >= files.length) return;
  current = i;
  document.querySelectorAll("#list div[data-i]").forEach(el => el.classList.toggle("sel", +el.dataset.i === i));
  document.querySelector(`#list div[data-i="${i}"]`)?.scrollIntoView({ block: "nearest" });
  const f = files[i];
  $("title").textContent = f.path;
  const buf = await (await fetch("api/motion?f=" + encodeURIComponent(f.path))).arrayBuffer();
  const [n, nb] = new Uint32Array(buf, 0, 2), fps = new Float32Array(buf, 8, 1)[0];
  motion = { n, nb, fps, pos: new Float32Array(buf, 12, n * nb * 3), quat: new Float32Array(buf, 12 + n * nb * 12, n * nb * 4) };
  $("slider").max = n - 1;
  play(false);
  tPlay = 0; setFrame(0);
  const root = bodies[0].position;
  controls.target.set(root.x, root.y, 0.3); camera.position.set(root.x + 1.4, root.y - 1.4, 0.8);
  if (f.video) {
    // play() only drives the video once its metadata is in; without waiting, the first pass ran without it.
    const ready = new Promise((res) => {
      video.addEventListener("loadedmetadata", res, { once: true });
      video.addEventListener("error", res, { once: true });
      setTimeout(res, 5000);
    });
    video.src = "api/video/" + encodeURIComponent(f.video);
    await ready;
    if (current !== i) return;  // another file was picked while the video loaded
  } else { video.removeAttribute("src"); video.load(); }
  showVideo();
  play(true);
}

function showVideo() { $("vid").style.display = files[current]?.video && $("showvid").checked ? "block" : "none"; resize(); }
function hasVideo() { return files[current]?.video && $("showvid").checked && video.readyState >= 1; }

function play(on) {
  playing = on; $("play").textContent = on ? "❚❚" : "▶";
  video.playbackRate = +$("speed").value;
  if (hasVideo()) { video.currentTime = frame / motion.fps; on ? video.play().catch(() => {}) : video.pause(); }
  last = performance.now(); tPlay = frame / (motion?.fps || 30);
}

function seek(f) { setFrame(f); tPlay = frame / motion.fps; if (hasVideo()) video.currentTime = tPlay; }

function tick(now) {
  if (playing && motion) {
    const dur = (motion.n - 1) / motion.fps;
    let t = hasVideo() && !video.paused ? video.currentTime : tPlay + (now - last) / 1000 * +$("speed").value;
    if (t >= dur || (hasVideo() && video.ended)) {
      if ($("loop").checked) { t = 0; if (hasVideo()) { video.currentTime = 0; video.play().catch(() => {}); } }
      else { t = dur; play(false); }
    }
    tPlay = t; setFrame(Math.round(t * motion.fps));
  }
  last = now;
  controls.update(); renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// --- file list ---
function renderList() {
  const q = $("filter").value.toLowerCase();
  const list = $("list"); list.innerHTML = "";
  let shown = 0, lastDir = null;
  files.forEach((f, i) => {
    if (q && !f.path.toLowerCase().includes(q)) return;
    const dir = f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")) : ".";
    if (dir !== lastDir) { const d = document.createElement("div"); d.className = "dir"; d.textContent = dir + "/"; list.appendChild(d); lastDir = dir; }
    const el = document.createElement("div");
    el.dataset.i = i; el.textContent = (f.video ? "🎞 " : "   ") + f.path.slice(dir === "." ? 0 : dir.length + 1);
    el.title = f.path; if (i === current) el.classList.add("sel");
    el.onclick = () => load(i); list.appendChild(el); shown++;
  });
  $("count").textContent = `${shown} von ${files.length} Dateien`;
}
function visible() { return [...document.querySelectorAll("#list div[data-i]")].map(el => +el.dataset.i); }
$("filter").oninput = renderList;
renderList();

$("play").onclick = () => play(!playing);
$("slider").oninput = (e) => { if (playing) play(false); seek(+e.target.value); };
$("speed").onchange = () => { video.playbackRate = +$("speed").value; };
$("showvid").onchange = () => { showVideo(); play(playing); };
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type === "text") return;
  if (e.code === "Space") { play(!playing); e.preventDefault(); }
  else if (e.code === "ArrowRight") { if (playing) play(false); seek(frame + 1); }
  else if (e.code === "ArrowLeft") { if (playing) play(false); seek(frame - 1); }
  else if (e.code === "ArrowDown" || e.code === "ArrowUp") {
    const v = visible(), k = v.indexOf(current);
    load(v[Math.max(0, Math.min(v.length - 1, k + (e.code === "ArrowDown" ? 1 : -1)))] ?? v[0]); e.preventDefault();
  }
});
if (files.length) load(visible()[0]);
</script></body></html>
"""


def make_handler(npz_dir, links, meshes, videos):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype, code=200, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            path, q = url.path, urllib.parse.parse_qs(url.query)
            try:
                if path == "/":
                    self.send(PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/api/robot":
                    self.send(json.dumps({"bodies": BODY_NAMES, "links": links}).encode(), "application/json")
                elif path == "/api/list":
                    out = []
                    for p in sorted(npz_dir.rglob("*.npz")):
                        rel = p.relative_to(npz_dir).as_posix()
                        out.append({"path": rel, "video": p.stem if p.stem in videos else None})
                    self.send(json.dumps(out).encode(), "application/json")
                elif path.startswith("/api/mesh/"):
                    name = path.rsplit("/", 1)[1]
                    self.send(meshes[name], "application/octet-stream", extra={"Cache-Control": "max-age=86400"})
                elif path == "/api/motion":
                    p = (npz_dir / q["f"][0]).resolve()
                    if npz_dir not in p.parents:
                        raise KeyError(p)
                    d = np.load(p)
                    pos, quat = d["body_pos_w"].astype("<f4"), d["body_quat_w"].astype("<f4")
                    head = struct.pack("<IIf", pos.shape[0], pos.shape[1], float(d["fps"].reshape(-1)[0]))
                    self.send(head + pos.tobytes() + quat.tobytes(), "application/octet-stream")
                elif path.startswith("/api/video/"):
                    self.send_video(videos[urllib.parse.unquote(path.rsplit("/", 1)[1])])
                else:
                    self.send(b"not found", "text/plain", 404)
            except (KeyError, FileNotFoundError, ValueError) as e:
                self.send(f"not found: {e}".encode(), "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_video(self, p):
            size = p.stat().st_size
            ctype = "video/mp4" if p.suffix.lower() in (".mp4", ".mov") else "video/webm"
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            start, end = 0, size - 1
            if m:
                start = int(m.group(1) or 0)
                end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            with open(p, "rb") as fh:
                fh.seek(start)
                body = fh.read(end - start + 1)
            extra = {"Accept-Ranges": "bytes"}
            if m:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self.send(body, ctype, 206 if m else 200, extra)

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(REPO / "source/motion/hightorque/pi_plus/npz"), help="Folder searched recursively for .npz")
    ap.add_argument("--video_dirs", nargs="*", default=[str(REPO / "GVHMR/docs/example_video")],
                    help="Folders searched for source videos with the same name as an NPZ")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default: only reachable through the SSH tunnel)")
    ap.add_argument("--mesh_grid", type=float, default=0.002, help="Mesh decimation grid in m (0.002 = 2 mm)")
    args = ap.parse_args()

    npz_dir = pathlib.Path(args.dir).resolve()
    print("Loading meshes …", flush=True)
    links, meshes = load_robot(args.mesh_grid)
    print(f"  {len(meshes)} meshes, {sum(map(len, meshes.values())) / 1e6:.1f} MB after decimation", flush=True)
    videos = find_videos(args.video_dirs)
    n_npz = sum(1 for _ in npz_dir.rglob("*.npz"))
    print(f"  {n_npz} NPZ under {npz_dir}, {len(videos)} source videos", flush=True)
    with Server((args.host, args.port), make_handler(npz_dir, links, meshes, videos)) as srv:
        print(f"Serving on http://{args.host}:{args.port}  (tunnel: ssh -L {args.port}:localhost:{args.port} {os.environ.get('USER', 'user')}@<host>)", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
