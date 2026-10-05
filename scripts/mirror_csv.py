#!/usr/bin/env python3
"""Mirror a retargeted pi_plus motion CSV left <-> right (reflection through the x-z plane).

Meant to run right before csv_to_npz.py (after csv_cut_pi_plus.py / add_walkready.py).
Output goes to <input>_mirrored.csv unless --output_csv is given.

What a mirror does to a frame
-----------------------------
  * root position:    y -> -y
  * root orientation: R -> M R M with M = diag(1, -1, 1); for a quaternion (x, y, z, w)
                      that is (-x, y, -z, w)  (yaw and roll flip, pitch stays)
  * joints:           l_<j> and r_<j> swap their trajectories, each multiplied by
                      MIRROR_SIGN[<j>]

Why every joint sign is -1 on the bitbots Pi Plus
-------------------------------------------------
Reflecting a revolute joint with axis a and angle q gives a rotation about -M a by q.
If the counterpart joint's axis equals -M a the angle carries over unchanged, if it
equals +M a it is negated. In pi_plus_bitbots/mjcf/pi_plus_22dof.xml:

  hip_pitch, calf, ankle_pitch, shoulder_pitch, elbow:  l axis = -(r axis) along y  -> -1
  hip_roll, ankle_roll, shoulder_roll:                  both +x                     -> -1
  thigh, upper_arm (yaw):                               both -z                     -> -1

The shoulder ref="+-1.5708" (and the URDF's joint-frame rpy) are mirror-symmetric too,
so they need no extra offset. Every asymmetric joint limit in the model is the mirror
image of its counterpart's, which is the same statement. --verify (on by default)
checks all of this by forward kinematics on the MJCF instead of trusting the table.
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_XML = os.path.join(
    REPO_ROOT,
    "source/whole_body_tracking/whole_body_tracking/assets/hightorque/pi_plus_bitbots/mjcf/pi_plus_22dof.xml",
)

ROOT_COLS = ["root pos x", "root pos y", "root pos z", "root rot x", "root rot y", "root rot z", "root rot w"]
# joint name without the l_/r_ prefix -> sign applied when copying one side onto the other
MIRROR_SIGN = {
    "hip_pitch": -1.0, "hip_roll": -1.0, "thigh": -1.0, "calf": -1.0, "ankle_pitch": -1.0, "ankle_roll": -1.0,
    "shoulder_pitch": -1.0, "shoulder_roll": -1.0, "upper_arm": -1.0, "elbow": -1.0,
}
JOINTS = [f"{s}_{j}" for s in ("l", "r") for j in MIRROR_SIGN]


def read_csv(path):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = [h.strip() for h in next(rdr)]
        rows = [r for r in rdr if r]
    missing = [c for c in ROOT_COLS + JOINTS if c not in header]
    if missing:
        sys.exit(f"error: {path} is missing columns {missing}. Is this a pi_plus retargeting CSV with a header?")
    return header, rows


def mirror_rows(header, data):
    """data: (N, C) float array laid out as `header`. Returns the mirrored copy."""
    col = {h: i for i, h in enumerate(header)}
    out = data.copy()
    out[:, col["root pos y"]] *= -1.0
    out[:, col["root rot x"]] *= -1.0
    out[:, col["root rot z"]] *= -1.0
    for j, sign in MIRROR_SIGN.items():
        out[:, col[f"l_{j}"]] = sign * data[:, col[f"r_{j}"]]
        out[:, col[f"r_{j}"]] = sign * data[:, col[f"l_{j}"]]
    return out


def verify_fk(xml_path, header, frames, n_random=200, tol=2e-3):
    """FK check: body l_X in the mirrored pose must be the reflection of body r_X in the original.

    Uses the given frames plus random in-range configurations. The tolerance absorbs the
    model's tiny built-in asymmetry (l shoulder z 0.11698 vs r 0.117).
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(model)
    col = {h: i for i, h in enumerate(header)}
    qadr = {j: model.joint(f"{j}_joint").qposadr[0] for j in JOINTS}
    rng_lo = {j: model.jnt_range[model.joint(f"{j}_joint").id][0] for j in JOINTS}
    rng_hi = {j: model.jnt_range[model.joint(f"{j}_joint").id][1] for j in JOINTS}

    def fk(row):
        d.qpos[:] = 0.0
        d.qpos[0:3] = [row[col["root pos x"]], row[col["root pos y"]], row[col["root pos z"]]]
        x, y, z, w = (row[col[c]] for c in ROOT_COLS[3:])
        d.qpos[3:7] = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
        for j in JOINTS:
            d.qpos[qadr[j]] = row[col[j]]
        mujoco.mj_kinematics(model, d)
        return d.xpos.copy(), d.xmat.reshape(-1, 3, 3).copy()

    rng = np.random.default_rng(0)
    rand = np.tile(frames[:1], (n_random, 1))
    for j in JOINTS:
        rand[:, col[j]] = rng.uniform(rng_lo[j], rng_hi[j], n_random)
    q = rng.normal(size=(n_random, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for k, c in enumerate(ROOT_COLS[3:]):
        rand[:, col[c]] = q[:, k]
    samples = np.concatenate([frames, rand])

    # Only the paired l_/r_ bodies (plus the root) carry the motion; the head/camera are
    # fixed and off-centre on the real robot, so they have no mirror image to compare to.
    names = [model.body(b).name for b in range(model.nbody)]
    pairs = [(i, names.index(("r_" if n[0] == "l" else "l_") + n[2:]))
             for i, n in enumerate(names) if n[:2] in ("l_", "r_")]
    pairs.append((model.body("base_link").id,) * 2)

    # Positions must reflect exactly. Orientations: both frames are right-handed, so the
    # mirrored body frame equals M R_partner M up to a constant per-body rotation; that
    # relation Rb^T M Ra M must therefore not change from pose to pose.
    M = np.diag([1.0, -1.0, 1.0])
    mirrored = mirror_rows(header, samples)
    worst_pos, worst_rot, worst_body, rel0 = 0.0, 0.0, "", None
    for a, b in zip(samples, mirrored):
        pa, Ra = fk(a)
        pb, Rb = fk(b)
        for i, p in pairs:
            ep = float(np.linalg.norm(pb[i] - M @ pa[p]))
            if ep > worst_pos:
                worst_pos, worst_body = ep, names[i]
        rel = np.stack([Rb[i].T @ M @ Ra[p] @ M for i, p in pairs])
        rel0 = rel if rel0 is None else rel0
        worst_rot = max(worst_rot, float(np.max(np.abs(rel - rel0))))

    ok = worst_pos < tol and worst_rot < tol
    print(f"[verify] FK over {len(samples)} poses ({len(frames)} from the clip, {n_random} random): "
          f"max body position error {worst_pos * 1000:.3f} mm ({worst_body}), "
          f"max frame-relation drift {worst_rot:.2e} -> {'OK' if ok else 'FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_csv")
    ap.add_argument("--output_csv", default=None, help="default: <input>_mirrored.csv")
    ap.add_argument("--xml", default=DEFAULT_XML, help="MJCF used for the FK verification")
    ap.add_argument("--no-verify", action="store_true", help="skip the MuJoCo forward-kinematics check")
    args = ap.parse_args()

    header, rows = read_csv(args.input_csv)
    num_cols = [i for i, h in enumerate(header) if h.lower() != "frame"]
    data = np.array([[float(r[i]) for i in num_cols] for r in rows])
    num_header = [header[i] for i in num_cols]
    mirrored = mirror_rows(num_header, data)

    if not args.no_verify:
        idx = np.linspace(0, len(data) - 1, min(len(data), 50)).astype(int)
        if not verify_fk(args.xml, num_header, data[idx]):
            sys.exit("error: mirror verification failed -- MIRROR_SIGN does not match the robot model")

    out_path = args.output_csv or os.path.splitext(args.input_csv)[0] + "_mirrored.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r, m in zip(rows, mirrored):
            out = list(r)
            for k, i in enumerate(num_cols):
                out[i] = repr(float(m[k]) + 0.0)  # + 0.0 turns -0.0 into 0.0
            w.writerow(out)
    print(f"wrote {len(rows)} mirrored frames -> {out_path}")


if __name__ == "__main__":
    main()
