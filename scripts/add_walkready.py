#!/usr/bin/env python3
"""Bracket a retargeted pi_plus motion CSV with a velocity-continuous walkready pose.

Takes a CSV in GMR/retargeting convention (columns: frame, root pos x/y/z,
root rot x/y/z/w, then the 20 joints in CSV_JOINTS order) and crossfades the
*continuing* motion into/out of a fixed walkready pose so the start/end blend in
smoothly (no abrupt velocity jump). Output goes to <input>_walkready.csv.

  FRONT (--front): walkready fades OUT while the motion fades IN, anchored at frame 0.
  BACK  (--back):  the motion fades OUT into walkready starting at --back-fade-start;
                   everything after (back-fade-start + back-fade-len) is discarded.
                   --back-fast-* fades selected joints faster (e.g. a broken IK arm).

  --joint-only:    keep the source frame's root pose (position + orientation) instead
                   of forcing an upright, standing-height walkready root. Only the joints
                   fade to the walkready targets. Use this to bracket a non-upright motion
                   (e.g. lying supine): the joints reach walkready while the body stays on
                   its back, exactly as the motion already has it.

The walkready pose is written so that scripts/csv_to_npz.py (which sign-flips the
joints in INV) reproduces the SIM-convention targets in WALKREADY_SIM.

Examples
--------
    # full bracket (what we built for the cartwheel)
    python scripts/add_walkready.py RetargetData/gvhmr/csv/pi_plus/miro_rad_cut.csv \
        --front --back --back-fade-start 70 \
        --front-fade-len 24 --back-fade-len 48 \
        --back-fast-fade-len 30 --hold-front 6 --hold-back 30

    # only fade out the end, default fast joints (left arm) faster
    python scripts/add_walkready.py motion.csv --back --back-fade-start 120 \
        --back-fade-len 40 --back-fast-fade-len 24
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# pi_plus constants (robot/convention specific -- edit here if the robot changes)
# ----------------------------------------------------------------------------
# CSV column order of the 20 dofs (after the frame + 7 root columns).
CSV_JOINTS = [
    "l_hip_pitch", "l_hip_roll", "l_thigh", "l_calf", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_pitch", "r_hip_roll", "r_thigh", "r_calf", "r_ankle_pitch", "r_ankle_roll",
    "l_shoulder_pitch", "l_shoulder_roll", "l_upper_arm", "l_elbow",
    "r_shoulder_pitch", "r_shoulder_roll", "r_upper_arm", "r_elbow",
]
# Joints whose sign csv_to_npz.py flips (GMR <-> sim). Must match that script.
INV = [
    "l_thigh", "l_calf", "r_hip_pitch", "r_thigh", "r_ankle_pitch",
    "l_upper_arm", "r_shoulder_pitch", "r_upper_arm", "r_elbow",
]
# walkready joint targets in SIM convention (radians).
# make_walkready() sign-flips the INV joints to
# write CSV convention, and csv_to_npz.py flips them back to exactly these sim values.
WALKREADY_SIM = {
    "r_hip_pitch": 0.6, "r_hip_roll": 0.0, "r_thigh": 0.0, "r_calf": 1.2,
    "r_ankle_pitch": 0.6, "r_ankle_roll": 0.0,
    "l_hip_pitch": -0.6, "l_hip_roll": 0.0, "l_thigh": 0.0, "l_calf": -1.2,
    "l_ankle_pitch": -0.6, "l_ankle_roll": 0.0,
    "l_shoulder_pitch": 1.57, "l_shoulder_roll": 1.22, "l_upper_arm": 0.0, "l_elbow": 0.0,
    "r_shoulder_pitch": -1.57, "r_shoulder_roll": -1.22, "r_upper_arm": 0.0, "r_elbow": 0.0,
}
# FK-grounded pelvis height for the bent-knee walkready (calibrated to init_state z=0.351).
WALKREADY_Z = 0.3173
DEFAULT_FAST_JOINTS = ["l_shoulder_pitch", "l_shoulder_roll", "l_upper_arm", "l_elbow"]

# verification reference (walkready_state in ordered_relevant_joint_names order)
_VERIFY_ORDER = [
    "r_hip_pitch", "r_hip_roll", "r_thigh", "r_calf", "r_ankle_pitch", "r_ankle_roll",
    "l_hip_pitch", "l_hip_roll", "l_thigh", "l_calf", "l_ankle_pitch", "l_ankle_roll",
]

N_COLS = 1 + 3 + 4 + len(CSV_JOINTS)  # frame + pos + quat + dofs


# ----------------------------------------------------------------------------
# math helpers (pose vector layout: [pos(3), quat xyzw(4), dofs(20)] = 27)
# ----------------------------------------------------------------------------
def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def slerp(q0, q1, t):
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = math.acos(d)
    s = math.sin(th0)
    return (math.sin((1 - t) * th0) / s) * q0 + (math.sin(t * th0) / s) * q1


def yaw_of(q):  # q = (x, y, z, w) -> heading about world z
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def upright_quat(yaw):  # (x, y, z, w)
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def blend(a, b, t):  # uniform crossfade a -> b
    out = np.empty(27)
    out[0:3] = a[0:3] * (1 - t) + b[0:3] * t
    out[3:7] = slerp(a[3:7], b[3:7], t)
    out[7:] = a[7:] * (1 - t) + b[7:] * t
    return out


def fade_blend(O, P, w_body, w_fast, fast_cols):  # per-joint weights (root/body vs fast joints)
    out = np.empty(27)
    out[0:3] = O[0:3] * (1 - w_body) + P[0:3] * w_body
    out[3:7] = slerp(O[3:7], P[3:7], w_body)
    w = np.full(20, w_body)
    for c in fast_cols:
        w[c - 7] = w_fast
    out[7:] = O[7:] * (1 - w) + P[7:] * w
    return out


# ----------------------------------------------------------------------------
def load_csv(path):
    with open(path) as f:
        rdr = csv.reader(f)
        header = [h.strip() for h in next(rdr)]
        data = np.array([[float(x) for x in row] for row in rdr if row])
    if len(header) != N_COLS or header[0].lower() != "frame":
        sys.exit(f"error: expected {N_COLS} columns starting with 'frame', got {len(header)} "
                 f"(header[0]={header[0]!r}). Is this a pi_plus retargeting CSV?")
    if header[8:] != CSV_JOINTS:
        sys.exit(f"error: joint columns do not match expected CSV_JOINTS order.\n"
                 f"  file: {header[8:]}\n  want: {CSV_JOINTS}")
    return header, data


def make_walkready(src, joint_only=False):
    """Build a walkready pose vector [pos(3), quat xyzw(4), dofs(20)] from a source row.

    Normal mode reroots the pose upright: root x/y from ``src``, z=WALKREADY_Z (standing
    height), quat = upright at ``src``'s yaw. joint_only keeps ``src``'s full root pose
    (position AND orientation) untouched and only swaps in the walkready joint targets --
    used to bracket a non-upright motion (e.g. lying supine) without lifting/righting it.
    """
    joints = np.zeros(len(CSV_JOINTS))
    for k, j in enumerate(CSV_JOINTS):
        if j in WALKREADY_SIM:  # fixed target: sim -> csv (negate the inverted ones)
            v = WALKREADY_SIM[j]
            joints[k] = -v if j in INV else v
        else:  # fallback for any joint without a target: keep the source frame's value
            joints[k] = src[7 + k]
    if joint_only:
        return np.concatenate([src[0:3], src[3:7], joints])  # keep root pos + orientation
    return np.concatenate([[src[0], src[1], WALKREADY_Z], upright_quat(yaw_of(src[3:7])), joints])


def build(header, data, args):
    N = data.shape[0]
    row = lambda i: data[min(max(i, 0), N - 1), 1:].copy()
    frame0 = row(0)

    do_front = args.front or not (args.front or args.back)
    do_back = args.back or not (args.front or args.back)

    fl = args.front_fade_len if args.front_fade_len is not None else args.fade_len
    bl = args.back_fade_len if args.back_fade_len is not None else args.fade_len
    bfl = args.back_fast_fade_len  # None -> uniform back fade
    fast_cols = []
    if bfl is not None:
        names = [s.strip() for s in args.back_fast_joints.split(",") if s.strip()]
        for nm in names:
            if nm not in CSV_JOINTS:
                sys.exit(f"error: --back-fast-joints unknown joint {nm!r}")
            fast_cols.append(7 + CSV_JOINTS.index(nm))

    # validation
    if do_back and args.back_fade_start is None:
        sys.exit("error: --back requires --back-fade-start (frame where the fade to walkready begins)")
    if do_front and not (1 <= fl <= N - 1):
        sys.exit(f"error: front fade length {fl} out of range for {N} frames")
    S = args.back_fade_start
    if do_back:
        if not (0 <= S <= N - 1):
            sys.exit(f"error: --back-fade-start {S} out of range [0, {N - 1}]")
        if do_front and S < fl + 1:
            sys.exit(f"error: --back-fade-start ({S}) overlaps the front fade (needs >= {fl + 1})")
        if S + bl > N - 1:
            print(f"[warn] back-fade-start+back-fade-len = {S + bl} exceeds last frame {N - 1}; "
                  f"clamping (tail held).")

    # poses
    wr_start = make_walkready(frame0, joint_only=args.joint_only)
    wr_end = None
    if do_back:
        oe = row(S + bl)
        wr_end = make_walkready(oe, joint_only=args.joint_only)

    seq = []
    # ---- front ----
    if do_front:
        seq += [wr_start] * args.hold_front
        for k in range(0, fl + 1):                       # walkready -> continuing motion
            seq.append(blend(wr_start, row(k), smoothstep(k / fl)))
        pure_start = fl + 1
    else:
        pure_start = 0
    # ---- pure original ----
    pure_end = (S - 1) if do_back else (N - 1)
    for i in range(pure_start, pure_end + 1):
        seq.append(row(i))
    # ---- back ----
    if do_back:
        for i in range(S, S + bl + 1):                   # motion -> walkready (tail dropped)
            k = i - S
            w_body = smoothstep(k / bl)
            if bfl is not None:
                seq.append(fade_blend(row(i), wr_end, w_body, smoothstep(k / bfl), fast_cols))
            else:
                seq.append(blend(row(i), wr_end, w_body))
        seq += [wr_end] * args.hold_back

    meta = dict(N=N, do_front=do_front, do_back=do_back, fl=fl, bl=bl, bfl=bfl, S=S,
                fast_cols=fast_cols, offset=(args.hold_front if do_front else 0),
                wr_start=wr_start, wr_end=wr_end)
    return np.array(seq), meta


def write_csv(path, header, seq):
    out = np.column_stack([np.arange(len(seq)), seq])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in out:
            w.writerow([int(r[0])] + [f"{v:.6f}" for v in r[1:]])


def verify(seq, data, meta, args, header):
    c = {n: i for i, n in enumerate(header)}
    jc = list(range(c["l_hip_pitch"], c["r_elbow"] + 1))
    out = np.column_stack([np.arange(len(seq)), seq])

    def to_sim(rj):
        return {j: (-rj[k] if j in INV else rj[k]) for k, j in enumerate(CSV_JOINTS)}

    def legs_ok(simvals):
        return all(abs(simvals[j] - t) < 1e-9 for j, t in zip(_VERIFY_ORDER, args.walkready_state))

    print(f"  frames: {len(seq)}  (NaN: {bool(np.isnan(seq).any())})")
    if meta["do_front"]:
        print(f"  hold-start legs == walkready_state: {legs_ok(to_sim(seq[0][7:]))}")
    if meta["do_back"]:
        print(f"  hold-end   legs == walkready_state: {legs_ok(to_sim(seq[-1][7:]))}")

    def handoff(orig_f, label):
        idx = meta["offset"] + orig_f
        if 2 <= idx < len(out) - 2:
            s = [np.abs(out[i + 1, jc] - out[i, jc]).sum() for i in range(idx - 2, idx + 3)]
            print(f"  continuity @ {label} (orig f{orig_f}): " + " ".join(f"{v:.2f}" for v in s))

    if meta["do_front"]:
        handoff(meta["fl"], "front->motion ")
    if meta["do_back"]:
        handoff(meta["S"], "motion->back  ")

    synth = []
    if meta["do_front"]:
        synth += list(range(0, meta["offset"] + meta["fl"]))
    if meta["do_back"]:
        synth += list(range(meta["offset"] + meta["S"], len(out) - 1))
    if synth:
        mx = max(np.abs(out[i + 1, jc] - out[i, jc]).max() for i in synth)
        print(f"  max synthetic joint step: {mx:.3f} rad/frame (limit ~0.53 @30fps)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Bracket a pi_plus retargeting CSV with a velocity-continuous walkready pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="input motion CSV (GMR/retargeting convention)")
    p.add_argument("--front", action="store_true", help="fade walkready in at the start")
    p.add_argument("--back", action="store_true", help="fade the motion out into walkready at the end")
    p.add_argument("--back-fade-start", type=int, default=None,
                   help="original frame where the back fade begins; frames after "
                        "(start + back-fade-len) are discarded. REQUIRED with --back")
    p.add_argument("--fade-len", type=int, default=36, help="default fade length (frames) for both ends")
    p.add_argument("--front-fade-len", type=int, default=None, help="override front fade length")
    p.add_argument("--back-fade-len", type=int, default=None, help="override back fade length")
    p.add_argument("--back-fast-fade-len", type=int, default=None,
                   help="faster fade length for selected back joints (e.g. a broken IK arm)")
    p.add_argument("--back-fast-joints", type=str, default=",".join(DEFAULT_FAST_JOINTS),
                   help="comma-separated joints for --back-fast-fade-len")
    p.add_argument("--joint-only", action="store_true",
                   help="keep the source frame's root pose (position + orientation); only the "
                        "joints fade to walkready. Use for non-upright motions (e.g. supine).")
    p.add_argument("--hold-front", type=int, default=6, help="static walkready frames at the very start")
    p.add_argument("--hold-back", type=int, default=30, help="static walkready frames at the very end")
    p.add_argument("--output", type=str, default=None, help="output path (default: <input>_walkready.csv)")
    args = p.parse_args(argv)
    # default both if neither flag given
    if not args.front and not args.back:
        pass  # build() treats "neither" as "both"
    args.walkready_state = [WALKREADY_SIM[j] for j in _VERIFY_ORDER]
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        out_path = base + "_walkready" + ext

    header, data = load_csv(args.input)
    seq, meta = build(header, data, args)
    write_csv(out_path, header, seq)

    pure = (meta["S"] if meta["do_back"] else meta["N"]) - (meta["fl"] + 1 if meta["do_front"] else 0)
    parts = []
    if meta["do_front"]:
        parts.append(f"hold{args.hold_front}+fadein{meta['fl'] + 1}")
    parts.append(f"cart{pure}")
    if meta["do_back"]:
        parts.append(f"endfade{meta['bl'] + 1}+hold{args.hold_back}")
    print(f"wrote {out_path}  ({'+'.join(parts)})")
    verify(seq, data, meta, args, header)


if __name__ == "__main__":
    main()
