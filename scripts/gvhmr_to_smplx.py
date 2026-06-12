"""Convert GVHMR demo output (hmr4d_results.pt) into an AMASS-style SMPL-X .npz
that scripts/smplx_to_pi_plus.py (load_smplx_file) can consume.

GVHMR's `smpl_params_global` is already SMPL-X (it uses the "supermotion" model),
so the body pose maps 1:1 to AMASS `pose_body`. The only non-trivial part is the
world up-axis: AMASS/SMPL-X is z-up, GVHMR's gravity-aligned world frame may differ
(often y-up). Use --rot_axis/--rot_deg to align it; verify in the viewer:

    # 1) convert (no rotation first)
    python scripts/gvhmr_to_smplx.py GVHMR/outputs/demo/tennis/hmr4d_results.pt \
        -o source/motion/gvhmr/tennis.npz

    # 2) check orientation in the viewer (no --save_path)
    python scripts/smplx_to_pi_plus.py --smplx_file source/motion/gvhmr/tennis.npz

    # 3) if the figure lies down / is tilted, e.g. y-up -> z-up:
    python scripts/gvhmr_to_smplx.py GVHMR/outputs/demo/tennis/hmr4d_results.pt \
        -o source/motion/gvhmr/tennis.npz --rot_axis x --rot_deg 90
"""

import argparse
import pathlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from rich import print

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to GVHMR hmr4d_results.pt")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output .npz path")
    parser.add_argument(
        "--frame",
        choices=["global", "incam"],
        default="global",
        help="Which GVHMR frame to export (global = world-grounded; use that for trajectory).",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Source video fps -> mocap_frame_rate.")
    parser.add_argument("--gender", default="neutral", help="SMPL-X gender (matches your body model).")
    parser.add_argument(
        "--rot_axis", choices=["x", "y", "z"], default="x", help="Axis of the world up-alignment rotation."
    )
    parser.add_argument(
        "--rot_deg",
        type=float,
        default=0.0,
        help="Degrees to rotate the whole motion about --rot_axis (0 = none). E.g. 90 for y-up -> z-up.",
    )
    args = parser.parse_args()

    key = f"smpl_params_{args.frame}"

    data = torch.load(args.input, map_location="cpu")
    if key not in data:
        raise KeyError(f"'{key}' not found in {args.input}. Available: {list(data.keys())}")
    p = data[key]

    # GVHMR SMPL-X params -> numpy
    global_orient = p["global_orient"].detach().cpu().numpy().astype(np.float64)  # (N, 3) axis-angle
    body_pose = p["body_pose"].detach().cpu().numpy().astype(np.float32)           # (N, 63)
    transl = p["transl"].detach().cpu().numpy().astype(np.float64)                 # (N, 3)
    betas = p["betas"].detach().cpu().numpy().astype(np.float32)                   # (N, 10)

    num_frames = body_pose.shape[0]
    print(f"[bold]Loaded[/bold] {key}: {num_frames} frames, body_pose {body_pose.shape}, betas {betas.shape}")

    # World up-axis alignment: rotate trans (vectors) and global_orient (orientation),
    # leaving body_pose untouched (it is relative to the root).
    if args.rot_deg != 0.0:
        R_align = R.from_euler(args.rot_axis, args.rot_deg, degrees=True)
        transl = R_align.apply(transl)
        global_orient = (R_align * R.from_rotvec(global_orient)).as_rotvec()
        print(f"[yellow]Applied world rotation[/yellow]: {args.rot_deg}deg about {args.rot_axis}")
    else:
        print("[dim]No world rotation applied (use --rot_deg/--rot_axis if the figure is tilted/lying down).[/dim]")

    # betas: GVHMR gives (N, 10) ~constant -> mean over frames -> (10,).
    # The SMPL-X model used by smplx_to_pi_plus.py has num_betas=16 (AMASS convention),
    # so zero-pad the 10 GVHMR betas to 16 (the extra coeffs carry no shape info).
    NUM_BETAS = 16
    betas_mean = betas.mean(axis=0).astype(np.float32)  # (10,)
    if betas_mean.shape[0] < NUM_BETAS:
        betas_mean = np.concatenate(
            [betas_mean, np.zeros(NUM_BETAS - betas_mean.shape[0], dtype=np.float32)]
        )  # (16,)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        gender=args.gender,
        betas=betas_mean,                                # (10,)
        root_orient=global_orient.astype(np.float32),    # (N, 3)
        pose_body=body_pose,                             # (N, 63)
        trans=transl.astype(np.float32),                 # (N, 3)
        mocap_frame_rate=np.float64(args.fps),
    )
    print(f"[green]Saved[/green] {out_path}  (fps={args.fps}, gender={args.gender})")
