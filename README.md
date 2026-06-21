# Mini Pi Plus based on BeyondMimic

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab/v2.2.0/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Pixi](https://img.shields.io/badge/pixi-managed-orange.svg)](https://pixi.sh)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

![gif](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/6363667a0f27da450e1059a30c2b274b.gif)

## Introduction

This project is built on top of [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking),
a versatile humanoid motion-control framework by qiayuanl et al. that provides highly dynamic motion
tracking with state-of-the-art motion quality for real-world deployment.

This repo covers motion-tracking training, ships assets for HighTorque / Bit-Bots `pi_plus` robots,
and tunes the configuration for them. You can train sim-to-real motions from the datasets in
`source/motion` without changing any parameters.

## Installation

This project is managed with [Pixi](https://pixi.sh), which pins every dependency (CUDA, PyTorch,
Isaac Sim, Isaac Lab, rsl_rl, …) in `pixi.toml` / `pixi.lock` so the environment is reproducible.
You do **not** need Conda.

### Prerequisites

1. **Install Pixi** (if you don't have it):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```
2. **Clone Isaac Lab as a sibling directory.** This repo references Isaac Lab via editable paths
   (`../IsaacLab/source/...`), so the two repos must sit next to each other:
   ```text
   <parent>/
   ├── IsaacLab/      # github.com/isaac-sim/IsaacLab @ v2.2.0
   └── Mini-Pi-Plus_BeyondMimic/   # this repo
   ```
   ```bash
   git clone https://github.com/bit-bots/BeyondMimic.git Mini-Pi-Plus_BeyondMimic
   git clone -b v2.2.0 https://github.com/isaac-sim/IsaacLab.git
   ```

### Set up the environment

From the repo root:

```bash
cd Mini-Pi-Plus_BeyondMimic
pixi install          # resolves + installs the default environment (Isaac Sim, Isaac Lab, torch, …)
```

This pulls Isaac Sim 5.0.0, Isaac Lab 2.2.0 and rsl_rl 2.3.1 automatically — no manual Isaac Sim
download required. The local `whole_body_tracking` package is installed editable as part of the env.

### Activate the environment

To activate the workspace, run the following in the terminal you want to use:

```bash
pixi shell            # default environment
pixi shell -e 5090    # Blackwell GPUs (RTX 50xx)
pixi shell -e gmr     # GMR retargeting environment
pixi shell -e gvhmr   # GVHMR video-to-motion environment
```

Once the shell is active you can run the commands below **without** the `pixi run` prefix
(just `python scripts/...`). Alternatively, run a single command inside the workspace without
activating the shell by prefixing it with `pixi run`:

```bash
pixi run python scripts/rsl_rl/train.py ...
pixi run -e 5090 python scripts/rsl_rl/train.py ...
```

To list the predefined task shortcuts, run `pixi task list`.

The examples in this README use the `pixi run` prefix; drop it if you are already inside `pixi shell`.

### Environments

The workspace defines four Pixi environments. Pick the one that matches your task / GPU
(select it via `pixi shell -e <env>` or `pixi run -e <env> ...`):

| Environment | Flag        | Purpose |
|-------------|-------------|---------|
| `default`   | *(none)*    | Training / play / sim2sim. PyTorch on **CUDA 12.6** — works on RTX 40xx and older. |
| `5090`      | `-e 5090`   | Same as `default` but PyTorch on **CUDA 12.8** for Blackwell GPUs (RTX 50xx, `sm_120`). Use this if you see `no kernel image available` / `sm_120 is not compatible`. |
| `gmr`       | `-e gmr`    | Isolated Python 3.10 env for GMR motion retargeting only (see below). |
| `gvhmr`     | `-e gvhmr`  | Isolated Python 3.10 env (**CUDA 12.1**) for GVHMR video-to-motion extraction (see below). Needs a cu121-capable GPU (`sm_<=90`, e.g. RTX 40xx). |

## Motion Tracking

### Obtain Data & GMR Retargeting Data

Use the GMR project for dataset retargeting (original project: https://github.com/YanjieZe/GMR;
a copy is vendored in `GMR/`). Retargeting runs in the dedicated `gmr` environment.

> If you use a GMR checkout from the original link instead of the vendored `GMR/`,
> set `numpy==1.24.4` in its `setup.py` before installing.

> To make HighTorque robots easy to use, CSV-format retargeted data and converted NPZ templates
> are provided in `source/motion`. If you need to retarget other files, follow the steps below.

### Motion Preprocessing & Registry Setup

```bash
# Retargeting (gmr env)
pixi run -e gmr python scripts/bvh_to_robot.py --bvh_file GMR/MotionData/lafan1/{xxx}.bvh --robot pi_football --save_path GMR/RetargetData/lafan1/csv/pi_plus/{xxx}.csv --rate_limit

# Retargeting on a headless machine (no display / no X11).
# bvh_to_robot.py always opens the MuJoCo GUI viewer, which fails on a headless
# host with: GLFWError "X11: The DISPLAY environment variable is missing".
# Use the headless variant instead — same CSV output (column reorder, wxyz->xyzw,
# --keep_wrist), but no viewer/rendering. --save_path is required; --rate_limit is
# accepted but ignored.
pixi run -e gmr python scripts/bvh_to_robot_headless.py --bvh_file GMR/MotionData/lafan1/{xxx}.bvh --robot pi_football --save_path GMR/RetargetData/lafan1/csv/pi_plus/{xxx}.csv

# Trimming
pixi run python scripts/csv_cut_pi_plus.py --input_csv GMR/RetargetData/lafan1/csv/pi_plus/pi_plus_dance1_subject2.csv --output_csv GMR/RetargetData/lafan1/csv/pi_plus/{xxx}.csv --start_frame {number} --end_frame {number} --remove_frame_column --z_offset 0.00 --decimal_places 6

# NPZ format conversion (add --headless to skip the graphical interface)
# --input_file is the trimmed CSV from the previous step.
pixi run python scripts/csv_to_npz.py --robot pi_plus --input_file GMR/RetargetData/lafan1/csv/pi_plus/{xxx}.csv --input_fps 30 --output_name source/motion/hightorque/pi_plus/npz/{motion_name}

# Data playback (interactive viewer; needs a display)
pixi run python scripts/replay_npz.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz

# Data playback -> render to mp4 (works headless)
pixi run python scripts/replay_npz.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --headless --video
```

`replay_npz.py` can render the motion to an mp4 instead of (or in addition to) the live
viewer. This is the only way to inspect a motion on a headless host (no display), since the
interactive viewer needs an X display. Video flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--video` | off | Enable recording (implicitly sets `--enable_cameras`). |
| `--video_path PATH` | `<motion_file>.mp4` | Output mp4 path. If omitted, written next to the NPZ with the same name. |
| `--video_length N` | `0` | Number of frames to record. `0` records exactly one full motion loop. |
| `--headless` | off | Run without opening a Kit window. Required on a headless host. |

The output is a fixed 1280x720 H.264 mp4 (via `imageio` + ffmpeg) at `1 / sim_dt` fps (50 fps). On a
headless host the file cannot be displayed in place — copy it off the machine (e.g. `scp`) to view it.

### Motion from Video (GVHMR)

Instead of retargeting an existing mocap dataset (BVH/lafan1), you can extract motion **from a single
monocular RGB video** with [GVHMR](https://github.com/zju3dv/GVHMR) (vendored in `GVHMR/`) and feed it
into the same pi_plus pipeline. Video pose estimation runs in the dedicated `gvhmr` environment; the
later retarget/trim/convert steps reuse the `gmr` and `default` envs.

**One-time setup** (model files are license-gated and *not* managed by Pixi):

1. Download the pretrained checkpoints (GVHMR, HMR2, ViTPose, YOLO) into `GVHMR/inputs/checkpoints/`
   and the SMPL-X / SMPL body models into `GVHMR/inputs/checkpoints/body_models/{smplx,smpl}/`
   — see `GVHMR/docs/INSTALL.md` (register at smpl-x.is.tue.mpg.de and smpl.is.tue.mpg.de).
2. The retarget step also needs the SMPL-X model at `assets/body_models/smplx/SMPLX_NEUTRAL.npz`.
   Symlink it to the GVHMR copy (avoids a 100 MB duplicate; `assets/body_models/` is gitignored and
   the license-gated files must not be committed):
   ```bash
   mkdir -p assets/body_models/smplx
   ln -s "$(pwd)/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" assets/body_models/smplx/SMPLX_NEUTRAL.npz
   ```

**Pipeline** (example name `tennis`):

```bash
# 1) Video -> GVHMR (SMPL-X .pt)  [gvhmr env]
#    The gvhmr-demo task runs with cwd=GVHMR; the --video path is relative to GVHMR/.
pixi run -e gvhmr gvhmr-demo --video docs/example_video/tennis.mp4
#   -> GVHMR/outputs/demo/tennis/hmr4d_results.pt

# 2) Bridge: GVHMR .pt -> AMASS-style SMPL-X .npz  [gmr env]
#    GVHMR's world frame is y-up, AMASS/SMPL-X is z-up -> rotate +90 about X.
pixi run -e gmr python scripts/gvhmr_to_smplx.py GVHMR/outputs/demo/tennis/hmr4d_results.pt --output RetargetData/gvhmr/smplx/tennis.npz --rot_axis x --rot_deg 90

# 3) Retarget: SMPL-X .npz -> pi_plus CSV (20 DOF)  [gmr env]
#    smplx_to_pi_plus.py opens the MuJoCo GUI viewer by default, which fails on a
#    headless host with: "Could not initialize GLFW". Add --headless to skip the
#    viewer entirely (no display / no X11 / no xvfb needed) — same CSV output.
pixi run -e gmr python scripts/smplx_to_pi_plus.py --smplx_file RetargetData/gvhmr/smplx/tennis.npz --save_path RetargetData/gvhmr/csv/pi_plus/tennis.csv --headless

# 4) Trim / ground: z-offset drops the ~5 cm float (optional frame range)  [default env]
pixi run python scripts/csv_cut_pi_plus.py --input_csv RetargetData/gvhmr/csv/pi_plus/tennis.csv --output_csv RetargetData/gvhmr/csv/pi_plus/tennis_cut.csv --z_offset -0.05
#   optional: --start_frame {n} --end_frame {m}

# 5) NPZ conversion -> training-ready motion  [default env]
pixi run python scripts/csv_to_npz.py --robot pi_plus --input_file RetargetData/gvhmr/csv/pi_plus/tennis_cut.csv --input_fps 30 --output_name source/motion/hightorque/pi_plus/npz/tennis
```

From step 5 on, the `.npz` is identical to a lafan1-derived motion — replay / train / evaluate it
with the same commands as above.

> **Storage layout.** GVHMR artifacts mirror the lafan1 convention: the (robot-agnostic) SMPL-X bridge
> output goes to `RetargetData/gvhmr/smplx/`, the pi_plus CSVs (raw + `_cut`) to
> `RetargetData/gvhmr/csv/pi_plus/`, and the final training NPZ to `source/motion/hightorque/pi_plus/npz/`.

**Notes:**
- The y-up→z-up rotation (`--rot_axis x --rot_deg 90`) is GVHMR-specific. Verify the figure stands
  upright by running step 3 without `--save_path` and without `--headless` (opens the viewer); if it
  lies down / is upside down, adjust `--rot_deg`. This visual check needs a display — on a headless
  host either retarget with `--headless` and inspect the result after step 5 via `replay_npz.py --headless --video`.
- For a moving/handheld camera GVHMR uses its default SimpleVO. `--static_cam` (tripod) and
  `--use_dpvo` (needs compiling the optional DPVO submodule) are alternatives — see `tools/demo/demo.py`.
- `csv_to_npz.py` applies the pi_plus URDF axis inversions automatically — no manual sign flips needed.

### Model Training

> **Keep the task variant consistent.** The `-Wo-v0` task drops two observation terms
> (`motion_anchor_pos_b` + `base_lin_vel`), so its policy takes a **109-dim** observation input,
> whereas the full `Tracking-Flat-PI-Plus-v0` task uses **115**. A checkpoint must be played,
> exported and sim-to-sim evaluated with the **same** task it was trained with — mixing them up
> produces a tensor-shape mismatch (e.g. `[N, 115]` vs `[N, 109]`).

Train the policy with the following command:

```bash
pixi run python scripts/rsl_rl/train.py --task=Tracking-Flat-PI-Plus-Wo-v0 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --headless --log_project_name pi_plus_beyondmimic
# Remove --logger wandb if you don't want to use wandb
# Resume training with --resume {load_run_name}
# On a Blackwell GPU (RTX 50xx): pixi run -e 5090 python scripts/rsl_rl/train.py ...

# Other available arguments:
#   --save_interval=10
#   --experiment_name={experiment_name}
#   --log_dir_path={log_dir_path}
# If --experiment_name and --run_name are given, logs are saved to:
#   logs/rsl_rl/{experiment_name}/{%Y-%m-%d_%H-%M-%S}_{run_name}
# If --log_dir_path is given, logs are saved there instead.
```

### Model Export

Play the trained policy with the following command:

```bash
pixi run python scripts/rsl_rl/play.py --task=Tracking-Flat-PI-Plus-Wo-v0 --checkpoint {logs_path_to}/model_xxx.pt --num_envs=1 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz
```

#### Headless video rendering

To render a video of the policy on a headless host (no display), add `--headless --video`.
Passing `--video` implicitly enables cameras, so `--enable_cameras` is not needed:

```bash
pixi run python scripts/rsl_rl/play.py --task=Tracking-Flat-PI-Plus-Wo-v0 --checkpoint {logs_path_to}/model_xxx.pt --num_envs=1 --headless --video --video_length 200 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz
```

The mp4 is written to `<checkpoint_dir>/videos/play/` and is named after the checkpoint
(e.g. `model_xxx-step-0.mp4`), so different checkpoints don't overwrite each other.
`--video_length` is the number of steps to record; the play loop exits once that many
steps are reached.

By default the reference-trajectory coordinate frames are **not** drawn, so they don't
overlay the motion. Add `--viz_trajectory` to show them:

```bash
pixi run python scripts/rsl_rl/play.py --task=Tracking-Flat-PI-Plus-Wo-v0 --checkpoint {logs_path_to}/model_xxx.pt --num_envs=1 --viz_trajectory --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz
```

![if](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/e7faf89fbdbf87cf909bbf81ceeb1a7f.gif)

### Model Evaluation

Validate the policy in MuJoCo with the following command:

```bash
pixi run python scripts/sim2sim.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --xml_path source/whole_body_tracking/whole_body_tracking/assets/hightorque/pi_plus/mjcf/pi_20dof.xml --policy_path {logs_path_to}/exported/{model_xxx}.onnx --save_json --loop
# Use --loop to play the policy repeatedly
```

![f](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/de78f3ab232911f9a93e936cb5463164.gif)

## Code Structure

The following is an overview of the project's code structure:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains atomic functions for defining BeyondMimic's MDP:
    - **`commands.py`** — command library for computing variables from reference motion, current robot state and errors (pose/velocity error computation, initial-state randomization, adaptive sampling).
    - **`rewards.py`** — DeepMimic reward functions and smoothing terms.
    - **`events.py`** — domain-randomization terms.
    - **`observations.py`** — observation terms for motion tracking and data collection.
    - **`terminations.py`** — early termination and timeouts.
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Environment (MDP) hyperparameter configurations for tracking tasks.
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/*/agents/rsl_rl_ppo_cfg.py`**
  PPO hyperparameters for tracking tasks.
- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Robot-specific settings (skeleton parameters, joint stiffness/damping, action scaling).
- **`scripts`**
  Utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

## Acknowledgements

This repository builds on the following upstream projects:

- [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) by qiayuanl et al. (HybridRobotics) — the original motion-tracking framework.
- [HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic](https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic) — pi_plus assets and configuration this work is based on.

Licensed under MIT; see `LICENCE` for the original copyright notice.
