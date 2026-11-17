#!/usr/bin/env python3
"""Render the cylinder grasp to a GIF, with the tactile readings beside it.

Usage:
    python3 examples/record_grasp.py
    python3 examples/record_grasp.py --side left
    python3 examples/record_grasp.py --fps 30 --width 960 --height 720

Mirrors ``mjcf/examples/record_grasp.py``: each frame is the scene from a fixed
camera, with the pressure maps drawn in the right column in the shape of the
hand. Frames are rendered off-screen with an Isaac Lab camera sensor and written
as an animated GIF with Pillow.

One thing differs from the MuJoCo script: the camera is a scene sensor
(``CameraCfg``) rather than a viewer camera, because off-screen capture through
``/OmniverseKit_Persp`` needs a viewport that a headless run does not have.

The red force arrows are drawn here rather than with Kit's debug-draw overlay,
which is a viewport layer and never reaches a sensor render. The tactile points
are projected into the camera and the shafts drawn into the frame, with the same
length scale and colour as the interactive viewer.

Dependencies: isaaclab, isaacsim, numpy, matplotlib, pillow.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))

# grasp_cylinder parses argv at import time, so hand it the flags it needs
# before importing it. Our own options are parsed first, then dropped.
_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--side", choices=("left", "right"), default="right")
_parser.add_argument("--fps", type=float, default=20.0)
_parser.add_argument("--width", type=int, default=640)
_parser.add_argument("--height", type=int, default=480)
_parser.add_argument("--out", type=Path, default=None,
                     help="GIF path (default: examples/grasp_<side>.gif)")
args = _parser.parse_args()

sys.argv = [sys.argv[0], "--side", args.side, "--tactiles", "--viz", "none"]

import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402
import PIL.ImageDraw  # noqa: E402

import grasp_cylinder as gc  # noqa: E402
import grasp_trajectory as traj  # noqa: E402
import hand as hand_lib  # noqa: E402
from quiet import silence_settings_warnings  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass, instantiate  # noqa: E402

from tactile_viewer import arrow_length  # noqa: E402

# apex_hand/mjcf/examples/lib, for the panel layout the MuJoCo demo uses. The
# USD examples/lib has a module of the same name, so load this one by path.
_MJCF_LIB = Path(__file__).resolve().parents[2] / "mjcf" / "examples" / "lib"

# Fixed camera, looking at the grasp from the front-left so both the palm-side
# fingers and the thumb are visible. Same framing as the MuJoCo script.
CAM_EYE = (0.55, -0.45, 0.35)
CAM_TARGET = (0.05, 0.0, 0.15)


def _load_mjcf_panel():
    """The MuJoCo demo's TactilePanel, imported by path to avoid a name clash."""
    spec = importlib.util.spec_from_file_location(
        "mjcf_tactile_viewer", _MJCF_LIB / "tactile_viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tactile_figure(panel_module, side, maps):
    """Draw the pressure maps the way TactilePanel lays them out, to an array.

    Reuses the MuJoCo panel's placement so the GIF and the interactive window
    cannot drift apart; only the backend differs (Agg, no window).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.0, 8.0), dpi=100)
    panel = panel_module.TactilePanel.__new__(panel_module.TactilePanel)
    panel.side = side
    panel.fig = fig
    panel.ax = fig.add_subplot(111)
    panel.ax.set_aspect("equal")
    panel.ax.axis("off")
    panel._artists = {}
    panel._placed = False
    panel._place(maps)
    fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=panel_module.CMAP,
            norm=plt.Normalize(panel_module.KPA_MIN, panel_module.KPA_MAX)),
        ax=panel.ax, fraction=0.04, pad=0.02).set_label("kPa")
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return image


def _compose(frame, panel):
    """Put the rendered scene and the pressure maps side by side."""
    scene = PIL.Image.fromarray(frame)
    pad = PIL.Image.fromarray(panel)
    # Match heights so the two panels sit in one band.
    width = int(pad.width * scene.height / pad.height)
    pad = pad.resize((width, scene.height), PIL.Image.LANCZOS)
    out = PIL.Image.new("RGB", (scene.width + pad.width, scene.height))
    out.paste(scene, (0, 0))
    out.paste(pad, (scene.width, 0))
    return out


def _camera_rays(camera):
    """Eye, world-to-camera rotation and intrinsics, for projecting points.

    The pose is the OpenGL convention (``quat_w_opengl``, ``(x, y, z, w)``):
    forward is -Z, up is +Y, so a point's camera-plane distance is ``-z``.
    """
    eye = camera.data.pos_w.torch[0].cpu().numpy().astype(float)
    qx, qy, qz, qw = camera.data.quat_w_opengl.torch[0].cpu().numpy()
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
    intrinsic = camera.data.intrinsic_matrices.torch[0].cpu().numpy().astype(float)
    return eye, rotation, intrinsic


def _draw_arrows(image, depth, tactiles, camera, length_of, samples=40):
    """Draw the per-point force arrows into an RGB frame, MuJoCo-style.

    Start at the sphere surface, extend along the outward normal, with the
    length from ``length_of`` (the same sqrt scale the viewport uses, so the
    weak points stay visible). The viewport arrows point the other way (into
    the hand, which is the direction the object pushes); here the shafts are
    drawn outward so they leave the silhouette instead of vanishing inside the
    palm, as the MuJoCo demo does.

    The arrows are geometry in the MuJoCo demo, so the hand and the object hide
    the ones behind them. A flat overlay has no depth, so each shaft is sampled
    along its length and only the samples nearer than the rendered surface are
    drawn -- otherwise every arrow shows through the palm.
    """
    pixels = np.asarray(image)
    height, width = pixels.shape[:2]
    eye, rotation, intrinsic = _camera_rays(camera)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    surface = np.asarray(depth).reshape(height, width)

    starts, ends = [], []
    for pad, forces in tactiles.forces.items():
        points = tactiles.point_positions.get(pad)
        normals = tactiles.point_normals.get(pad)
        if points is None or normals is None:
            continue
        for index in np.nonzero(forces > 0.0)[0]:
            starts.append(points[index])
            ends.append(points[index] + length_of(forces[index]) * normals[index])
    if not starts:
        return pixels

    # One row per shaft, sampled at ``samples`` points from start to tip.
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    fractions = np.linspace(0.0, 1.0, samples)[None, :, None]
    cloud = starts[:, None, :] + fractions * (ends - starts)[:, None, :]
    local = (cloud - eye) @ rotation
    distance = -local[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * local[:, :, 0] / distance + cx
        v = -fy * local[:, :, 1] / distance + cy
    u = np.rint(u).astype(int)
    v = np.rint(v).astype(int)
    inside = ((distance > 1e-4) & (u >= 0) & (u < width)
              & (v >= 0) & (v < height))
    visible = np.zeros_like(inside)
    visible[inside] = distance[inside] < surface[v[inside], u[inside]] - 2e-3

    canvas = PIL.Image.fromarray(pixels)
    draw = PIL.ImageDraw.Draw(canvas)
    colour = (220, 30, 30)
    for row in range(len(starts)):
        run_start = None
        for column in range(samples):
            if not visible[row, column]:
                run_start = None
                continue
            if run_start is None:
                run_start = column
            if column + 1 == samples or not visible[row, column + 1]:
                if column > run_start:      # a lone visible sample is a dot
                    draw.line([(u[row, run_start], v[row, run_start]),
                               (u[row, column], v[row, column])],
                              fill=colour, width=3)
                run_start = None
    return np.asarray(canvas)


def _scene_cfg(ox, oy, width, height):
    """The grasp scene plus one camera aimed at the grasp point."""

    @configclass
    class RecordSceneCfg(gc.SceneCfg):
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=height,
            width=width,
            # Depth is only for hiding arrow shafts that are behind geometry.
            data_types=["rgb", "distance_to_image_plane"],
            # No offset: the pose is set once from the eye/target pair below.
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.01, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0),
                                       rot=(0.0, 0.0, 0.0, 1.0),
                                       convention="world"),
        )

    return RecordSceneCfg


def main():
    silence_settings_warnings()
    gc.args_cli.visualizer = []
    gc.args_cli.enable_cameras = True
    out_path = args.out or (HERE / f"grasp_{args.side}.gif")

    dt = float(gc.MANIFEST["timestep"])
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device="cuda:0", physics=PhysxCfg())

    from isaaclab.app import launch_simulation
    with launch_simulation(sim_cfg, gc.args_cli):
        import torch

        from tactiles import RobotTactiles

        sim = sim_utils.SimulationContext(sim_cfg)
        ox, oy = traj.SCENE_XY
        scene = instantiate(_scene_cfg(ox, oy, args.width, args.height)(
            num_envs=1, env_spacing=1.0))
        sim.reset()
        dt = sim.get_physics_dt()

        robot = scene["robot"]
        cylinder = scene["cylinder"]
        hand_lib.place_hand(robot, traj.PARK, traj.CONTACT_QUAT, args.side)

        camera = scene["camera"]
        camera.set_world_poses_from_view(
            torch.tensor([[ox + CAM_EYE[0], oy + CAM_EYE[1], CAM_EYE[2]]]),
            torch.tensor([[ox + CAM_TARGET[0], oy + CAM_TARGET[1],
                           CAM_TARGET[2]]]))

        tactiles = RobotTactiles(gc.MANIFEST, scene["contact"], robot, args.side,
                                 verbose=False)
        panel_module = _load_mjcf_panel()

        names = robot.joint_names
        limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
        grasp = traj.Grasp(limits, names, dt)
        total = (traj.MOVE_TIME + traj.SETTLE_TIME + traj.CLOSE_TIME
                 + traj.SETTLE_TIME + traj.LIFT_TIME + traj.HOLD_TIME)
        steps = int(round(total / dt))
        # Capture at --fps, independent of the physics rate.
        steps_per_frame = max(1, int(round((1.0 / args.fps) / dt)))

        print(f"[record] {args.side} hand, {steps * dt:.1f} s at "
              f"{dt * 1000:.1f} ms, capturing every {steps_per_frame} steps "
              f"({args.fps:g} fps)")

        frames = []
        start_z = float(cylinder.data.root_pos_w.torch[0, 2])
        for step in range(steps):
            hand_lib.drive_step(robot, scene, sim, grasp, args.side,
                                traj.CONTACT_QUAT, dt)
            tactiles.update(robot)

            if step % steps_per_frame:
                continue
            image = camera.data.output["rgb"]
            frame = (image.torch[0].cpu().numpy() if hasattr(image, "torch")
                     else np.asarray(image))
            depth = camera.data.output["distance_to_image_plane"]
            depth = (depth.torch[0].cpu().numpy() if hasattr(depth, "torch")
                     else np.asarray(depth))
            frame = _draw_arrows(frame, depth, tactiles, camera, arrow_length)
            panel = _tactile_figure(panel_module, args.side,
                                    tactiles.pressure_kpa)
            frames.append(_compose(frame, panel))
            if len(frames) % 20 == 0:
                height = float(cylinder.data.root_pos_w.torch[0, 2])
                print(f"[record] frame {len(frames)} (t={step * dt:.2f}s, "
                      f"cylinder {(height - start_z) * 1000:+.0f} mm)")

        if not frames:
            raise SystemExit("[record] no frames captured")

        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0, optimize=True)
        height = float(cylinder.data.root_pos_w.torch[0, 2])
        print(f"[record] saved {out_path} ({len(frames)} frames, "
              f"cylinder rose {(height - start_z) * 1000:+.0f} mm)")


if __name__ == "__main__":
    main()
