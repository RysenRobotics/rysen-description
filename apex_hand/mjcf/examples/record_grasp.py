#!/usr/bin/env python3
"""Render the cylinder grasp to a GIF, with the tactile readings beside it.

Usage:
    MUJOCO_GL=osmesa python3 examples/record_grasp.py
    python3 examples/record_grasp.py --side left
    python3 examples/record_grasp.py --fps 30 --width 960 --height 720

Runs from any working directory. Off-screen rendering needs an OpenGL backend;
in a container without a GPU that means ``MUJOCO_GL=osmesa``.

Each frame is the same thing the viewer shows -- the scene from a fixed camera,
plus the red tactile force arrows -- with the pressure maps drawn in the right
column in the shape of the hand. Frames are captured off-screen with
``mujoco.Renderer`` and written as an animated GIF with Pillow.

Dependencies: mujoco, numpy, matplotlib, pillow.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(EXAMPLES_DIR))
sys.path.insert(0, str(EXAMPLES_DIR / "lib"))
import grasp_cylinder as grasp  # noqa: E402
import tactile_viewer  # noqa: E402
from mj_tactiles import RobotTactiles  # noqa: E402

# Playback rate of the GIF. The trajectory is 9 s long, so 30 fps is ~270 frames.
FPS = 30
# Rendering size of the 3D panel. The pressure column is added at this height.
# 640x480 is the default offscreen framebuffer; asking for more needs an
# <visual><global offwidth=.../></visual> clause in the model.
WIDTH = 640
HEIGHT = 480

# Fixed camera, looking at the grasp from the front-left so both the palm-side
# fingers and the thumb are visible. Placed relative to the hand's contact point.
CAM_LOOKAT = (0.06, -0.09, 0.20)
CAM_DISTANCE = 0.44
CAM_AZIMUTH = 128.0
CAM_ELEVATION = -14.0

# Geom groups to draw: 0 (floor, cylinder), 1 (visual mesh) and 2 (translucent
# tactile pads).
RENDER_GROUPS = (0, 1, 2)


def _tactile_figure(side, maps, height_px):
    """Draw the pressure maps the way TactilePanel lays them out, to an array.

    Reuses TactilePanel's placement so the still frames and the interactive
    window cannot drift apart; only the backend differs (Agg, no window).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tactile_viewer import TactilePanel

    fig = plt.figure(figsize=(6.0, 8.0), dpi=100)
    panel = TactilePanel.__new__(TactilePanel)
    panel.side = side
    panel.fig = fig
    panel.ax = fig.add_subplot(111)
    panel.ax.set_aspect("equal")
    panel.ax.axis("off")
    panel._artists = {}
    panel._placed = False
    panel._place(maps)
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=tactile_viewer.CMAP,
                              norm=plt.Normalize(0.0, tactile_viewer.KPA_MAX)),
        ax=panel.ax, fraction=0.04, pad=0.02).set_label("kPa")
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return image


def _compose(frame, panel):
    """Put the rendered scene and the pressure maps side by side."""
    import PIL.Image

    scene = PIL.Image.fromarray(frame)
    pad = PIL.Image.fromarray(panel)
    # Match heights so the two panels sit in one band.
    width = int(pad.width * scene.height / pad.height)
    pad = pad.resize((width, scene.height), PIL.Image.LANCZOS)
    out = PIL.Image.new("RGB", (scene.width + pad.width, scene.height))
    out.paste(scene, (0, 0))
    out.paste(pad, (scene.width, 0))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--out", type=Path, default=None,
                        help="GIF path (default: examples/grasp_<side>.gif)")
    parser.add_argument("--stride", type=int, default=1,
                        help="capture every Nth frame, to make a quick preview")
    args = parser.parse_args()

    out_path = args.out or (EXAMPLES_DIR / f"grasp_{args.side}.gif")

    model = grasp.build_model(args.side, "mesh", tactiles=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Draw the scene the way the viewer does; collision lives in groups 3 and 4,
    # which are off by default.
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    for group in RENDER_GROUPS:
        option.geomgroup[group] = 1

    tactiles = RobotTactiles(model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = CAM_LOOKAT
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION

    steps_per_frame = max(1, round((1.0 / args.fps) / model.opt.timestep))
    script = grasp.run(model, data)
    done = False

    frames = []
    renderer = mujoco.Renderer(model, args.height, args.width)
    index = 0
    while not done:
        for _ in range(steps_per_frame):
            try:
                next(script)
            except StopIteration:
                done = True
            mujoco.mj_step(model, data)

        tactiles.get_tactile_force(data)

        if index % args.stride == 0:
            renderer.update_scene(data, camera=cam, scene_option=option)
            tactiles.draw_tactile_force(renderer.scene)
            frame = renderer.render()
            panel = _tactile_figure(args.side, tactiles.tactile_maps_kpa,
                                    args.height)
            frames.append(_compose(frame, panel))
            if len(frames) % 20 == 0:
                print(f"frame {len(frames)} (t={data.time:.2f}s)")
        index += 1

    renderer.close()
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"saved {out_path} ({len(frames)} frames, t={data.time:.2f}s)")


if __name__ == "__main__":
    main()
