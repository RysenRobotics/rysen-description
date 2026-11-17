#!/usr/bin/env python3
"""Open the hand in the Isaac Sim viewport and drive it with sliders.

Usage:
    python3 examples/view_hand.py                  # right hand, viewport
    python3 examples/view_hand.py --side left
    python3 examples/view_hand.py --side both
    python3 examples/view_hand.py --tactiles
    python3 examples/view_hand.py --headless       # sweep, no window

The palm is held by the six-joint mount and only the finger joints move, the way
the URDF RViz setup shows the model. Every finger joint gets a slider; the
mimicked ones follow their driver and are left out, because commanding them
directly would fight the copy, and the mount joints are left out too, since the
loop re-drives them to a fixed pose every frame.

Headless has no sliders, so it sweeps instead -- still the quickest way to
confirm a freshly generated asset loads.
"""

import argparse
import sys
from pathlib import Path

# The settings plugin re-reads a viewport setting on every frame and logs an
# array-buffer warning each time; quiet.py drops it at the descriptor level,
# because carb's Python logging never sees it.

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))

from isaaclab.app import add_launcher_args  # noqa: E402
from quiet import silence_settings_warnings  # noqa: E402

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--side", choices=("left", "right", "both"), default="right")
parser.add_argument("--tactiles", action="store_true",
                    help="load the variant carrying the 619 tactile bodies")
parser.add_argument("--show-collision", action="store_true",
                    help="draw the collision meshes instead of the visual ones")
parser.add_argument("--headless", action="store_true",
                    help="sweep every joint instead of opening a viewport")
add_launcher_args(parser)
args_cli = parser.parse_args()

# The launcher only opens a window when a visualizer is asked for. This tool is a
# viewer, so Kit is the default and --headless selects the sweep.
if not args_cli.headless and not args_cli.visualizer:
    args_cli.visualizer = "kit"

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass, instantiate  # noqa: E402

import hand as hand_lib  # noqa: E402

SIDES = ("left", "right") if args_cli.side == "both" else (args_cli.side,)
MANIFEST = hand_lib.manifest(SIDES[0])

# Palm height above the origin. With REST_QUAT applied the hand grows upward
# from the wrist, so the palm plate stays just clear of the grid plane.
MOUNT_HEIGHT = 0.10

# The URDF frame has the finger chain running +z out of the wrist -- the palm
# mesh spans z 0..0.133 and index_j0 sits at z +0.1188 -- so the asset is
# authored "fingers up". The body frame PhysX ends up simulating is that frame
# rotated 180 deg about X (its centre of mass resolves 68 mm *below* the link
# origin, and every finger joint resolves to -z), so an identity rotation hangs
# the fingers straight down. This undoes the flip. Rotating about X leaves the
# palm normal (+-x) alone, so the hand still faces the same way.
REST_QUAT = (0.0, 1.0, 0.0, 0.0)

# Half the gap between the two palms when both hands are shown.
BOTH_SPREAD = 0.13

# Headless sweep: one full oscillation, sampled every 60 steps.
SWEEP_SECONDS = 6.0
SWEEP_PRINT_EVERY = 60


def mount_pos(side):
    """Where the palm is pinned. Sides are spaced along y, across the view."""
    if len(SIDES) == 1:
        return (0.0, 0.0, MOUNT_HEIGHT)
    return (0.0, BOTH_SPREAD if side == "right" else -BOTH_SPREAD, MOUNT_HEIGHT)


@configclass
class SceneCfg(InteractiveSceneCfg):
    """One hand, or two side by side -- the fields are fixed at class creation."""

    # Without a light the hand renders as a black silhouette: the asset carries
    # physics materials only, so everything an object's shading depends on comes
    # from the scene.
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=2500.0))

    robot = hand_lib.hand_cfg(SIDES[0], args_cli.tactiles,
                              prim_path="{ENV_REGEX_NS}/Robot",
                              pos=mount_pos(SIDES[0]))
    contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/tactile/.*", update_period=0.0
    ) if args_cli.tactiles else None
    second = hand_lib.hand_cfg(SIDES[1], args_cli.tactiles,
                               prim_path="{ENV_REGEX_NS}/Second",
                               pos=mount_pos(SIDES[1])) if len(SIDES) > 1 else None
    contact_second = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Second/.*/tactile/.*", update_period=0.0
    ) if args_cli.tactiles and len(SIDES) > 1 else None


def sweep_targets(step, dt, limits, device):
    """Every joint oscillates over its own range, offset so they do not align."""
    import torch

    time = step * dt
    centres = (limits[:, 0] + limits[:, 1]) / 2.0
    spans = (limits[:, 1] - limits[:, 0]) / 2.0 * 0.85
    phase = np.linspace(0.0, 2.0 * np.pi, len(centres), endpoint=False)
    values = centres + spans * np.sin(2.0 * np.pi * time / SWEEP_SECONDS + phase)
    return torch.tensor(values[None, :], dtype=torch.float32, device=device)


def _slider_setter(targets, index):
    def _on_change(model):
        targets[index] = model.get_value_as_float()
    return _on_change


class JointPanel:
    """A slider per finger joint, one block per hand.

    ``targets`` is what the main loop feeds to the actuators, so a drag takes
    effect on the next physics step without any extra plumbing.

    The six mount joints are left out. They are re-driven to ``mount_pos`` every
    frame to hold the hand in place, so a slider there would be overwritten
    before it could do anything; this viewer is for looking at joint ranges, not
    for moving the hand around.
    """

    def __init__(self, robots, sides):
        import omni.ui as ui

        self.targets = [np.zeros(len(robot.joint_names), dtype=np.float32)
                        for robot in robots]
        self.window = ui.Window("Joints", width=470, height=760)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=4):
                    for robot, side, targets in zip(robots, sides, self.targets):
                        self._section(ui, robot, side, targets)

    def _section(self, ui, robot, side, targets):
        mimicked = {name for name, _driver, _m, _o in hand_lib.mimic_pairs(side)}
        mount = set(hand_lib.manifest(side)["mount_joints"])
        limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
        ui.Label(side, height=26, style={"font_size": 16})
        for index, name in enumerate(robot.joint_names):
            if name in mimicked or name in mount:
                continue
            with ui.HStack(height=20):
                ui.Label(name.replace(side + "_", ""), width=150)
                model = ui.SimpleFloatModel(float(targets[index]))
                model.add_value_changed_fn(_slider_setter(targets, index))
                ui.FloatSlider(model=model, min=float(limits[index][0]),
                               max=float(limits[index][1]))


def show_collision_geometry():
    """Draw the collision meshes instead of the visual ones, tinted green.

    Every collision placement is authored with purpose=guide so it stays out of
    the render, and the visual ones with default. Swapping the two shows what
    PhysX actually collides with: hulls for the pads, SDF meshes for the rest,
    and the tactile spheres as authored.

    The tint matters: the SDF shapes are cooked from the source meshes, so
    without it the swap is invisible and only the pad hulls look different.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    tint = Gf.Vec3f(0.15, 0.8, 0.3)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Imageable):
            continue
        path = str(prim.GetPath())
        imageable = UsdGeom.Imageable(prim)
        if "/collision/" in path:
            imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
            UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([tint])
        elif "/visual/" in path:
            imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)


def main():
    silence_settings_warnings()
    dt = float(MANIFEST["timestep"])
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device="cuda:0",
                                      physics=PhysxCfg())
    with launch_simulation(sim_cfg, args_cli):
        from tactiles import RobotTactiles

        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(0.55, -0.35, 0.35), target=(0.0, 0.0, 0.20))
        scene = instantiate(SceneCfg(num_envs=1, env_spacing=1.0))
        if args_cli.show_collision:
            show_collision_geometry()
        sim.reset()
        dt = sim.get_physics_dt()

        robots = [scene["robot"]]
        contacts = [scene["contact"] if args_cli.tactiles else None]
        if len(SIDES) > 1:
            robots.append(scene["second"])
            contacts.append(scene["contact_second"] if args_cli.tactiles else None)

        for robot, side in zip(robots, SIDES):
            print(f"[view] {side} hand: {robot.num_joints} joints, "
                  f"{robot.num_bodies} bodies"
                  f"{', tactile' if args_cli.tactiles else ''}")

        tactiles = [RobotTactiles(hand_lib.manifest(side), contact, robot, side)
                    if args_cli.tactiles else None
                    for robot, side, contact in zip(robots, SIDES, contacts)]

        quat = REST_QUAT

        # The asset's joints start at zero, which leaves the palm on the world
        # origin -- with two hands that means they sit inside each other. The
        # `init_state` in the config only fills the default buffers; this Isaac
        # Lab build does not write them into the solver, so put the hand in
        # place explicitly, the same way grasp_cylinder does.
        for robot, side in zip(robots, SIDES):
            hand_lib.place_hand(robot, mount_pos(side), quat, side)

        if args_cli.headless:
            run_sweep(robots, tactiles, quat, dt, sim, scene)
        else:
            run_viewport(robots, tactiles, quat, dt, sim, scene)


def run_viewport(robots, tactiles, quat, dt, sim, scene):
    """Hold each palm still and follow the sliders until the window closes."""
    import torch

    panel = JointPanel(robots, SIDES)
    while sim.is_running():
        for robot, side, targets in zip(robots, SIDES, panel.targets):
            commanded = torch.tensor(targets[None, :], dtype=torch.float32,
                                     device=robot.device)
            hand_lib.mirror_mimic_targets(side, commanded, robot.joint_names)
            hand_lib.mount_targets(robot, mount_pos(side), quat, side, commanded)
            robot.actuators.target_command.set_position_index(value=commanded, full_data=True)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        for robot, tactile in zip(robots, tactiles):
            if tactile is not None:
                tactile.update(robot)


def run_sweep(robots, tactiles, quat, dt, sim, scene):
    """Drive every joint through its range and report where it ended up."""
    total = int(round(SWEEP_SECONDS / dt))
    for step in range(total + 1):
        if not sim.is_running():
            break
        for robot, side in zip(robots, SIDES):
            limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
            targets = sweep_targets(step, dt, limits, robot.device)
            hand_lib.mirror_mimic_targets(side, targets, robot.joint_names)
            hand_lib.mount_targets(robot, mount_pos(side), quat, side, targets)
            robot.actuators.target_command.set_position_index(value=targets, full_data=True)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        if step % SWEEP_PRINT_EVERY == 0:
            for robot, side, tactile in zip(robots, SIDES, tactiles):
                state = robot.data.joint_pos.torch[0].cpu().numpy()
                line = "  ".join(f"{name.replace(side + '_', '')}={value:+.2f}"
                                 for name, value in zip(robot.joint_names, state))
                print(f"  t={step * dt:5.2f}s  {side}: {line}")
                if tactile is not None:
                    print(f"         tactile: {tactile.active_points()} points, "
                          f"peak {tactile.peak_kpa():.1f} kPa")

    for robot, side in zip(robots, SIDES):
        limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
        state = robot.data.joint_pos.torch[0].cpu().numpy()
        print(f"[view] {side} final joint positions (rad) and ranges:")
        for name, value, limit in zip(robot.joint_names, state, limits):
            flag = "" if limit[0] - 1e-4 <= value <= limit[1] + 1e-4 else "  OUT OF RANGE"
            print(f"  {name:20s} {value:+.3f}  [{limit[0]:+.3f}, {limit[1]:+.3f}]{flag}")


if __name__ == "__main__":
    main()
