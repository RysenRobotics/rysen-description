#!/usr/bin/env python3
"""Drive the hand onto a cylinder, close the fingers around it, and lift it.

Usage:
    python3 examples/grasp_cylinder.py                     # right hand
    python3 examples/grasp_cylinder.py --side left
    python3 examples/grasp_cylinder.py --tactiles          # arrows + pressure maps
    python3 examples/grasp_cylinder.py --viz none --tactiles
    python3 examples/grasp_cylinder.py --viz none --tactiles --show-tactile

The hand's root is teleported along the recorded trajectory (see
``lib/grasp_trajectory.py``); the joints are position controlled. The cylinder is
a free body on the floor, so "did it work" is whether it ends up in the air.

With ``--tactiles`` the per-point forces are read after every step. In the GUI
they are shown the same two ways as the MuJoCo demo: red arrows in the viewport,
and a separate window tiling each pad's pressure map in the shape of the hand.
``--show-tactile`` also opens the pressure window when there is no Kit viewport.
The console gets a running summary either way.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))

from isaaclab.app import add_launcher_args  # noqa: E402

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--side", choices=("left", "right"), default="right")
parser.add_argument("--tactiles", action="store_true",
                    help="load the variant carrying the 619 tactile bodies")
parser.add_argument("--show-tactile", action="store_true",
                    help="also draw the pressure maps in a matplotlib window "
                         "(automatic when a Kit viewport is open)")
add_launcher_args(parser)
args_cli = parser.parse_args()

# Fixed pacing: run the whole recorded sequence at the manifest timestep, and
# sample the console every 250 physics steps.
PRINT_EVERY = 250

# The launcher only opens a window when a visualizer is asked for. This sequence
# is worth watching, so Kit is the default; --viz none gives the console-only run.
# Only an omitted --viz becomes Kit: --viz none deliberately passes an empty list
# and must stay empty, or the visualisation code would think a viewport exists.
if args_cli.visualizer is None:
    args_cli.visualizer = "kit"

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg  # noqa: E402
from isaaclab.utils import configclass, instantiate  # noqa: E402

import grasp_trajectory as trajectory  # noqa: E402
import hand as hand_lib  # noqa: E402
from quiet import silence_settings_warnings  # noqa: E402

MANIFEST = hand_lib.manifest(args_cli.side)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground",
                          spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=2500.0))

    # A plain cylinder to grasp: tall enough for all four fingers and the thumb,
    # and light enough that a partial grip can pick it up.
    cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=trajectory.CYLINDER_RADIUS,
            height=trajectory.CYLINDER_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=trajectory.CYLINDER_MASS),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=UsdPhysicsRigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.3),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.65, 0.02, 0.02)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=trajectory.CYLINDER_POS),
    )

    robot = hand_lib.hand_cfg(args_cli.side, args_cli.tactiles,
                              prim_path="{ENV_REGEX_NS}/Robot",
                              pos=trajectory.PARK,
                              rot=trajectory.CONTACT_QUAT)

    contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/tactile/.*", update_period=0.0
    ) if args_cli.tactiles else None


def main():
    silence_settings_warnings()
    dt = float(MANIFEST["timestep"])
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device="cuda:0",
                                      physics=PhysxCfg())
    with launch_simulation(sim_cfg, args_cli):
        from tactiles import RobotTactiles

        sim = sim_utils.SimulationContext(sim_cfg)
        ox, oy = trajectory.SCENE_XY
        sim.set_camera_view(eye=(ox + 0.55, oy - 0.45, 0.35),
                            target=(ox + 0.05, oy, 0.15))
        scene = instantiate(SceneCfg(num_envs=1, env_spacing=1.0))
        sim.reset()
        dt = sim.get_physics_dt()

        robot = scene["robot"]
        cylinder = scene["cylinder"]
        hand_lib.place_hand(robot, trajectory.PARK, trajectory.CONTACT_QUAT,
                            args_cli.side)
        names = robot.joint_names
        limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()

        tactiles = RobotTactiles(MANIFEST, scene["contact"], robot, args_cli.side) \
            if args_cli.tactiles else None
        # The MuJoCo grasp demo shows the array as red arrows plus a pressure
        # window whenever the tactile model is loaded. Do the same in a GUI;
        # headless runs stay console-only unless the window is asked for. The
        # window runs in its own process (see tactile_window) because this
        # virtualenv has no GUI matplotlib backend and Qt crashes Kit.
        panel = None
        arrows = None
        has_viewport = bool(args_cli.visualizer)
        if tactiles is not None:
            from tactile_viewer import TactileArrows
            if has_viewport:
                arrows = TactileArrows()
            if has_viewport or args_cli.show_tactile:
                from tactile_window import TactileWindow
                panel = TactileWindow(args_cli.side)

        print(f"[grasp] {args_cli.side} hand, "
              f"tactile {'on' if args_cli.tactiles else 'off'}")
        total = (trajectory.MOVE_TIME + trajectory.SETTLE_TIME
                 + trajectory.CLOSE_TIME + trajectory.SETTLE_TIME
                 + trajectory.LIFT_TIME + trajectory.HOLD_TIME)
        print(f"[grasp] sequence {total:.1f} s at dt={dt * 1000:.1f} ms")

        grasp = trajectory.Grasp(limits, names, dt)
        start_z = float(cylinder.data.root_pos_w.torch[0, 2])
        thumb_index = robot.body_names.index(f"{args_cli.side}_thumb_link3")
        step = 0
        total_steps = int(round(total / dt))
        while sim.is_running() and step < total_steps:
            hand_lib.drive_step(robot, scene, sim, grasp, args_cli.side,
                                trajectory.CONTACT_QUAT, dt)
            if tactiles is not None:
                tactiles.update(robot)
                if arrows is not None:
                    arrows.update(tactiles)
                if panel is not None:
                    panel.update(tactiles)
            if step % PRINT_EVERY == 0:
                height = float(cylinder.data.root_pos_w.torch[0, 2])
                position = cylinder.data.root_pos_w.torch[0].cpu().numpy()
                message = (f"  t={step * dt:5.2f}s  cylinder"
                           f" ({position[0]:+.3f},{position[1]:+.3f},{height:+.3f})"
                           f" rose {(height - start_z) * 1000:+.0f} mm")
                thumb = robot.data.body_pos_w.torch[0, thumb_index].cpu().numpy()
                message += (f"  thumb({thumb[0]:+.3f},{thumb[1]:+.3f},{thumb[2]:+.3f})"
                            f" r={np.hypot(thumb[0] - ox, thumb[1] - oy) * 1000:.0f}mm")
                if tactiles is not None:
                    message += (f"  tactile: {tactiles.active_points()} pts, "
                                f"peak {tactiles.peak_kpa():6.1f} kPa, "
                                f"sum {tactiles.total_force():6.1f} N, "
                                f"net {tactiles.net_force():5.2f} N")
                print(message)
            step += 1

        palm_index = robot.body_names.index(f"{args_cli.side}_palm_link")
        print(f"[grasp] palm at "
              f"{np.round(robot.data.body_pos_w.torch[0, palm_index].cpu().numpy(), 3)}"
              f" want {np.round(grasp.position, 3)}")
        pose = cylinder.data.root_pose_w.torch[0].cpu().numpy()
        tilt = np.degrees(2.0 * np.arccos(min(abs(float(pose[3])), 1.0)))
        print(f"[grasp] cylinder at {np.round(pose[:3], 3)} "
              f"quat {np.round(pose[3:], 3)} tilt {tilt:.1f} deg")
        height = float(cylinder.data.root_pos_w.torch[0, 2])
        lifted = height - start_z
        print(f"[grasp] finished after {step * dt:.2f} s: cylinder rose "
              f"{lifted * 1000:+.0f} mm "
              f"({'lifted' if lifted > 0.05 else 'NOT lifted'})")
        if tactiles is not None:
            print(f"[grasp] peak pressure {tactiles.peak_kpa():.1f} kPa over "
                  f"{tactiles.active_points()} of "
                  f"{sum(len(i['names']) for i in tactiles.pads.values())} points")
        if arrows is not None:
            arrows.close()
        if panel is not None:
            panel.close()


if __name__ == "__main__":
    main()
