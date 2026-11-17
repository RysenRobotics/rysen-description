"""Asset configuration and joint helpers shared by the examples.

The generated USD carries the geometry, collision and drives; the
``manifest.json`` next to it carries what the examples need at runtime and would
be awkward to dig out of USD: joint limits, the mimic relations, and the tactile
point layout.
"""

import json
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
USD_DIR = EXAMPLES_DIR.parent

def asset_dir(side):
    return USD_DIR / f"apex_hand_{side}"


def manifest(side):
    path = asset_dir(side) / "manifest.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nrun: python3 tools/urdf_to_usd.py --side {side}")
    return json.loads(path.read_text(encoding="utf-8"))


def usd_path(side, tactiles=False, mount=True):
    """Path of one generated variant.

    ``mount=True`` selects the ``_mount`` variant, which carries a six-joint
    root chain so a standalone scene can hold and move the hand. ``mount=False``
    is the bare hand for scenes that attach it to something of their own.
    """
    m = manifest(side)
    key = "mesh" + ("_mount" if mount else "") + ("_tactile" if tactiles else "")
    name = m["variants"].get(key)
    if name is None:
        raise SystemExit(
            f"{side}: variant '{key}' was not generated "
            f"(run: python3 tools/urdf_to_usd.py --side {side})")
    return asset_dir(side) / name


def mount_values(pos, quat, names):
    """Six mount joint values that put the root body at ``pos`` / ``quat``.

    Joint order is tx, ty, tz, rx, ry, rz. Translation comes first, so the
    rotation accumulates as R = Rx * Ry * Rz -- intrinsic xyz.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    rpy = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")
    raw = np.array([*pos, *rpy], dtype=float)
    # Measured on this chain: the Y and Z prismatic axes run opposite to the
    # world axes, while X matches. Feeding the values through unchanged puts the
    # palm at (x, -y, -z).
    raw[1] = -raw[1]
    raw[2] = -raw[2]
    return {name: float(value) for name, value in zip(names, raw)}


def hand_cfg(side, tactiles=False, prim_path="/World/Robot",
             pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), mount=True):
    """Articulation config for one hand.

    Actuator gains and effort limits come from the manifest, i.e. from the
    generator config, so the examples and the USD never disagree -- the same
    numbers as the MJCF model (kp 20, kv 0.17, URDF effort per joint).

    ``mount=False`` loads the variant without the six-joint root chain; the hand
    is then a floating articulation that the caller has to hold itself.
    """
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    import isaaclab.sim as sim_utils

    m = manifest(side)
    gains = m["actuators"]
    # Every joint gets a drive, mimic included. The MJCF model leaves the
    # mimicked joint unactuated because a second position servo would fight the
    # equality constraint; PhysX has no such constraint here, so the coupling is
    # a target copy (``mirror_mimic_targets``) and both ends need a motor.
    driven = list(m["joints"])
    mount_joints = list(m.get("mount_joints", [])) if mount else []

    # One actuator group covering every joint of the articulation. Splitting the
    # mount into its own group meant two target buffers, and the mount half of
    # the command never reached the solver.
    names = [joint["name"] for joint in driven] + mount_joints
    stiffness = {joint["name"]: float(gains["kp"]) for joint in driven}
    damping = {joint["name"]: float(gains["kv"]) for joint in driven}
    effort = {joint["name"]: float(joint["effort"]) for joint in driven}
    # The mount carries the whole hand, so it needs gains three orders of
    # magnitude above a finger's -- same control law, different numbers.
    # Same numbers the USD drive carries. A stiffer mount is not "more
    # accurate": at 1e6 N/m a millimetre of displacement is a kilonewton, which
    # turns the hand into an immovable wall and flicks a 0.3 kg cylinder away on
    # contact. MuJoCo's weld works out to roughly 7.5e4 N/m.
    for name in mount_joints:
        stiffness[name] = 1.0e4
        damping[name] = 1.0e2
        effort[name] = 1.0e4
    actuators = {
        "all": ImplicitActuatorCfg(
            joint_names_expr=names,
            stiffness=stiffness,
            damping=damping,
            effort_limit_sim=effort,
        )
    }
    return ArticulationCfg(
        prim_path=prim_path,
        # The mounted variants anchor the base to the world with a fixed joint,
        # so the articulation is grounded and the six joints carry the hand.
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path(side, tactiles, mount))),
        # With a mount, the pose comes from its joints. These values fill the
        # default buffers; this Isaac Lab build does not write them into the
        # solver on reset, so a scene still has to call `place_hand` once to
        # actually move the palm off the origin. Without a mount the root is
        # free and the spawn transform is all that positions it.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0) if mount else tuple(pos),
            rot=(1.0, 0.0, 0.0, 0.0) if mount else tuple(rot),
            joint_pos=mount_values(pos, rot, mount_joints) if mount_joints else {}),
        actuators=actuators,
    )


def mimic_pairs(side):
    """``[(mimicked joint, driver joint, multiplier, offset), ...]``."""
    pairs = []
    for joint in manifest(side)["joints"]:
        mimic = joint["mimic"]
        if mimic:
            pairs.append((joint["name"], mimic["joint"],
                          float(mimic.get("multiplier", "1.0")),
                          float(mimic.get("offset", "0.0"))))
    return pairs


def mirror_mimic_targets(side, targets, joint_names):
    """Copy each driver's target onto its mimicked partner, in place.

    ``targets`` is a ``(num_envs, num_joints)`` tensor indexed like
    ``joint_names``. PhysX has no mimic constraint (the URDF importer drops the
    relation and MuJoCo needed an equality for it), so the coupling lives here:
    the mimicked joint is commanded to the same angle as its driver, which for
    this hand is always a 1:1 relation over an identical range.
    """
    index = {name: i for i, name in enumerate(joint_names)}
    for mimicked, driver, multiplier, offset in mimic_pairs(side):
        if mimicked in index and driver in index:
            targets[:, index[mimicked]] = (
                multiplier * targets[:, index[driver]] + offset)
    return targets


def _mount_ids(robot, side):
    """Joint indices of the six mount joints, cached on the robot."""
    ids = getattr(robot, "_mount_joint_ids", None)
    if ids is None:
        index = {name: position for position, name in enumerate(robot.joint_names)}
        ids = [index[name] for name in manifest(side)["mount_joints"]
               if name in index]
        robot._mount_joint_ids = ids
    return ids


def place_hand(robot, pos, quat, side):
    """Put the hand at a pose by writing the mount joint state.

    Used once after reset: the mount joints start at zero, which puts the palm
    on the world origin -- inside whatever the example is about to grasp.
    """
    import torch

    order = manifest(side)["mount_joints"]
    ids = _mount_ids(robot, side)
    values = mount_values(pos, quat, order)
    state = torch.tensor([[values[name] for name in order]],
                         dtype=torch.float32, device=robot.device)
    robot.write_joint_state_to_sim(position=state,
                                   velocity=torch.zeros_like(state),
                                   joint_ids=ids)


def drive_step(robot, scene, sim, grasp, side, quat, dt):
    """Advance the grasp by one physics step and sync the scene.

    The order matters: every joint target is written in one call before
    ``write_data_to_sim``, because driving the mount through its own
    ``joint_ids`` call left the hand falling under gravity.
    """
    import numpy as np
    import torch

    grasp.step()
    targets = mirror_mimic_targets(
        side,
        torch.tensor(np.asarray(grasp.targets, dtype=np.float32)[None, :],
                     device=robot.device),
        robot.joint_names)
    mount_targets(robot, grasp.position, quat, side, targets)
    robot.actuators.target_command.set_position_index(value=targets,
                                                      full_data=True)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt)


def mount_targets(robot, pos, quat, side, targets):
    """Write the six mount joint targets into ``targets``, in place.

    The asset carries a six-joint mount (three prismatic, three revolute) instead
    of a floating base that has to be teleported. Writing the root pose every
    step discards velocity and drags the colliders through whatever they touch;
    driving joints keeps the hand inside the solver, so a contact pushes back.

    The caller sets every joint target in one call afterwards: routing the mount
    through ``joint_ids`` on its own left the hand falling under gravity.
    """
    import numpy as np
    import torch

    # Joint order is tx, ty, tz, rx, ry, rz. Translation first, so the rotation
    # accumulates as R = Rx * Ry * Rz -- intrinsic xyz, the lowercase sequence.
    order = manifest(side)["mount_joints"]
    table = mount_values(pos, quat, order)
    values = np.array([table[name] for name in order], dtype=np.float32)

    targets[:, _mount_ids(robot, side)] = torch.tensor(values, dtype=targets.dtype,
                                   device=targets.device)
    return targets
