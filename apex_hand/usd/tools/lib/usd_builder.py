"""Assemble one USD variant of the hand from a parsed URDF.

The layout is one root Xform carrying the articulation root, one Xform per
rigid body, and a ``Physics`` scope holding every joint:

    /apex_hand_right                       Xform + ArticulationRootAPI
      Physics/                             Scope
        right_index_j0                    PhysicsRevoluteJoint
        right_index_tactile0_point_000_fix PhysicsFixedJoint   (tactile only)
      Materials/                           hard_link, soft_tactile
      meshes/                              38 Mesh prims, reference targets
      right_palm_link                      Xform + RigidBodyAPI + MassAPI
        visual/<mesh>                      Mesh (internal reference)
        collision/<mesh>                   Mesh (same reference, CollisionAPI)
        frames/right_index_tip             Xform marker, purpose=guide
      right_index_link1
        tactile/right_index_tactile0_point_000
                                           Xform + RigidBodyAPI (tactile only)

Bodies are flat under the root; only the tactile spheres are nested, under the
pad they sit in, because their placement is then written in the pad's frame.
"""

import json
from pathlib import Path

import numpy as np

import usd_config as cfg
from mesh_io import load_mesh, matrix_to_gf, reference_mesh, write_mesh_prim
from transforms import (axis_alignment_quat, diagonalize_inertia, matrix_to_quat,
                        quat_conjugate, quat_multiply, rpy_to_matrix, transform_matrix)


# --------------------------------------------------------------------------- #
# Mesh library
# --------------------------------------------------------------------------- #
def build_mesh_library(stage, urdf_path, bodies, scope_path):
    """Write every mesh a body uses into ``stage`` as ``<scope>/<stem>``.

    Returns ``{mesh filename: prim path}``. Meshes shared by several bodies are
    written once; the visual and collision placements both reference the entry.

    The collision approximation is authored here as well as on the collision
    placement. PhysX resolves a referenced mesh back to this prim and reads the
    token from it; a token that only exists on the referencing prim is ignored,
    and every shape silently falls back to a convex hull.
    """
    from pxr import PhysxSchema, UsdPhysics

    from urdf_loader import mesh_path, mesh_prim_name

    written = {}
    for body in bodies:
        for _kind, filename, _transform in body["meshes"]:
            if filename in written:
                continue
            prim_path = f"{scope_path}/{mesh_prim_name(filename)}"
            prim = write_mesh_prim(stage, prim_path, load_mesh(mesh_path(urdf_path, filename)))
            approximation = cfg.approximation_for(Path(filename).stem)
            if approximation == "sdf":
                sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                sdf.CreateSdfResolutionAttr().Set(int(cfg.SDF_RESOLUTION))
                sdf.CreateSdfMarginAttr().Set(float(cfg.SDF_MARGIN))
                sdf.CreateSdfNarrowBandThicknessAttr().Set(float(cfg.SDF_NARROW_BAND))
            elif approximation == "convexDecomposition":
                decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                decomp.CreateMaxConvexHullsAttr().Set(
                    int(cfg.max_convex_hulls_for(Path(filename).stem)))
                decomp.CreateErrorPercentageAttr().Set(
                    float(cfg.CONVEX_ERROR_PERCENTAGE))
                decomp.CreateShrinkWrapAttr().Set(bool(cfg.CONVEX_SHRINK_WRAP))
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
                approximation)
            written[filename] = prim_path
    return written


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #
def _define_material(stage, path, visual_rgb, physics=None, opacity=1.0):
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    if physics is not None:
        # Physics side: PhysX reads these from UsdPhysics.MaterialAPI.
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr().Set(float(physics["static_friction"]))
        api.CreateDynamicFrictionAttr().Set(float(physics["dynamic_friction"]))
        api.CreateRestitutionAttr().Set(float(physics["restitution"]))

    stiffness = (physics or {}).get("compliant_contact_stiffness", 0.0)
    if stiffness and stiffness > 0.0:
        # Turns the contact into an implicit spring. Both materials are
        # compliant: hard plastic for the links, soft silicone for the pads.
        # PhysX takes the compliant side's parameters when the other side is a
        # plain rigid material, so this covers contacts with the example's
        # cylinder and ground too.
        compliant = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        compliant.CreateCompliantContactStiffnessAttr().Set(float(stiffness))
        compliant.CreateCompliantContactDampingAttr().Set(
            float(physics.get("compliant_contact_damping", 0.0)))

    # Render side: a constant colour, enough to tell the shells from the pads.
    shader = UsdShade.Shader.Define(stage, f"{path}/preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*visual_rgb))
    # Below 1 the renderer treats the surface as transparent; UsdPreviewSurface
    # has no other switch.
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def build_materials(stage, root_path, translucent_pads=False):
    """Create the three contact materials and return ``{name: UsdShade.Material}``.

    The soft material is also what the pad shells render with, so in the tactile
    variant it goes translucent and the buried array shows through. The point
    material carries the physics of the spheres as well as their red render
    colour: binding it only for display left the array using the shell's
    contact parameters.
    """
    from pxr import UsdGeom

    scope = UsdGeom.Scope.Define(stage, f"{root_path}/{cfg.MATERIALS_SCOPE}")
    materials = {}
    for name, values in ((cfg.MATERIAL_HARD_LINK, cfg.HARD_LINK_MATERIAL),
                         (cfg.MATERIAL_SOFT_TACTILE, cfg.SOFT_TACTILE_MATERIAL)):
        rgb, opacity = cfg.SHELL_RGB, 1.0
        if translucent_pads and name == cfg.MATERIAL_SOFT_TACTILE:
            rgb, opacity = cfg.TACTILE_SHELL_RGBA[:3], cfg.TACTILE_SHELL_RGBA[3]
        materials[name] = _define_material(
            stage, f"{scope.GetPath()}/{name}", rgb, values, opacity)
    materials[cfg.MATERIAL_TACTILE_POINT] = _define_material(
        stage, f"{scope.GetPath()}/{cfg.MATERIAL_TACTILE_POINT}",
        cfg.TACTILE_POINT_RGB, cfg.TACTILE_POINT_MATERIAL)
    return materials


def _bind(material, prim, purpose):
    from pxr import UsdShade

    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if purpose:
        api.Bind(material, UsdShade.Tokens.weakerThanDescendants, purpose)
    else:
        api.Bind(material)


# --------------------------------------------------------------------------- #
# Bodies
# --------------------------------------------------------------------------- #
def _rest_transforms(root_body, movable):
    """Root-frame pose of every body with all joints at zero.

    Bodies are authored flat under the root and their placement lives on
    ``Physics/<joint>``, so without a rest pose the file only reads correctly
    after PhysX has forward-kinematicked the articulation -- in a viewport every
    part would pile up at the origin until the simulation is played.
    """
    rest = {root_body: np.eye(4)}
    pending = list(movable)
    while pending:
        progressed = False
        for joint in list(pending):
            if joint["parent"] not in rest:
                continue
            rest[joint["child"]] = rest[joint["parent"]] @ transform_matrix(
                joint["xyz"], joint["rpy"])
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError(
                f"joints out of tree: {[joint['name'] for joint in pending]}")
    return rest


def _add_body(stage, root_path, body, rest):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    prim = UsdGeom.Xform.Define(stage, f"{root_path}/{body['name']}").GetPrim()
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix_to_gf(rest))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)

    mass = body["mass"] if body["mass"] > 0.0 else 1e-3
    mass_api.CreateMassAttr().Set(float(mass))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*body["com"]))
    if body["inertia"] is not None:
        values, quat = diagonalize_inertia(body["inertia"])
        # A degenerate (flat, or single-point) inertia is legal input and illegal
        # USD: clamp the eigenvalues to something the solver can invert.
        values = np.maximum(values, mass * 1e-8)
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, values)))
        mass_api.CreatePrincipalAxesAttr().Set(
            Gf.Quatf(float(quat[0]), Gf.Vec3f(*map(float, quat[1:]))))

    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx.CreateLinearDampingAttr().Set(0.0)
    physx.CreateAngularDampingAttr().Set(0.0)
    # The fingers are hit at a few m/s at most; clamping depenetration keeps a
    # bad first frame from launching the hand.
    physx.CreateMaxDepenetrationVelocityAttr().Set(5.0)
    return prim


def _add_meshes(stage, body_prim, body, mesh_library, materials):
    """Author the visual and collision placements of one body's meshes.

    The ``*_tactile*`` shells are the contact surface at the pads and cook as
    convex hulls; the ``*_pad`` links are coordinate frames with no mesh. The
    shells are drawn on top of the phalanx, so leaving them out makes the hand
    thinner than it looks and the visible pad sinks into whatever it touches.
    """
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    visual_scope = UsdGeom.Scope.Define(stage, f"{body_prim.GetPath()}/{cfg.VISUAL_SCOPE}")
    collision_scope = UsdGeom.Scope.Define(
        stage, f"{body_prim.GetPath()}/{cfg.COLLISION_SCOPE}")

    for kind, filename, transform in body["meshes"]:
        stem = Path(filename).stem
        mesh_prim = mesh_library[filename]
        material = materials[cfg.material_name(stem)]
        if kind == "visual":
            prim = reference_mesh(stage, f"{visual_scope.GetPath()}/{stem}",
                                  mesh_prim, transform, purpose="default")
            _bind(material, prim, "")
            continue
        prim = reference_mesh(stage, f"{collision_scope.GetPath()}/{stem}",
                              mesh_prim, transform, purpose="guide")
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            cfg.approximation_for(stem))
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr().Set(float(cfg.CONTACT_OFFSET))
        collision.CreateRestOffsetAttr().Set(float(cfg.REST_OFFSET))
        _bind(material, prim, "physics")


def _add_frames(stage, body_prim, body_name, frames):
    """Author ``*_tip`` / ``*_pad`` reference links as plain markers."""
    from pxr import UsdGeom

    selected = [frame for frame in frames if frame["body"] == body_name]
    if not selected:
        return
    scope = UsdGeom.Scope.Define(stage, f"{body_prim.GetPath()}/{cfg.FRAMES_SCOPE}")
    for frame in selected:
        prim = UsdGeom.Xform.Define(stage, f"{scope.GetPath()}/{frame['name']}").GetPrim()
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix_to_gf(frame["transform"]))
        UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)


# --------------------------------------------------------------------------- #
# Tactile
# --------------------------------------------------------------------------- #
def _add_tactile(stage, root_path, body_prims, pads, placements, materials):
    """Author one small rigid body per tactile point, fixed to its pad body.

    MuJoCo can read a contact force off a geom, so the MJCF model keeps the
    spheres as collision geometry of the pad body. PhysX only reports a force per
    rigid body, so each sphere gets a body of its own. They sit inside the pad
    (see ``config.yaml``), which is what an object sinks through to reach them.
    Their contact spring comes from ``tactile_point``, not from the shell.
    """
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    joints_scope = f"{root_path}/{cfg.JOINTS_SCOPE}"
    point_material = materials[cfg.MATERIAL_TACTILE_POINT]
    described = []
    for pad, points in pads.items():
        if pad not in placements:
            raise SystemExit(f"tactile pad {pad} is not a link of the base URDF")
        body_name, pad_transform = placements[pad]
        body_prim = body_prims[body_name]
        rotation = pad_transform[:3, :3]
        scope = UsdGeom.Scope.Define(
            stage, f"{body_prim.GetPath()}/{cfg.TACTILE_SCOPE}").GetPrim()

        for name, position, rpy in points:
            local = (pad_transform @ np.array([*position, 1.0]))[:3]
            frame = rotation @ rpy_to_matrix(rpy)
            quat = matrix_to_quat(frame)

            path = f"{scope.GetPath()}/{name}"
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            xform = UsdGeom.Xformable(prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, local)))
            xform.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(*map(float, quat[1:]))))
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass = float(cfg.TACTILE_MASS)
            mass_api.CreateMassAttr().Set(mass)
            # Solid sphere: I = 2/5 m r^2, the same for all three axes.
            inertia = 0.4 * mass * cfg.TACTILE_RADIUS ** 2
            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(inertia, inertia, inertia))
            # Isaac Lab's ContactSensor refuses to build without this API.
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)

            sphere = UsdGeom.Sphere.Define(stage, f"{path}/{cfg.SHAPE_PRIM}")
            sphere.CreateRadiusAttr().Set(float(cfg.TACTILE_RADIUS))
            sphere_prim = sphere.GetPrim()
            UsdPhysics.CollisionAPI.Apply(sphere_prim)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(sphere_prim)
            # A small fraction of the radius: a generous contact offset on a 1 mm
            # sphere stretches it into a 1.5 mm one and the array stops matching
            # the pad.
            collision.CreateContactOffsetAttr().Set(float(cfg.TACTILE_RADIUS) * 0.1)
            collision.CreateRestOffsetAttr().Set(0.0)
            # Render and physics both use the point material: the array is
            # stiffer than the silicone shell it sits in, so it must not inherit
            # the shell's contact spring.
            _bind(point_material, sphere_prim, "")
            _bind(point_material, sphere_prim, "physics")

            joint = UsdPhysics.FixedJoint.Define(stage, f"{joints_scope}/{name}_fix")
            joint.CreateBody0Rel().SetTargets([body_prim.GetPath()])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, local)))
            joint.CreateLocalRot0Attr().Set(
                Gf.Quatf(float(quat[0]), Gf.Vec3f(*map(float, quat[1:]))))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            joint.CreateCollisionEnabledAttr().Set(False)

            described.append({
                "name": name,
                "pad": pad,
                "body": body_name,
                "pos": [float(v) for v in local],
                # Contact normal in the pad body frame: the point's +z.
                "normal": [float(v) for v in frame[:, 2]],
            })
    return described


# --------------------------------------------------------------------------- #
# Joints
# --------------------------------------------------------------------------- #
def _add_joint(stage, root_path, joint, body_prims):
    """Author one revolute joint, with the axis baked into the joint frame.

    URDF gives the child frame in the parent frame and the axis in the child
    frame. USD names the axis (X, Y or Z) and puts the joint frame where the
    caller says, so the frame is rotated to make the axis land on Z and the
    rotation is undone on the child side -- localRot0 * localRot1^-1 must still
    equal the URDF placement.
    """
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    alignment = axis_alignment_quat(joint["axis"])
    placement = rpy_to_matrix(joint["rpy"])
    rot0 = quat_multiply(matrix_to_quat(placement), alignment)
    rot1 = quat_conjugate(alignment)

    path = f"{root_path}/{cfg.JOINTS_SCOPE}/{joint['name']}"
    prim = UsdPhysics.RevoluteJoint.Define(stage, path)
    prim.CreateBody0Rel().SetTargets([body_prims[joint["parent"]].GetPath()])
    prim.CreateBody1Rel().SetTargets([body_prims[joint["child"]].GetPath()])
    prim.CreateAxisAttr().Set("Z")
    prim.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in joint["xyz"]]))
    prim.CreateLocalRot0Attr().Set(Gf.Quatf(float(rot0[0]), Gf.Vec3f(*map(float, rot0[1:]))))
    prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    prim.CreateLocalRot1Attr().Set(Gf.Quatf(float(rot1[0]), Gf.Vec3f(*map(float, rot1[1:]))))
    # Adjacent links overlap slightly in this CAD model; leaving jointed pairs
    # out of collision keeps that from pushing the hand apart.
    prim.CreateCollisionEnabledAttr().Set(False)

    limit = joint["limit"]
    lower = float(limit.get("lower", 0.0))
    upper = float(limit.get("upper", 0.0))
    effort = float(limit.get("effort", 1.0))
    velocity = float(limit.get("velocity", 0.0))
    prim_prim = prim.GetPrim()
    # USD stores angular limits in degrees; PhysX reads these two attributes.
    prim_prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(
        float(np.degrees(lower)))
    prim_prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(
        float(np.degrees(upper)))

    drive = UsdPhysics.DriveAPI.Apply(prim_prim, "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(cfg.DEFAULT_KP))
    drive.CreateDampingAttr().Set(float(cfg.DEFAULT_KV))
    drive.CreateMaxForceAttr().Set(effort)
    drive.CreateTargetPositionAttr().Set(0.0)

    physx = PhysxSchema.PhysxJointAPI.Apply(prim_prim)
    physx.CreateArmatureAttr().Set(float(cfg.JOINT_ARMATURE))
    physx.CreateJointFrictionAttr().Set(float(cfg.JOINT_FRICTION))
    if velocity > 0.0:
        # USD stores angular velocities in degrees per second, like the limits
        # above; PhysX reads this one as-is. Writing rad/s here would clamp the
        # joint at 1/57th of the URDF's 400 deg/s.
        physx.CreateMaxJointVelocityAttr().Set(float(np.degrees(velocity)))

    # PhysX has no mimic constraint; the relation is carried as metadata so the
    # control layer can mirror the driver joint (see examples/lib/hand.py).
    mimic = joint["mimic"]
    if mimic is not None:
        prim_prim.SetCustomDataByKey("urdf:mimicJoint", str(mimic["joint"]))
        prim_prim.SetCustomDataByKey(
            "urdf:mimicMultiplier", float(mimic.get("multiplier", 1.0)))
        prim_prim.SetCustomDataByKey(
            "urdf:mimicOffset", float(mimic.get("offset", 0.0)))
    return prim


# --------------------------------------------------------------------------- #
# Mount
# --------------------------------------------------------------------------- #
def _add_mount(stage, root_path, root_body, body_prims, rest):
    """Author the six-joint mount between the world and the root body.

    Three prismatic joints carry translation and three revolute joints rotation,
    chained through massless links, so the root body has six actuated degrees of
    freedom instead of being pinned by writing its pose. That distinction is what
    keeps the hand physical: a teleported root discards velocity and drags its
    colliders through whatever they were touching, where an actuated one is
    pushed back by the contact it just made.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    scope = UsdGeom.Scope.Define(stage, f"{root_path}/{cfg.MOUNT_SCOPE}")
    origin = rest[root_body]

    def spawn(name, mass):
        prim = UsdGeom.Xform.Define(
            stage, f"{scope.GetPath()}/{name}").GetPrim()
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix_to_gf(origin))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(mass))
        return prim

    previous = spawn(f"{root_body}_mount_base", cfg.MOUNT_BASE_MASS)
    # Pin the base to the world. An unanchored articulation free-falls, and the
    # mount joints would then only move the hand relative to a base that is
    # itself dropping. An empty body0 is how UsdPhysics says "the world".
    anchor = UsdPhysics.FixedJoint.Define(
        stage, f"{root_path}/{cfg.JOINTS_SCOPE}/{root_body}_mount_anchor")
    anchor.CreateBody1Rel().SetTargets([previous.GetPath()])
    anchor.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in origin[:3, 3]]))
    anchor.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))

    names = []
    for index, (name, kind, axis) in enumerate(cfg.MOUNT_CHAIN):
        last = index == len(cfg.MOUNT_CHAIN) - 1
        child = body_prims[root_body] if last else spawn(
            f"{root_body}_mount_link{index}", cfg.MOUNT_LINK_MASS)

        joint_name = f"{root_body}_mount_{name}"
        path = f"{root_path}/{cfg.JOINTS_SCOPE}/{joint_name}"
        if kind == "prismatic":
            joint = UsdPhysics.PrismaticJoint.Define(stage, path)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            kp, kv = cfg.MOUNT_LINEAR_KP, cfg.MOUNT_LINEAR_KV
            limit = cfg.MOUNT_TRAVEL
        else:
            joint = UsdPhysics.RevoluteJoint.Define(stage, path)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            kp, kv = cfg.MOUNT_ANGULAR_KP, cfg.MOUNT_ANGULAR_KV
            limit = cfg.MOUNT_SWING_DEG

        joint.CreateBody0Rel().SetTargets([previous.GetPath()])
        joint.CreateBody1Rel().SetTargets([child.GetPath()])
        joint.CreateAxisAttr().Set(axis)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateLowerLimitAttr().Set(float(-limit))
        joint.CreateUpperLimitAttr().Set(float(limit))
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(kp))
        drive.CreateDampingAttr().Set(float(kv))
        drive.CreateMaxForceAttr().Set(1e4)
        drive.CreateTargetPositionAttr().Set(0.0)

        names.append(joint_name)
        previous = child
    return names


# --------------------------------------------------------------------------- #
# Whole file
# --------------------------------------------------------------------------- #
def build_variant(path, urdf_path, tactile_urdf_path, variant):
    """Write one variant of a hand. Returns the manifest dict."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    from tactile_points import load_points
    from urdf_loader import build_bodies, load_urdf

    _suffix, wants_tactile, wants_mount = cfg.VARIANTS[variant]
    robot, links, joints = load_urdf(urdf_path)
    root_body, bodies, movable, frames, placements = build_bodies(
        links, joints, cfg.FRAME_LINK_SUFFIXES)

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_path = f"/{robot}"
    root_prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root_prim)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    articulation.CreateEnabledSelfCollisionsAttr().Set(bool(cfg.SELF_COLLISION))
    articulation.CreateSolverPositionIterationCountAttr().Set(
        int(cfg.SOLVER_POSITION_ITERATIONS))
    articulation.CreateSolverVelocityIterationCountAttr().Set(
        int(cfg.SOLVER_VELOCITY_ITERATIONS))
    articulation.CreateArticulationEnabledAttr().Set(True)

    UsdGeom.Scope.Define(stage, f"{root_path}/{cfg.JOINTS_SCOPE}")
    mesh_scope = UsdGeom.Scope.Define(stage, f"{root_path}/{cfg.GEOMETRY_SCOPE}")
    mesh_library = build_mesh_library(stage, urdf_path, bodies, str(mesh_scope.GetPath()))
    materials = build_materials(stage, root_path, wants_tactile)

    rest = _rest_transforms(root_body, movable)
    body_prims = {}
    for body in bodies:
        prim = _add_body(stage, root_path, body, rest[body["name"]])
        body_prims[body["name"]] = prim
        _add_meshes(stage, prim, body, mesh_library, materials)
        _add_frames(stage, prim, body["name"], frames)

    for joint in movable:
        _add_joint(stage, root_path, joint, body_prims)
    mount_joints = (_add_mount(stage, root_path, root_body, body_prims, rest)
                    if wants_mount else [])

    pads = {}
    taxels = {}
    if wants_tactile:
        if tactile_urdf_path is None or not Path(tactile_urdf_path).exists():
            raise SystemExit(f"missing tactile URDF for {robot}")
        pads = load_points(tactile_urdf_path, cfg.TACTILE_SURFACE_INSET,
                           cfg.TACTILE_INSIDE_STEP, cfg.TACTILE_STRIDE)
        for info in _add_tactile(stage, root_path, body_prims, pads, placements,
                                 materials):
            taxels[info.pop("name")] = info

    stage.Save()
    print(f"  {path.name}: {len(bodies)} bodies, {len(movable)} joints, "
          f"{len({name for body in bodies for _k, name, _t in body['meshes']})} meshes, "
          f"{len(taxels)} tactile bodies"
          f"{'' if wants_mount else ', no mount'}")

    return {
        "robot": robot,
        "approximation": cfg.MESH_STYLE_APPROXIMATION,
        "usd": path.name,
        "bodies": [body["name"] for body in bodies],
        "root_body": root_body,
        "has_tactile": wants_tactile,
        "has_mount": wants_mount,
        "tactile_bodies": len(taxels),
        "tactile": taxels,
        "actuators": {"kp": float(cfg.DEFAULT_KP), "kv": float(cfg.DEFAULT_KV),
                      "armature": float(cfg.JOINT_ARMATURE)},
        "joints": [
            {
                "name": joint["name"],
                "parent": joint["parent"],
                "child": joint["child"],
                "lower": float(joint["limit"].get("lower", 0.0)),
                "upper": float(joint["limit"].get("upper", 0.0)),
                "effort": float(joint["limit"].get("effort", 1.0)),
                "velocity": float(joint["limit"].get("velocity", 0.0)),
                "mimic": joint["mimic"],
            }
            for joint in movable
        ],
        "frames": [{"name": frame["name"], "body": frame["body"]} for frame in frames],
        "mount_joints": mount_joints,
    }


def write_manifest(asset_dir, manifest):
    (Path(asset_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
