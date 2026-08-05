"""Generate ``assets/pick_cube.usd`` from ``task/pick_cube.task.yaml``.

The task scene is the robot plus a cube, two cameras, and a few physics
overrides. It is authored with **plain USD** -- no ``SimulationApp`` -- so this
runs in seconds, needs no GPU, and does not disturb anyone else using Isaac on
the same machine.

Two rules this file exists to enforce:

* ``omx_f.usd`` is **never modified**. It is part of the sim<->real calibration
  chain that the user's ``jog`` / ``to_sim`` / ``to_real`` tools depend on.
  Everything here is an override layered on top of it.
* It is brought in as a **sublayer, not a reference**. ``omx_f.usd`` is a
  flattened stage: its root layer holds 16 ``Flattened_Prototype_*`` specs plus
  root-scope ``/visuals``, ``/colliders`` and ``/meshes``, and
  ``link6/collisions`` points at those prototypes internally. Referencing the
  default prim would leave all of that dangling and **silently** drop every bit
  of visual and collision geometry -- the stage still opens, it is just empty.

The generated USD is a build artifact and is gitignored; regenerate it rather
than copying it between machines.

    ./python.sh sim/build_scene.py [--force]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_config  # noqa: E402

# USD cameras express aperture in tenths of a scene unit; 20.955 is the
# 36 mm-film convention Isaac's own cameras use, so FOV maths lines up with
# what the GUI shows.
HORIZONTAL_APERTURE = 20.955


def focal_length_for_fov(fov_deg: float, aperture: float = HORIZONTAL_APERTURE) -> float:
    return (aperture / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def define_camera(stage, path, spec, parent_relative=False):
    """Author a UsdGeom.Camera whose FOV matches ``spec``."""
    cam = UsdGeom.Camera.Define(stage, path)
    w, h = spec["resolution"]
    aperture_v = HORIZONTAL_APERTURE * (h / w)
    cam.CreateFocalLengthAttr(focal_length_for_fov(spec["horizontal_fov_deg"]))
    cam.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE)
    cam.CreateVerticalApertureAttr(aperture_v)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*spec["clipping"]))

    # Both cameras are aimed with look-at rather than hand-turned Euler angles.
    # The first wrist camera was written as a rotation and ended up facing
    # backwards at the arm's base, then 13 degrees too high once the sign was
    # fixed. Naming the point to look at cannot go wrong in either way.
    #
    # USD cameras face -Z, and SetLookAt returns a world->view matrix, so the
    # camera's transform is its inverse. For a parented camera the same maths is
    # done in the parent's frame.
    eye, target = ((spec["translate"], spec["look_at"]) if parent_relative
                   else (spec["eye"], spec["target"]))
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)
    )
    UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
    return cam


def define_cube(stage, path, spec, material):
    """A dynamic, collidable cube with an explicit high-friction material.

    The explicit material matters: Isaac's default physics material is
    ``static_friction=0.2``, which is far too slippery for a 0.14 N pinch.
    """
    size = spec["size"]
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*spec["color"])])
    # Spawned properly by scene.py each episode; this is just a sane resting
    # place so the generated stage opens with the cube visible on the ground.
    # Both ops are authored even though the resting pose needs neither rotated
    # nor animated: an episode reset over the ROS prim service can only *write*
    # attributes that already exist, so a cube with no orient op could never be
    # given a yaw from outside Isaac.
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(0.21, 0.0, size / 2))
    x.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    prim = cube.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec["mass"])
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    return cube


def define_physics_material(stage, path, spec):
    mat = UsdShade.Material.Define(stage, path)
    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(spec["static_friction"])
    phys.CreateDynamicFrictionAttr(spec["dynamic_friction"])
    phys.CreateRestitutionAttr(spec["restitution"])
    return mat


def override_attr(stage, prim_path, attr_name, value, report):
    """Override an attribute that lives in the sublayer.

    The root layer is stronger than its sublayers, so authoring here wins. If
    the attribute is not present in the composed stage the override would be
    inert, so say so loudly rather than leaving a silent no-op behind.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        report.append(f"FAIL  prim 不存在: {prim_path}")
        return False
    attr = prim.GetAttribute(attr_name)
    if not attr:
        report.append(f"FAIL  {prim_path} 沒有屬性 {attr_name}")
        return False
    before = attr.Get()
    attr.Set(value)
    report.append(f"  ok  {prim_path}.{attr_name}: {before} -> {value}")
    return True


def editable_source(stage, prim_path):
    """The prim you can actually author on for ``prim_path``.

    Most of this robot's geometry is reached through instance proxies, which
    cannot be overridden. The proxy's contents come from a reference to a
    ``/Flattened_Prototype_N`` prim; that one is a normal prim in this stage's
    layer stack, so an override there composes through.

    Do not guess the prototype path by name. ``/visuals/link5/mesh_1`` looks like
    the source of ``/omx_f/link5/visuals/mesh_1`` and is not -- it is a separate,
    unused copy, so authoring there is a silent no-op. Ask the prim index.

    Returns ``(path, None)`` for a prim that is already editable, or
    ``(resolved_path, reason)`` where ``reason`` is set when resolution failed.
    """
    from pxr import Pcp

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None, f"prim 不存在: {prim_path}"
    if not prim.IsInstanceProxy():
        return prim_path, None

    suffix = []
    node = prim
    while node.IsInstanceProxy():
        suffix.append(node.GetName())
        node = node.GetParent()

    refs = [a for a in node.GetPrimIndex().rootNode.children
            if a.arcType == Pcp.ArcTypeReference]
    if not refs:
        return None, f"{node.GetPath()} 是 instance 但找不到 reference 弧"
    arc = refs[0]
    if arc.layerStack.identifier.rootLayer != stage.GetRootLayer():
        return None, (f"{prim_path} 的 reference 來自別的 layer stack "
                      f"({arc.layerStack.identifier.rootLayer.identifier}),"
                      "在這裡覆寫不會生效")

    resolved = arc.path
    for name in reversed(suffix):
        resolved = resolved.AppendChild(name)
    return str(resolved), None


def hide_prim(stage, prim_path, report):
    """Make a prim invisible in this scene only.

    Authored as an ``over`` in the root layer, so the robot asset keeps the prim
    and every other tool that opens it still sees it. Give it the path you see in
    the Stage tree; instance proxies are resolved for you.
    """
    resolved, why = editable_source(stage, prim_path)
    if why:
        report.append(f"FAIL  隱藏 {prim_path}: {why}")
        return False
    prim = stage.GetPrimAtPath(resolved)
    if not prim.IsA(UsdGeom.Imageable):
        report.append(f"FAIL  {resolved} 不是 Imageable,沒有 visibility")
        return False
    UsdGeom.Imageable(prim).MakeInvisible()
    via = "" if resolved == prim_path else f" (經由 {resolved})"
    report.append(f"  ok  {prim_path} -> invisible{via}")
    return True


def add_cube_tf_nodes(stage, cfg, report):
    """Put the cube's pose on ROS, in both directions.

    ``omx_f.usd``'s ActionGraph already publishes ``/joint_states`` and subscribes
    ``/joint_command`` -- that is how ``jog`` drives the twin. So the *arm* was
    always reachable from outside Isaac, but nothing could say where the cube is
    or put it somewhere, and that is most of what running an episode means.

    Two stock nodes close the gap, and they are deliberately different kinds:

    * **reading** goes over ``tf2_msgs`` -- a stream, standard messages, and the
      pose is wanted every tick anyway.
    * **writing** goes over ``ROS2ServicePrim``'s ``set_prim_attribute`` -- a
      request, used a handful of times per episode.

    ``ROS2SubscribeTransformTree`` looks like the obvious writer and is not: it
    modifies **articulation roots**, and a loose rigid body is not one. Measured
    2026-08-05 -- Isaac subscribes to the topic, receives the transform, and the
    cube does not move, under four different frame-naming conventions. Writing
    ``xformOp:translate`` *does* move it mid-simulation (also measured), which is
    what the service does.
    """
    graph = f"{cfg.robot_root}/ActionGraph"
    if not stage.GetPrimAtPath(graph).IsValid():
        report.append(f"FAIL  找不到 {graph} —— sublayer 沒帶進 ActionGraph")
        return False
    tick = Sdf.Path(f"{graph}/on_playback_tick.outputs:tick")
    cube = f"{cfg.task_root}/cube"

    pub = stage.DefinePrim(f"{graph}/ros2_publish_cube_tf", "OmniGraphNode")
    pub.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(
        "isaacsim.ros2.bridge.ROS2PublishTransformTree")
    pub.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int).Set(1)
    pub.CreateAttribute("inputs:execIn", Sdf.ValueTypeNames.UInt).AddConnection(tick)
    pub.CreateAttribute("inputs:timeStamp", Sdf.ValueTypeNames.Double).AddConnection(
        Sdf.Path(f"{graph}/isaac_read_simulation_time.outputs:simulationTime"))
    pub.CreateAttribute("inputs:topicName", Sdf.ValueTypeNames.String).Set(
        cfg["ros"]["cube_tf_topic"])
    pub.CreateRelationship("inputs:targetPrims").SetTargets([Sdf.Path(cube)])
    pub.CreateAttribute("ui:nodegraph:node:pos", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(631.0, 120.0))

    svc = stage.DefinePrim(f"{graph}/ros2_service_prim", "OmniGraphNode")
    svc.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(
        "isaacsim.ros2.bridge.ROS2ServicePrim")
    svc.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int).Set(1)
    svc.CreateAttribute("inputs:execIn", Sdf.ValueTypeNames.UInt).AddConnection(tick)
    # Every name is authored explicitly. A node written straight into USD does
    # **not** pick up the .ogn defaults -- an unauthored string input reads as
    # empty, and an empty service name advertises nothing at all. The first
    # version of this authored only node:type and execIn, and Isaac ran it
    # happily while offering no services.
    for port, value in (
            ("getAttributeServiceName", "get_prim_attribute"),
            ("getAttributesServiceName", "get_prim_attributes"),
            ("setAttributeServiceName", "set_prim_attribute"),
            ("primsServiceName", "get_prims"),
            ("nodeNamespace", ""),
            ("qosProfile", "")):
        svc.CreateAttribute(f"inputs:{port}", Sdf.ValueTypeNames.String).Set(value)
    svc.CreateAttribute("inputs:context", Sdf.ValueTypeNames.UInt64).Set(0)
    svc.CreateAttribute("ui:nodegraph:node:pos", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(631.0, 260.0))

    report.append(f"  ok  {graph}/ros2_publish_cube_tf -> {cfg['ros']['cube_tf_topic']}"
                  f" ({cfg['ros']['cube_frame']})")
    report.append(f"  ok  {graph}/ros2_service_prim    -> get/set_prim_attribute")
    return True


def add_camera_nodes(stage, cfg, report):
    """Publish both cameras on ROS.

    Needed twice over: M4 records from these, and until they exist the only way
    to see why a grasp failed is to describe it in numbers. A render product has
    to be created for each camera first -- ``ROS2CameraHelper`` publishes *a
    render product*, not a camera prim.

    Resolution comes from the same ``cameras.<name>.resolution`` the offline
    renders use, so what arrives over ROS and what ``preview_cameras.py`` writes
    are the same image.
    """
    graph = f"{cfg.robot_root}/ActionGraph"
    tick = Sdf.Path(f"{graph}/on_playback_tick.outputs:tick")
    cams = cfg["cameras"]
    ros = cfg["ros"]
    y = 400.0

    for name, prim_path in (("front", f"{cfg.task_root}/cam_front"),
                            ("wrist", f"{cams['wrist']['parent']}/cam_wrist")):
        if not stage.GetPrimAtPath(prim_path).IsValid():
            report.append(f"FAIL  找不到相機 {prim_path}")
            return False
        w, h = cams[name]["resolution"]

        rp = stage.DefinePrim(f"{graph}/render_product_{name}", "OmniGraphNode")
        rp.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(
            "isaacsim.core.nodes.IsaacCreateRenderProduct")
        rp.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int).Set(1)
        rp.CreateAttribute("inputs:execIn", Sdf.ValueTypeNames.UInt).AddConnection(tick)
        rp.CreateAttribute("inputs:width", Sdf.ValueTypeNames.UInt).Set(int(w))
        rp.CreateAttribute("inputs:height", Sdf.ValueTypeNames.UInt).Set(int(h))
        rp.CreateAttribute("inputs:enabled", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateRelationship("inputs:cameraPrim").SetTargets([Sdf.Path(prim_path)])
        rp.CreateAttribute("ui:nodegraph:node:pos", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(300.0, y))

        pub = stage.DefinePrim(f"{graph}/ros2_camera_{name}", "OmniGraphNode")
        pub.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(
            "isaacsim.ros2.bridge.ROS2CameraHelper")
        pub.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int).Set(2)
        # Chained off the render product's execOut, not off the tick: the
        # product must exist before anything tries to publish it.
        pub.CreateAttribute("inputs:execIn", Sdf.ValueTypeNames.UInt).AddConnection(
            Sdf.Path(f"{graph}/render_product_{name}.outputs:execOut"))
        pub.CreateAttribute("inputs:renderProductPath", Sdf.ValueTypeNames.Token
                            ).AddConnection(
            Sdf.Path(f"{graph}/render_product_{name}.outputs:renderProductPath"))
        pub.CreateAttribute("inputs:topicName", Sdf.ValueTypeNames.String).Set(
            ros["camera_topics"][name])
        pub.CreateAttribute("inputs:type", Sdf.ValueTypeNames.Token).Set("rgb")
        pub.CreateAttribute("inputs:frameId", Sdf.ValueTypeNames.String).Set(f"cam_{name}")
        pub.CreateAttribute("inputs:nodeNamespace", Sdf.ValueTypeNames.String).Set("")
        pub.CreateAttribute("inputs:qosProfile", Sdf.ValueTypeNames.String).Set("")
        pub.CreateAttribute("inputs:frameSkipCount", Sdf.ValueTypeNames.UInt).Set(
            int(ros.get("camera_frame_skip", 0)))
        pub.CreateAttribute("inputs:enabled", Sdf.ValueTypeNames.Bool).Set(True)
        pub.CreateAttribute("ui:nodegraph:node:pos", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(631.0, y))

        report.append(f"  ok  {graph}/ros2_camera_{name} -> {ros['camera_topics'][name]}"
                      f"  {w}x{h}")
        y += 140.0
    return True


def build(cfg: task_config.Config, force: bool) -> int:
    out = cfg.scene_usd
    if out.exists() and not force:
        print(f"{out} 已存在。要重建請加 --force。")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)

    # Relative so the pair moves together, and resolved from wherever
    # task_config found omx_bridge_image -- see paths.omx_bridge in the task
    # yaml for the candidates tried.
    rel = os.path.relpath(cfg.robot_usd, out.parent)

    stage = Usd.Stage.CreateNew(str(out))
    stage.GetRootLayer().subLayerPaths = [rel]
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    robot = stage.GetPrimAtPath(cfg.robot_root)
    if not robot or not robot.IsValid():
        print(f"sublayer 沒帶進 {cfg.robot_root} —— 檢查 {rel}")
        return 1
    stage.SetDefaultPrim(robot)

    task_root = cfg.task_root
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, task_root)

    material = define_physics_material(
        stage, f"{task_root}/PhysicsMaterials/cube", cfg["cube"]
    )
    define_cube(stage, f"{task_root}/cube", cfg["cube"], material)

    cams = cfg["cameras"]
    define_camera(stage, f"{task_root}/cam_front", cams["front"])
    wrist = cams["wrist"]
    define_camera(stage, f"{wrist['parent']}/cam_wrist", wrist, parent_relative=True)

    # ── overrides on the sublayer's prims ───────────────────────────────
    report = []
    ov = cfg["overrides"]
    for group in ("arm_drive", "gripper_drive"):
        g = ov[group]
        for j in g["joints"]:
            p = f"{cfg.robot_root}/joints/{j}"
            override_attr(stage, p, "drive:angular:physics:stiffness",
                          g["stiffness"], report)
            override_attr(stage, p, "drive:angular:physics:damping",
                          g["damping"], report)
    mj = ov["mimic_joint"]
    mp = f"{cfg.robot_root}/joints/{mj['joint']}"
    override_attr(stage, mp, "physxMimicJoint:rotZ:naturalFrequency",
                  mj["natural_frequency"], report)
    override_attr(stage, mp, "physxMimicJoint:rotZ:dampingRatio",
                  mj["damping_ratio"], report)

    fc = ov["finger_colliders"]
    for p in fc["prims"]:
        override_attr(stage, p, "physics:approximation", fc["approximation"], report)

    hidden_ok = all(hide_prim(stage, p, report) for p in ov["hide"]["prims"])
    service_ok = add_cube_tf_nodes(stage, cfg, report)
    camera_ok = add_camera_nodes(stage, cfg, report)

    stage.GetRootLayer().Save()

    # ── verify ─────────────────────────────────────────────────────────
    check = Usd.Stage.Open(str(out))
    prims = list(check.Traverse())
    # Hold the stage in a named variable: a temporary Stage is collected before the
    # traversal is consumed, and every prim in the range expires under you.
    base_stage = Usd.Stage.Open(str(cfg.robot_usd))
    base = list(base_stage.Traverse())
    added = [
        str(p.GetPath()) for p in prims
        if str(p.GetPath()).startswith(task_root) or "cam_wrist" in str(p.GetPath())
    ]
    deps = [d for d in check.GetRootLayer().GetCompositionAssetDependencies() if d]

    print(f"寫出 {out}")
    print("\n".join(report))
    print(f"\n  base   prims: {len(base)}")
    print(f"  scene  prims: {len(prims)}  (+{len(prims) - len(base)})")
    print(f"  新增: {', '.join(added)}")
    print(f"  外部相依: {deps}")

    ok = True
    if len(prims) < len(base):
        print("FAIL  prim 數變少了 —— sublayer 沒生效,幾何遺失")
        ok = False
    if deps != [rel]:
        print(f"FAIL  外部相依應該只有 {rel}")
        ok = False
    for must in (f"{task_root}/cube", f"{task_root}/cam_front",
                 f"{wrist['parent']}/cam_wrist"):
        if not check.GetPrimAtPath(must).IsValid():
            print(f"FAIL  少了 {must}")
            ok = False
    if not (hidden_ok and service_ok and camera_ok):
        ok = False
    for p in ov["hide"]["prims"]:
        # Check the path that was **asked** for, on a fresh open -- not the path
        # that was written to. Checking the latter passes even when the override
        # landed on an unrelated prim, which is exactly how the first version of
        # this hid the wrong copy of the marker and still reported success.
        target = check.GetPrimAtPath(p)
        vis = (UsdGeom.Imageable(target).ComputeVisibility()
               if target and target.IsValid() else "prim 不存在")
        if vis != UsdGeom.Tokens.invisible:
            print(f"FAIL  {p} 重開後仍然可見 ({vis})")
            ok = False
    print("\nRESULT:", "場景可用" if ok else "場景有問題,不要拿去用")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="覆蓋既有的 pick_cube.usd")
    args = ap.parse_args()
    try:
        cfg = task_config.load()
    except task_task_config.ConfigError as exc:
        print(f"[build_scene] {exc}")
        return 1
    print(cfg.summary(), "\n")
    return build(cfg, args.force)


if __name__ == "__main__":
    sys.exit(main())
