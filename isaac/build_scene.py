"""Generate ``assets/pick_cube.usd`` from ``task/pick_cube.task.yaml``.

The task scene is the robot plus a cube, five cameras, and a few physics
overrides. It is authored with **plain USD** -- no ``SimulationApp`` -- so this
runs in seconds, needs no GPU, and does not disturb anyone else using Isaac on
the same machine.

Two rules this file exists to enforce:

* ``omx_f.usd`` is **never modified from here**. It is part of the sim<->real
  calibration chain that ``jog`` / ``to_sim`` / ``to_real`` depend on, and it is
  owned by the omx_bridge_image repo -- which does change it (0.2.0 added a
  hidden ``/ik_target`` prim, a TF publisher and a ``ROS2ServicePrim``). What is
  authored here is only ever an override layered on top.

  One consequence to keep in mind: this scene **inherits** those additions, so
  ``add_cube_tf_nodes`` skips the ``ROS2ServicePrim`` when the sublayer already
  supplies one. Two of them advertise the same four service names.
* It is brought in as a **sublayer, not a reference**. ``omx_f.usd`` is a
  flattened stage whose root layer holds the ``Flattened_Prototype_*`` specs that
  ``link6/collisions`` points at. A reference to the default prim leaves those
  dangling and silently drops every bit of visual and collision geometry -- the
  stage still opens, it is just empty.

The generated USD is a build artifact and is gitignored; regenerate it rather
than copying it between machines.

    bash isaac/isaac_python.sh isaac/build_scene.py [--force]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
# 共用層(task_config / ik / spawn / moveit_ik)是 src/omx_vla_app 這個 ROS
# package。容器裡 colcon 已經裝好了,這行是 no-op;host 上的 Isaac 沒有 source
# 過 workspace,靠這行才找得到。三個環境因此可以用同一種 import 寫法。
sys.path.insert(0, str(REPO_ROOT / "src" / "omx_vla_app"))
sys.path.insert(0, str(HERE))            # texture.py 是 isaac/ 自己的
from omx_vla_app import task_config  # noqa: E402
import texture  # noqa: E402

# USD cameras express aperture in tenths of a scene unit; 20.955 is the
# 36 mm-film convention Isaac's own cameras use, so FOV maths lines up with
# what the GUI shows.
HORIZONTAL_APERTURE = 20.955


def focal_length_for_fov(fov_deg: float, aperture: float = HORIZONTAL_APERTURE) -> float:
    return (aperture / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def camera_prim_name(name):
    """``rear_right`` -> ``cam_rear_right``.

    One place decides how a camera's config key becomes a prim name, because
    three separate pieces of this file have to agree on it: the prim author, the
    render-product node and the verification pass.
    """
    return f"cam_{name}"


def define_camera(stage, path, spec, parent_relative=False):
    """Author a UsdGeom.Camera whose FOV matches ``spec``."""
    cam = UsdGeom.Camera.Define(stage, path)
    w, h = spec["resolution"]
    aperture_v = HORIZONTAL_APERTURE * (h / w)
    cam.CreateFocalLengthAttr(focal_length_for_fov(spec["horizontal_fov_deg"]))
    cam.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE)
    cam.CreateVerticalApertureAttr(aperture_v)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*spec["clipping"]))

    # Aimed by naming the point to look at rather than by a hand-turned rotation.
    # USD cameras face -Z and SetLookAt returns a world->view matrix, so the
    # camera's transform is its inverse; for a parented camera the same maths is
    # done in the parent's frame.
    eye, target = ((spec["translate"], spec["look_at"]) if parent_relative
                   else (spec["eye"], spec["target"]))
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)
    )
    UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
    return cam


# The six quads, wound counter-clockwise so each face's normal points outward,
# in the order texture.cube_face_uvs() assigns atlas cells: +X -X +Y -Y +Z -Z.
BOX_FACES = ((1, 2, 6, 5), (3, 0, 4, 7), (2, 3, 7, 6),
             (0, 1, 5, 4), (4, 5, 6, 7), (3, 2, 1, 0))


def box_mesh_points(size):
    """The 8 corners of an axis-aligned cube centred on the origin."""
    h = size / 2.0
    return [Gf.Vec3f(x * h, y * h, z * h)
            for x, y, z in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))]


def define_cube(stage, path, spec, material, uvs=None):
    """A dynamic, collidable cube with an explicit high-friction material.

    The explicit material matters: Isaac's default physics material is
    ``static_friction=0.2``, which is far too slippery for a 0.14 N pinch.

    ⚠️ Authored as a **Mesh, not a UsdGeom.Cube**, because a procedural Cube
    carries no UVs and therefore cannot take a texture. The cube needs one: a
    uniformly coloured box is rotationally symmetric on camera, so
    ``spawn.yaw_deg`` would vary the expert's actions without varying the
    pixels. See isaac/texture.py.

    The physics is unchanged by that switch. ``boundingCube`` is an *exact*
    approximation for a box (the local AABB is the box), so mass, friction and
    contact behaviour match what the previous UsdGeom.Cube produced -- verified
    by re-running the expert, not assumed.
    """
    size = spec["size"]
    cube = UsdGeom.Mesh.Define(stage, path)
    cube.CreatePointsAttr(box_mesh_points(size))
    cube.CreateFaceVertexCountsAttr([4] * 6)
    cube.CreateFaceVertexIndicesAttr([i for face in BOX_FACES for i in face])
    cube.CreateExtentAttr([Gf.Vec3f(-size / 2, -size / 2, -size / 2),
                           Gf.Vec3f(size / 2, size / 2, size / 2)])
    # ⚠️ Without this the renderer applies Catmull-Clark and the cube comes out
    # as a rounded blob -- silently, and only in the render, so the physics still
    # behaves like a box while every camera shows a ball.
    cube.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if uvs is not None:
        st = UsdGeom.PrimvarsAPI(cube).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying)
        st.Set([Gf.Vec2f(*uv) for uv in uvs])
    # Kept as a fallback: if the texture fails to resolve, the cube stays red
    # rather than defaulting to grey and becoming invisible against the floor.
    cube.CreateDisplayColorAttr([Gf.Vec3f(*spec["color"])])
    # Spawned properly by scene.py each episode; this is just a sane resting
    # place so the generated stage opens with the cube visible on the ground.
    # Both ops are authored even though the resting pose needs neither: a reset
    # over the prim service can only **write attributes that already exist**, so
    # a cube with no orient op could never be given a yaw from outside Isaac.
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(0.21, 0.0, size / 2))
    x.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    prim = cube.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    # Exact for a box, and cheap. A Mesh without this defaults to a triangle
    # mesh collider, which PhysX will not let a dynamic body use.
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
        UsdPhysics.Tokens.boundingCube)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec["mass"])
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    return cube


def define_textured_material(stage, path, texture_file, roughness=0.75):
    """A UsdPreviewSurface driven by one texture, read through the ``st`` primvar.

    ``wrapS/wrapT`` are **clamp**, not repeat. Both surfaces here carry UVs that
    span 0..1 exactly once, so clamping is a no-op that also guarantees a
    mistake in the UVs shows up as a stretched edge rather than as a tiled
    pattern -- and a tiled floor is the one thing this must not produce (a
    periodic signal correlated with position aliases every tile-width).
    """
    mat = UsdShade.Material.Define(stage, path)

    reader = UsdShade.Shader.Define(stage, f"{path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, f"{path}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def define_table(stage, path, spec, material):
    """A textured quad over the work area. **Visual only -- no collider.**

    ⚠️ It does not replace the floor. ``/Environment/ground`` comes in from the
    sublayer, is 100 m across, and carries UVs spanning 0..1 over that whole
    span -- so a texture on it would give the 0.5 m work area about five pixels.
    Physics keeps using that floor; this quad only supplies what the cameras see.

    Sitting ``lift`` above z=0 avoids z-fighting with the floor underneath. It
    has to be **above**, not below, or the opaque floor hides it; the cube then
    appears to sink by that much, which at 0.5 mm against a 25 mm cube is 2% and
    not visible. Nothing physical touches it, so the resting height of the cube
    is unchanged.
    """
    half = spec["size"] / 2.0
    cx, cy = spec["center"][0], spec["center"][1]
    lift = spec["lift"]
    quad = UsdGeom.Mesh.Define(stage, path)
    quad.CreatePointsAttr([Gf.Vec3f(cx - half, cy - half, lift),
                           Gf.Vec3f(cx + half, cy - half, lift),
                           Gf.Vec3f(cx + half, cy + half, lift),
                           Gf.Vec3f(cx - half, cy + half, lift)])
    quad.CreateFaceVertexCountsAttr([4])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    quad.CreateExtentAttr([Gf.Vec3f(cx - half, cy - half, lift),
                           Gf.Vec3f(cx + half, cy + half, lift)])
    quad.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(quad).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    prim = quad.GetPrim()
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material)
    return quad


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

    **Do not guess the prototype path by name.** ``/visuals/link5/mesh_1`` looks
    like the source of ``/omx_f/link5/visuals/mesh_1`` and is a separate, unused
    copy; authoring there is a silent no-op. Ask the prim index.

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
    modifies **articulation roots**, and a loose rigid body is not one. Isaac
    subscribes, receives the transform, and the cube does not move -- under any
    frame naming. Writing ``xformOp:translate`` does move it mid-simulation,
    which is what the service does.
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

    # The sublayer may already carry one (omx_f.usd gained it with the drag-to-
    # move debug target). Two of these advertise the same four service names, so
    # adding a second is not merely redundant -- it is a conflict.
    if stage.GetPrimAtPath(f"{graph}/ros2_service_prim").IsValid():
        report.append(f"  --  {graph}/ros2_service_prim 已由 sublayer 提供,不重複加")
        return True

    svc = stage.DefinePrim(f"{graph}/ros2_service_prim", "OmniGraphNode")
    svc.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(
        "isaacsim.ros2.bridge.ROS2ServicePrim")
    svc.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int).Set(1)
    svc.CreateAttribute("inputs:execIn", Sdf.ValueTypeNames.UInt).AddConnection(tick)
    # Every name is authored explicitly: a node written straight into USD does
    # **not** pick up the .ogn defaults. An unauthored string input reads as
    # empty, an empty service name advertises nothing, and Isaac does not
    # complain.
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
    """Publish every camera in ``ros.camera_topics`` on ROS.

    A render product has to be created for each camera first --
    ``ROS2CameraHelper`` publishes *a render product*, not a camera prim.

    ⚠️ Each camera costs render time on every tick. Two 640x480 cameras
    published at ~57 Hz on the RTX Pro 6000; the rate falls roughly in
    proportion as cameras are added, and once it drops below the 30 Hz control
    loop the dataset starts repeating pixels between frames. That is visible in
    `meta.json`'s `image_age_s` and in the converter's "唯一影像/幀" figure --
    measure it after changing this list rather than assuming.
    """
    graph = f"{cfg.robot_root}/ActionGraph"
    tick = Sdf.Path(f"{graph}/on_playback_tick.outputs:tick")
    cams = cfg["cameras"]
    ros = cfg["ros"]
    y = 400.0

    for name in ros["camera_topics"]:
        spec = cams[name]
        parent = spec.get("parent")
        root = parent if parent else cfg.task_root
        prim_path = f"{root}/{camera_prim_name(name)}"
        if not stage.GetPrimAtPath(prim_path).IsValid():
            report.append(f"FAIL  找不到相機 {prim_path}")
            return False
        w, h = spec["resolution"]

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

    # ── textures ───────────────────────────────────────────────────────
    # Generated here rather than shipped: a seed makes them reproducible, and
    # PNG bytes do not belong in a source repo. Written beside the stage so the
    # asset paths in the material can stay relative and the scene remains
    # movable between machines.
    tspec = cfg["textures"]
    tex_dir = out.parent / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tbl = tspec["table"]
    if tbl.get("style", "wood") == "wood":
        texture.wood_texture(
            str(tex_dir / "table.png"), seed=tspec["seed"],
            light=tuple(tbl["wood_light"]), dark=tuple(tbl["wood_dark"]),
            rings=tbl["rings"], distortion=tbl["distortion"], knots=tbl["knots"],
            sharpness=tbl.get("sharpness", 1.6))
    else:
        texture.floor_texture(
            str(tex_dir / "table.png"), seed=tspec["seed"],
            base=tuple(tbl["base_color"]), contrast=tbl["contrast"],
            speckles=tbl["speckles"], speckle_radius=tuple(tbl["speckle_radius"]))
    texture.cube_texture(str(tex_dir / "cube.png"), seed=tspec["seed"])

    table_mat = define_textured_material(
        stage, f"{task_root}/Looks/table", "textures/table.png")
    define_table(stage, f"{task_root}/table", tspec["table"], table_mat)

    material = define_physics_material(
        stage, f"{task_root}/PhysicsMaterials/cube", cfg["cube"]
    )
    cube_mat = define_textured_material(
        stage, f"{task_root}/Looks/cube", "textures/cube.png", roughness=0.55)
    cube = define_cube(stage, f"{task_root}/cube", cfg["cube"], material,
                       uvs=texture.cube_face_uvs())
    # The visual binding is separate from the physics one above -- that goes on
    # the "physics" purpose, this on the default, so they do not displace
    # each other.
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(cube_mat)

    # One prim per entry in `ros.camera_topics` -- that dict is the camera list,
    # here and in the recorder, the converter and eval.py. A camera with a
    # `parent` rides on the arm and is placed in the parent's frame; everything
    # else is a fixed environment camera placed in world coordinates.
    cams = cfg["cameras"]
    camera_paths = {}
    for name in cfg["ros"]["camera_topics"]:
        spec = cams[name]
        parent = spec.get("parent")
        root = parent if parent else task_root
        camera_paths[name] = f"{root}/{camera_prim_name(name)}"
        define_camera(stage, camera_paths[name], spec,
                      parent_relative=bool(parent))

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
    # Everything under task_root is ours, plus any camera that rides on the arm
    # and therefore lives under the robot's own prims.
    arm_mounted = {p for p in camera_paths.values() if not p.startswith(task_root)}
    added = [
        str(p.GetPath()) for p in prims
        if str(p.GetPath()).startswith(task_root) or str(p.GetPath()) in arm_mounted
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
    for must in [f"{task_root}/cube", *camera_paths.values()]:
        if not check.GetPrimAtPath(must).IsValid():
            print(f"FAIL  少了 {must}")
            ok = False
    if not (hidden_ok and service_ok and camera_ok):
        ok = False
    for p in ov["hide"]["prims"]:
        # Check the path that was **asked** for, on a fresh open -- not the path
        # that was written to. Checking the latter passes even when the override
        # landed on an unrelated prim.
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
    except task_config.ConfigError as exc:
        print(f"[build_scene] {exc}")
        return 1
    print(cfg.summary(), "\n")
    return build(cfg, args.force)


if __name__ == "__main__":
    sys.exit(main())
