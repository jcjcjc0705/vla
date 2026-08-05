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

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

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

    xf = UsdGeom.Xformable(cam)
    if parent_relative:
        # Wrist camera: a fixed offset from the link it rides on.
        xf.AddTranslateOp().Set(Gf.Vec3d(*spec["translate"]))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*spec["rotate_xyz_deg"]))
    else:
        # World camera: look-at. USD cameras face -Z, and SetLookAt returns a
        # world->view matrix, so the camera's transform is its inverse.
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*spec["eye"]), Gf.Vec3d(*spec["target"]), Gf.Vec3d(0, 0, 1)
        )
        xf.AddTransformOp().Set(view.GetInverse())
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
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0.21, 0.0, size / 2))

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


def build(cfg: task_config.Config, force: bool) -> int:
    out = cfg.scene_usd
    if out.exists() and not force:
        print(f"{out} 已存在。要重建請加 --force。")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)

    # Relative so the pair moves together. vla/ and omx_sim2real/ sit side by
    # side on both machines, so the same relative path resolves on each.
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
    fc = ov["finger_colliders"]
    for p in fc["prims"]:
        override_attr(stage, p, "physics:approximation", fc["approximation"], report)

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
