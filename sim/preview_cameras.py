"""Render what each camera actually sees, at the moments that matter.

Camera placement is a *visual* judgement -- whether the cube is framed, whether
the fingers eat the frame, whether the approach is visible -- and none of that
shows up in the angles and distances that build_scene.py verifies. The wrist
camera was aimed three times off geometry alone before anyone looked through it.

Two moments are captured because they fail differently:

* ``far``   -- arm at home, cube out on the ground. If the cube is not in this
  frame the policy cannot begin an approach from the image.
* ``grasp`` -- cube at the grasp point, gripper closed. If the fingers occlude
  it here the policy is blind exactly when precision matters.

    bash sim/isaac_python.sh sim/preview_cameras.py [--out DIR]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))
import app  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SIM_DIR.parent / "data" / "preview",
                    help="PNG 輸出目錄(預設 vla/data/preview)")
    ap.add_argument("--front-dist", type=float, nargs="*", metavar="M",
                    help="額外用這些距離(公尺)再拍一次 cam_front,沿現有視線"
                         "推近或拉遠。用來選構圖,不會改到 yaml。")
    args = ap.parse_args()

    simulation_app = app.start(headless=True)
    sys.path.insert(0, str(SIM_DIR))          # SimulationApp prepends its own paths

    import numpy as np                        # noqa: E402
    import task_config                        # noqa: E402
    from scene import PickCubeScene           # noqa: E402

    cfg = task_config.load()
    scene = PickCubeScene(cfg, with_cameras=True)
    grip = cfg["gripper"]
    gi = cfg.joints.index(grip["joint"])
    q = np.zeros(len(cfg.joints), dtype=np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        print("要存 PNG 需要 pillow: bash sim/isaac_python.sh -m pip install pillow")
        return 1

    def move_front(dist):
        """Slide cam_front along its existing sight line to ``dist`` from the target.

        The Camera sensor reads the prim's transform every frame, so moving the
        prim is enough -- no rebuild. Direction and FOV are left alone: framing
        should be chosen by distance, because the FOV is the part that has to
        match a real D455 later.
        """
        from pxr import Gf, UsdGeom
        import omni.usd

        spec = cfg["cameras"]["front"]
        eye, target = np.array(spec["eye"]), np.array(spec["target"])
        eye = target + (eye - target) / np.linalg.norm(eye - target) * dist
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target),
                                       Gf.Vec3d(0, 0, 1))
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(f"{cfg.task_root}/cam_front")
        # build_scene authors a single transform op, but Camera.initialize()
        # rewrites it as translate+orient, so write whichever is actually there.
        m = view.GetInverse()
        ops = {o.GetOpName(): o for o in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        if "xformOp:transform" in ops:
            ops["xformOp:transform"].Set(m)
        else:
            ops["xformOp:translate"].Set(Gf.Vec3d(*eye))
            quat = m.ExtractRotationQuat()
            orient = ops["xformOp:orient"]
            orient.Set(Gf.Quatf(quat) if isinstance(orient.Get(), Gf.Quatf) else quat)
        half = math.radians(spec["horizontal_fov_deg"]) / 2
        print(f"cam_front 距離 {dist:.2f} m -> 視野寬 {2 * dist * math.tan(half) * 100:.0f} cm"
              f"  方塊約 {cfg['cube']['size'] / (2 * dist * math.tan(half)) * spec['resolution'][0]:.0f} px")
        return eye

    def capture(tag):
        # The render products need a few rendered frames before they hold an
        # image; reading straight after a physics-only step gives back zeros.
        for _ in range(12):
            scene.step(render=True)
        for name, img in scene.observe(render=True)["images"].items():
            out = args.out / f"{tag}_{name}.png"
            Image.fromarray(img.astype("uint8")).save(out)
            print(f"  {out}  {img.shape[1]}x{img.shape[0]}")

    # ── far: can the policy even see the cube to start an approach? ─────
    pos, quat = scene.sample_cube_pose(seed=0)
    scene.reset(seed=0, cube_pose=(pos, quat))
    q[gi] = grip["open"]
    scene.set_targets(q)
    for _ in range(60):
        scene.step()
    print(f"far   方塊 {np.round(pos, 3)}  EE {np.round(scene.ee_position(), 3)}")
    capture("far")

    # ── grasp: is the cube visible with the fingers around it? ──────────
    at = scene.grasp_point()
    scene.place_cube(at)
    q[gi] = grip["grasp"]
    scene.set_targets(q)
    for _ in range(150):
        scene.place_cube(at)
        scene.step()
    print(f"grasp 方塊 {np.round(at, 3)}  夾爪 {scene.joint_positions()[gi]:+.4f} rad")
    capture("grasp")

    # ── optional: the same far moment at other front-camera distances ───
    if args.front_dist:
        scene.reset(seed=0, cube_pose=(pos, quat))
        q[gi] = grip["open"]
        scene.set_targets(q)
        for _ in range(60):
            scene.step()
        for d in args.front_dist:
            move_front(d)
            for _ in range(12):
                scene.step(render=True)
            img = scene.observe(render=True)["images"]["front"]
            out = args.out / f"front_{int(d * 100):03d}cm.png"
            Image.fromarray(img.astype("uint8")).save(out)
            print(f"  {out}")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
