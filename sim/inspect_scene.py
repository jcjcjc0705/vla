"""Hold the scene still and stream it, so you can look at it.

grasp_test.py runs to completion and only then streams, which is useless for
inspection -- by the time you connect, the episode is over. This does the
opposite: it opens the scene, poses it, starts streaming, and then just idles.

    conda deactivate          # Isaac warns about this and means it
    bash sim/isaac_python.sh sim/inspect_scene.py --gripper 0.6 --cube 0.27

Connect with isaacsim-webrtc-streaming-client. What to look for:

* how wide the gap between the fingers actually is, and whether a 25 mm cube
  fits between them
* where the cube ends up relative to the fingers -- between them, past the
  fingertips, or interpenetrating

Click a finger in the Stage tree and read its Property panel to get real
coordinates. That is the number this script cannot work out on its own: the
grasp point was inferred from whole-link bounding boxes, which are dominated by
the mounting body near the pivot rather than the fingertips.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))
import app  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gripper", type=float, default=None,
                    help="gripper_joint_1 的角度(rad)。預設用 task yaml 的 open")
    ap.add_argument("--cube", type=float, default=None,
                    help="把方塊放在 x=這個值(m)、y/z 對齊夾爪。省略則放遠處")
    ap.add_argument("--no-stream", action="store_true", help="只印座標,不開串流")
    args = ap.parse_args()

    simulation_app = app.start(headless=True, livestream=not args.no_stream)
    sys.path.insert(0, str(SIM_DIR))

    import numpy as np                       # noqa: E402
    import task_config                       # noqa: E402
    from scene import PickCubeScene          # noqa: E402
    from pxr import Usd, UsdGeom             # noqa: E402

    cfg = task_config.load()
    scene = PickCubeScene(cfg, with_cameras=False)

    gi = cfg.joints.index(cfg["gripper"]["joint"])
    q = np.zeros(len(cfg.joints), dtype=np.float32)
    q[gi] = cfg["gripper"]["open"] if args.gripper is None else args.gripper
    scene.set_targets(q)

    if args.cube is None:
        scene.place_cube(np.array([2.0, 0.0, 1.0]))     # out of the way
    else:
        scene.place_cube(np.array([args.cube, -0.0016, 0.2106]))

    for _ in range(150):
        scene.step()

    # World-space extents of each finger, for comparison with what you see.
    stage = scene.world.stage
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    print("\n" + "=" * 66)
    print(f"gripper_joint_1  目標 {q[gi]:+.3f}  實際 {scene.joint_positions()[gi]:+.4f} rad")
    for link in ("link6", "link7"):
        r = cache.ComputeWorldBound(stage.GetPrimAtPath(f"/omx_f/{link}")).ComputeAlignedRange()
        lo, hi = r.GetMin(), r.GetMax()
        print(f"  {link} 世界 bbox (mm): "
              f"x[{lo[0]*1000:6.1f},{hi[0]*1000:6.1f}] "
              f"y[{lo[1]*1000:6.1f},{hi[1]*1000:6.1f}] "
              f"z[{lo[2]*1000:6.1f},{hi[2]*1000:6.1f}]")
    print(f"  EE            : {np.round(scene.ee_position() * 1000, 1)} mm")
    print(f"  方塊          : {np.round(scene.cube_pose()[0] * 1000, 1)} mm"
          f"   (邊長 {cfg['cube']['size']*1000:.0f} mm)")
    print("=" * 66)

    if args.no_stream:
        simulation_app.close()
        return 0

    print("\n串流中 —— 用 AppImage 連上來看。Ctrl+C 結束。")
    print("看兩件事:1) 兩指空隙容不容得下方塊  2) 方塊在指間還是指外\n")
    try:
        while simulation_app.is_running():
            scene.step(render=True)
    except KeyboardInterrupt:
        pass
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
