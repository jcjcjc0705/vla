"""M1: can the gripper actually hold the cube?

This isolates **physics** from kinematics. Reaching the cube is IK's job and IK
is M2, so here the cube is simply placed at the gripper's grasp point and the
gripper is closed on it. If it cannot hold a cube handed straight to it, no
amount of IK later will help.

Nothing here can fail in a way that kills the project -- in simulation the cube's
mass and friction, the drive stiffness and the collider approximation are all
knobs. The point of running it now is that changing any of them **invalidates
recorded data**, and re-recording 200 episodes is the expensive part. So find the
values first, write them into task/pick_cube.task.yaml, then record.

    bash sim/isaac_python.sh sim/grasp_test.py [--livestream] [--png OUT.png]
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
    ap.add_argument("--livestream", action="store_true", help="用 WebRTC 串流出來看")
    ap.add_argument("--png", type=Path, help="存一張兩台相機的畫面")
    args = ap.parse_args()

    simulation_app = app.start(headless=True, livestream=args.livestream)

    # Re-assert our path: SimulationApp prepends Isaac's own directories, and
    # a bare `config` there resolves to cv2/config.py. Generic module names are
    # a hazard in this interpreter, hence task_config rather than config.
    sys.path.insert(0, str(SIM_DIR))

    import numpy as np                       # noqa: E402
    import task_config                       # noqa: E402
    from scene import PickCubeScene          # noqa: E402

    cfg = task_config.load()
    print(cfg.summary(), "\n")
    scene = PickCubeScene(cfg, with_cameras=True)

    grip = cfg["gripper"]
    joints = cfg.joints
    gi = joints.index(grip["joint"])
    q = np.zeros(len(joints), dtype=np.float32)

    def settle(steps, render=False):
        for _ in range(steps):
            scene.step(render=render)

    # ── 1. open the gripper at home, then hand it the cube ─────────────
    scene.reset(seed=0, cube_pose=(np.array([1.0, 0.0, 0.5]), None))  # park it away
    q[gi] = grip["open"]
    scene.set_targets(q)
    settle(60)

    ee = scene.ee_position()
    print(f"夾爪張開後 EE 位置 = {np.round(ee, 4)}")
    scene.place_cube(ee)                     # cube straight into the jaws
    settle(10)

    before = scene.cube_pose()[0].copy()
    print(f"方塊置入 = {np.round(before, 4)}")

    # ── 2. close ───────────────────────────────────────────────────────
    q[gi] = grip["grasp"]
    scene.set_targets(q)
    settle(90)
    after_close = scene.cube_pose()[0].copy()
    d = scene.diagnose()
    print(f"夾合後     = {np.round(after_close, 4)}   "
          f"掉落 {(before[2] - after_close[2]) * 1000:+.1f} mm   "
          f"距 EE {d['ee_dist'] * 1000:.1f} mm")

    holding = d["ee_dist"] < cfg["success"]["max_ee_distance"]
    if not holding:
        print("\n✗ 夾合後方塊就掉了 —— 接觸/摩擦問題,不是力量問題")

    # ── 3. lift, using joint2 alone (no IK needed) ─────────────────────
    lifted_ok = False
    for _ in range(120):
        q[1] = max(q[1] - 0.004, -0.5)       # joint2 raises the arm
        scene.set_targets(q)
        scene.step()
        if scene.success():
            lifted_ok = True
            break

    d = scene.diagnose()
    cube = scene.cube_pose()[0]
    print(f"抬升後     = {np.round(cube, 4)}   "
          f"高度 {d['cube_z'] * 1000:.1f} mm (需 >{d['lift_needed'] * 1000:.1f})   "
          f"距 EE {d['ee_dist'] * 1000:.1f} mm (需 <{d['max_ee_dist'] * 1000:.1f})   "
          f"連續 {d['hold']}/{d['hold_needed']}")

    if args.png:
        obs = scene.observe(render=True)
        try:
            from PIL import Image
            args.png.parent.mkdir(parents=True, exist_ok=True)
            for name, img in obs["images"].items():
                out = args.png.with_name(f"{args.png.stem}_{name}{args.png.suffix}")
                Image.fromarray(img.astype("uint8")).save(out)
                print(f"存出 {out}  {img.shape}")
        except ImportError:
            print("要存 PNG 需要 pillow: ./python.sh -m pip install pillow")

    print("\n" + "=" * 62)
    if lifted_ok:
        print("✅ 夾得住也抬得起來 —— 把目前的參數當成基準寫進 task yaml")
    elif holding:
        print("△  夾住了但沒通過抬升判定。先看上面哪一項沒達標:")
        print("   高度不夠 → 抬升幅度太小;距 EE 太遠 → 抬升途中滑脫")
    else:
        print("✗ 夾不住。依序試(每次只改一項,改完重跑 build_scene.py):")
        print("   1. cube.mass 15 g → 8 g")
        print("   2. overrides.gripper_drive.stiffness 625 → 2000 → 5000")
        print("   3. overrides.finger_colliders.approximation → convexDecomposition")
        print("   4. overrides.pads.enabled → true")
    print("=" * 62)

    if args.livestream:
        print("\n串流中,Ctrl+C 結束")
        try:
            while True:
                scene.step(render=True)
        except KeyboardInterrupt:
            pass

    simulation_app.close()
    return 0 if lifted_ok else 1


if __name__ == "__main__":
    sys.exit(main())
