"""Closed-form kinematics for the OMX follower, specialised to top-down grasps.

Why closed form rather than a numerical solver such as Lula:

* The arm is ``joint1`` yaw + ``joint2/3/4`` **parallel** pitch + ``joint5`` roll.
  Fix the gripper pointing straight down and the three pitch joints must sum to
  a constant -- what is left is a planar 2R, which has an exact solution.
* It is 5-DOF. A numerical solver handed a 6-DOF pose target fails on most of
  them, because most orientations genuinely are unreachable. Asking only for
  what the arm can do removes the failure mode instead of reporting it.
* Lula also needs a hand-written c-space descriptor whose parameters are guesses.

The classic trap in hand-rolled OMX IK is the elbow offset: ``joint2 -> joint3``
is ``(0.0415, 0, 0.11315)``, **20.15 degrees off vertical**, and dropping it
biases every solution. That link is carried here as a vector with its own angle,
and ``--check`` runs FK on every solution so the trap cannot survive silently.

Geometry is read from the URDF rather than written down, so it cannot drift.

    bash sim/isaac_python.sh sim/ik.py --check 1000
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))
import task_config  # noqa: E402

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]


def _rot(axis, q):
    c, s = math.cos(q), math.sin(q)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class OMXKinematics:
    """Forward and inverse kinematics of the 5 arm joints.

    The "tool" is the point the fingers actually pinch: ``end_effector_link``
    shifted by ``gripper.grasp_offset``. Solving for the pinch rather than for
    the EE frame is what makes ``ik(cube_position)`` mean what it says.
    """

    def __init__(self, cfg: task_config.Config, position_only: bool = False):
        # position_only mimics MoveIt's `position_only_ik: True` -- the setting
        # ROBOTIS ships for this arm: damped least squares on position alone,
        # seeded from the current pose, orientation left to fall out. Provided so
        # the two can be compared on success rate rather than argued about.
        self.position_only = position_only
        self._seed = np.zeros(5)
        urdf = ET.parse(cfg.robot_urdf).getroot()
        joints = {j.get("name"): j for j in urdf.findall("joint")}

        def origin(name):
            o = joints[name].find("origin")
            xyz = np.array([float(v) for v in o.get("xyz").split()])
            rpy = o.get("rpy", "0 0 0")
            if any(abs(float(v)) > 1e-9 for v in rpy.split()):
                raise ValueError(
                    f"{name} 的 origin 有非零 rpy ({rpy});這份解析解假設全部為零"
                )
            return xyz

        def axis(name):
            a = [float(v) for v in joints[name].find("axis").get("xyz").split()]
            for letter, unit in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
                if np.allclose(a, unit):
                    return letter
            raise ValueError(f"{name} 的 axis {a} 不是座標軸,這份解析解不適用")

        self.origins = [origin(j) for j in ARM_JOINTS]
        self.axes = [axis(j) for j in ARM_JOINTS]
        if self.axes != ["z", "y", "y", "y", "x"]:
            raise ValueError(f"關節軸序列變成 {self.axes},解析解的假設不再成立")

        # Pinch point in link5's frame.
        self.ee_local = origin("end_effector_joint")
        self.tool_local = self.ee_local + np.array(cfg["gripper"]["grasp_offset"])

        # ── the constants the planar solve needs ────────────────────────
        # Yaw axis passes through origins[0] (the offset is applied before the
        # rotation, so it does not swing).
        self.yaw_xy = self.origins[0][:2].copy()
        self.shoulder_z = self.origins[0][2] + self.origins[1][2]
        if abs(self.origins[1][0]) > 1e-9 or abs(self.origins[1][1]) > 1e-9:
            raise ValueError("joint2 相對 joint1 有水平偏移,肩不在偏航軸上")

        # link2 -> link3, carried as (length, angle from +u) in the arm plane.
        du, dv = self.origins[2][0], self.origins[2][2]
        self.L1 = math.hypot(du, dv)
        self.alpha = math.atan2(dv, du)
        # link3 -> link4
        self.L2 = self.origins[3][0]
        # link4 -> link5, folded together with the tool's forward reach later
        self.L4x = self.origins[4][0]

        self.reach = self.L1 + self.L2 + self.L4x + np.linalg.norm(self.tool_local)

    # ── forward ────────────────────────────────────────────────────────
    def link5_frame(self, q):
        """``(origin, R)`` of link5 in world coordinates."""
        p = np.zeros(3)
        R = np.eye(3)
        for i, (off, ax) in enumerate(zip(self.origins, self.axes)):
            p = p + R @ off
            R = R @ _rot(ax, q[i])
        return p, R

    def fk(self, q):
        """Tool position and the link origins, in world coordinates."""
        p = np.zeros(3)
        R = np.eye(3)
        points = []
        for i, (off, ax) in enumerate(zip(self.origins, self.axes)):
            p = p + R @ off
            R = R @ _rot(ax, q[i])
            points.append(p.copy())
        return p + R @ self.tool_local, np.array(points)

    def in_ee(self, q, world_point):
        """``world_point`` in the **end_effector_link** frame.

        This is the frame ``gripper.grasp_offset`` is written in, so a measured
        value compares with the configured one directly. Measuring in link5's
        frame instead is off by the fixed 91.9 mm end-effector joint offset.

        ``end_effector_joint`` carries no rotation, so the two frames share an
        orientation and this is a translation only.
        """
        p, R = self.link5_frame(q)
        return R.T @ (np.asarray(world_point, dtype=float) - p) - self.ee_local

    def in_link5(self, q, world_point):
        p, R = self.link5_frame(q)
        return R.T @ (np.asarray(world_point, dtype=float) - p)

    # ── inverse ────────────────────────────────────────────────────────
    def _tool_in_link5(self, q5):
        """The pinch point in link5's frame after the roll.

        ``q5`` tilts the tool's lateral and vertical offsets into each other, so
        both the out-of-plane shift and the in-plane drop depend on it. Ignoring
        that costs up to 11 mm, which is most of a 25 mm cube.
        """
        return _rot("x", q5) @ self.tool_local

    def _jacobian(self, q, eps=1e-6):
        """3x5 position Jacobian of the tool point, by finite difference."""
        p0 = self.fk(q)[0]
        J = np.empty((3, 5))
        for i in range(5):
            dq = q.copy()
            dq[i] += eps
            J[:, i] = (self.fk(dq)[0] - p0) / eps
        return J

    def ik_position_only(self, target, seed=None, iters=200, damping=1e-3, tol=1e-5):
        """Damped least squares on position alone, seeded -- KDL's behaviour.

        Orientation is whatever falls out. The seed carries between calls, which
        is what a solver driving a trajectory actually does.
        """
        q = np.asarray(self._seed if seed is None else seed, dtype=float).copy()
        target = np.asarray(target, dtype=float)
        for _ in range(iters):
            err = target - self.fk(q)[0]
            if np.linalg.norm(err) < tol:
                self._seed = q.copy()
                return q
            J = self._jacobian(q)
            q = q + J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        return None

    def ik(self, target, cube_yaw=0.0, elbow="up", pitch=math.pi / 2, iters=4):
        """Joint angles putting the pinch point on ``target``, gripper pointing down.

        ``pitch`` is the gripper's downward tilt; pi/2 is straight down, which is
        the only case this task uses. ``cube_yaw`` orients the finger opening
        across the cube's faces.

        Returns ``None`` when the target is out of reach -- callers must check.
        """
        if self.position_only:
            return self.ik_position_only(target)
        target = np.asarray(target, dtype=float)
        rel = target - np.array([self.yaw_xy[0], self.yaw_xy[1], self.shoulder_z])
        R_h = math.hypot(rel[0], rel[1])
        theta = math.atan2(rel[1], rel[0])

        # q1 and q5 depend on each other through the tool's lateral offset: the
        # roll decides how far off the arm's plane the pinch sits, and the plane
        # is chosen by q1. Two or three passes converge to well under a micron.
        q1 = theta
        q5 = 0.0
        for _ in range(iters):
            # With the gripper pointing down, the finger opening axis is
            # horizontal at azimuth q1 + (pi/2 - q5). Line it up with a cube
            # face -- any multiple of 90 degrees does, so take the nearest.
            want = q1 + math.pi / 2 - cube_yaw
            q5 = want - round(want / (math.pi / 2)) * (math.pi / 2)
            t = self._tool_in_link5(q5)
            if abs(t[1]) > R_h:
                return None                     # target too close to the yaw axis
            u = math.sqrt(R_h * R_h - t[1] * t[1])
            q1 = theta - math.atan2(t[1], u)

        t = self._tool_in_link5(q5)
        u = math.sqrt(max(R_h * R_h - t[1] * t[1], 0.0))
        v = rel[2]

        # Everything from joint4 onward is one rigid piece rotated by the total
        # pitch: (L4x + tool_x) forward, tool_z down.
        cp, sp = math.cos(pitch), math.sin(pitch)
        fx, fz = self.L4x + t[0], t[2]
        wu = u - (fx * cp + fz * sp)
        wv = v - (-fx * sp + fz * cp)

        # Planar 2R from the shoulder to joint4.
        d2 = wu * wu + wv * wv
        cos_d = (d2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        if not -1.0 <= cos_d <= 1.0:
            return None                         # out of reach
        delta = math.acos(cos_d) * (1.0 if elbow == "up" else -1.0)

        psi = math.atan2(wv, wu)
        a = psi - math.atan2(self.L2 * math.sin(delta),
                             self.L1 + self.L2 * math.cos(delta))
        q2 = self.alpha - a
        q3 = -delta - self.alpha
        q4 = pitch - q2 - q3
        return np.array([q1, q2, q3, q4, q5])

    def hover(self, target, cube_yaw=0.0, prefer=0.12, floor=0.05, step=0.005):
        """The highest reachable point straight above ``target``, up to ``prefer``.

        A constant approach height cannot work: pointing the gripper down spends
        115 mm of vertical reach before the target counts, so the far edge of the
        workspace runs out of arm while the near edge has room to spare. Descend
        from the preferred height until the arm can actually get there.

        Returns ``(position, q)``, or ``None`` if even ``floor`` is unreachable.
        """
        target = np.asarray(target, dtype=float)
        h = prefer
        while h >= floor - 1e-9:
            at = np.array([target[0], target[1], target[2] + h])
            q = self.solve(at, cube_yaw)
            if q is not None:
                return at, q
            h -= step
        return None

    def solve(self, target, cube_yaw=0.0, min_z=0.01, **kw):
        """``ik`` plus elbow choice: prefer the pose that keeps the arm off the floor."""
        best = None
        for elbow in ("up", "down"):
            q = self.ik(target, cube_yaw, elbow=elbow, **kw)
            if q is None:
                continue
            _, pts = self.fk(q)
            clearance = pts[1:, 2].min()
            if clearance < min_z:
                continue
            if best is None or clearance > best[1]:
                best = (q, clearance)
        return None if best is None else best[0]


def check(cfg, n):
    """Round-trip every solution through FK -- the check that kills the offset trap."""
    kin = OMXKinematics(cfg)
    print(f"URDF: {cfg.robot_urdf}")
    print(f"  肩 (yaw 軸上) = ({kin.yaw_xy[0]:.5f}, {kin.yaw_xy[1]:.5f}, {kin.shoulder_z:.5f})")
    print(f"  L1 = {kin.L1:.5f} m,偏離垂直 {90 - math.degrees(kin.alpha):.2f}°   "
          f"L2 = {kin.L2:.5f}   最大伸展 ≈ {kin.reach:.3f} m")
    print(f"  夾持點 (link5 座標系) = {np.round(kin.tool_local, 5)}\n")

    s = cfg["spawn"]
    grasp_z = cfg["cube"]["size"] / 2
    lift_z = grasp_z + cfg["success"]["lift_height"]
    rng = np.random.default_rng(0)

    # The three poses an episode actually passes through. "hover" is adaptive --
    # testing it at a fixed height would only re-measure the reach limit.
    stages = [("抓取", lambda t, y: (np.array([t[0], t[1], grasp_z]),
                                     kin.solve([t[0], t[1], grasp_z], y))),
              ("抬升", lambda t, y: (np.array([t[0], t[1], lift_z]),
                                     kin.solve([t[0], t[1], lift_z], y))),
              ("懸停", lambda t, y: (kin.hover([t[0], t[1], grasp_z], y) or (None, None)))]
    rows = []
    hovers = []
    for name, fn in stages:
        ok, errs = 0, []
        for _ in range(n):
            r = rng.uniform(*s["radius"])
            th = math.radians(rng.uniform(*s["theta_deg"]))
            yaw = math.radians(rng.uniform(*s["yaw_deg"]))
            tgt = [r * math.cos(th), r * math.sin(th)]
            at, q = fn(tgt, yaw)
            if q is None:
                continue
            tool, _ = kin.fk(q)
            errs.append(np.linalg.norm(tool - at))
            if name == "懸停":
                hovers.append(at[2] - grasp_z)
            ok += 1
        errs = np.array(errs) if errs else np.array([np.nan])
        rows.append((name, ok / n, errs.max(), np.median(errs)))

    print(f"{'階段':>6s}  {'成功率':>7s}  {'FK 往返最大誤差':>16s}  {'中位數':>10s}")
    for name, rate, mx, md in rows:
        print(f"{name:>6s}  {rate * 100:6.1f}%  {mx * 1e6:14.2f} µm  {md * 1e6:8.2f} µm")
    if hovers:
        h = np.array(hovers) * 1000
        print(f"\n懸停高度(方塊上方):最低 {h.min():.0f} mm  中位數 {np.median(h):.0f} mm  最高 {h.max():.0f} mm")

    worst = max(r[2] for r in rows)
    rate = min(r[1] for r in rows)
    print()
    if worst > 1e-6:
        print(f"✗ FK 往返誤差 {worst * 1000:.3f} mm —— 解析解有錯,不要拿去錄資料")
        return 1
    if rate < 0.99:
        print(f"✗ 成功率只有 {rate * 100:.1f}% —— 生成範圍超出可達區,要縮 spawn")
        return 1
    print(f"✅ 成功率 {rate * 100:.1f}%,FK 往返誤差 < 1 µm —— 肘偏置處理正確")
    return 0


def reach_map(cfg):
    """How high can the gripper hover, pointing straight down, at each radius?

    Pointing the gripper down costs 115 mm of vertical reach before the target is
    even considered, so the far edge of the spawn annulus runs out of arm long
    before the near edge does. The expert's approach height has to follow this
    curve; a single constant height either clips the far targets or wastes travel
    on the near ones.
    """
    kin = OMXKinematics(cfg)
    s = cfg["spawn"]
    print("半徑 -> 夾爪垂直向下時可達的最高點(取 θ、yaw 最壞情況)\n")
    print(f"{'半徑':>8s}  {'最高懸停':>10s}  {'抓取高度可達':>12s}")
    lo, hi = s["radius"]
    worst_top = 1e9
    for r in np.linspace(lo, hi, 11):
        top = None
        for h in np.arange(0.30, 0.0, -0.005):
            if all(kin.solve([r * math.cos(math.radians(t)), r * math.sin(math.radians(t)), h],
                             math.radians(y)) is not None
                   for t in s["theta_deg"] + [0.0]
                   for y in (s["yaw_deg"][0], 0.0, s["yaw_deg"][1])):
                top = h
                break
        grasp_ok = kin.solve([r, 0.0, cfg["cube"]["size"] / 2], 0.0) is not None
        worst_top = min(worst_top, top if top is not None else 0.0)
        print(f"{r * 1000:7.0f}mm  {('%.0f mm' % (top * 1000)) if top else '  搆不到':>10s}"
              f"  {'可以' if grasp_ok else '搆不到':>12s}")
    print(f"\n整個環帶都能懸停的最高高度 = {worst_top * 1000:.0f} mm")
    print("→ 專家的接近高度要照這條曲線走,不能用單一常數。")
    return 0


def check_against_isaac(cfg, n):
    """Compare this file's FK against the simulator's own.

    The maths here comes from the **URDF**; the simulation runs the **USD**. They
    are supposed to agree, and "supposed to" is how a silent bias survives into
    200 recorded episodes. Worth re-running whenever either asset is regenerated.
    """
    sys.path.insert(0, str(SIM_DIR))
    import app

    simulation_app = app.start(headless=True)
    sys.path.insert(0, str(SIM_DIR))
    from scene import PickCubeScene

    kin = OMXKinematics(cfg)
    scene = PickCubeScene(cfg, with_cameras=False)
    scene.reset(seed=0, cube_pose=(np.array([1.0, 0.0, 0.5]), None))

    rng = np.random.default_rng(0)
    errs = []
    for _ in range(n):
        q = np.zeros(len(cfg.joints), dtype=np.float32)
        q[:5] = rng.uniform(-0.6, 0.6, 5)
        q[5] = cfg["gripper"]["open"]
        scene.set_targets(q)
        for _ in range(220):                    # let the drives actually arrive
            scene.step()
        measured = scene.joint_positions()[:5]
        errs.append(np.linalg.norm(kin.fk(measured)[0] - scene.grasp_point()))

    errs = np.array(errs)
    print(f"\n對照 Isaac 的 FK({n} 個隨機姿態,用**實測**關節角):")
    print(f"  夾持點差異  最大 {errs.max() * 1000:.3f} mm  中位數 {np.median(errs) * 1000:.3f} mm")
    ok = errs.max() < 1e-3
    print("  ✅ URDF 與 USD 的運動學一致" if ok else
          "  ✗ URDF 與 USD 不一致 —— IK 會系統性偏掉,先查 USD 是不是從這份 URDF 來的")
    simulation_app.close()
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", type=int, metavar="N", default=1000,
                    help="每個高度各取樣 N 個目標做 FK 往返驗證")
    ap.add_argument("--map", action="store_true",
                    help="印出各半徑可達的最高懸停高度")
    ap.add_argument("--isaac", type=int, nargs="?", const=20, metavar="N",
                    help="開 Isaac 對照 FK,確認 URDF 與 USD 的運動學一致")
    args = ap.parse_args()
    cfg = task_config.load()
    if args.map:
        return reach_map(cfg)
    if args.isaac:
        return check_against_isaac(cfg, args.isaac)
    return check(cfg, args.check)


if __name__ == "__main__":
    sys.exit(main())
