"""Run the scripted expert against Isaac, entirely over ROS.

    ros2 run omx_vla_app expert                       # 5 episodes
    ros2 run omx_vla_app expert --ros-args -p episodes:=20 -p holdout:=true

Isaac is opened by hand and **Play must be running**. Nothing here starts a
simulator and nothing here touches Isaac's Python: the arm is driven through the
same ``/sync/command`` seam ``jog`` uses, and the cube is read over ``tf2_msgs`` and
moved over Isaac's own prim service -- two stock nodes in the task scene's
ActionGraph, no Isaac-side scripting.

The state machine and the kinematics are imported from ``vla/sim/`` unchanged.
``expert.py`` and ``ik.py`` never imported ``isaacsim`` -- only numpy -- so the
same code that ran headless runs here. There is exactly one expert.

What this arrangement costs, stated plainly because it decides how M4 is written:
the control loop is now wall-clock rather than a fixed number of physics steps,
and states, actions and (later) images travel on separate topics with separate
timing. Recording by re-subscribing to those topics would teach a policy the DDS
jitter. Record ``state`` and ``action`` from **this node's own tick** -- both are
already in hand in the same iteration.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from sim_real_bridge.profile import ProfileError, load_profile
from sim_real_bridge.sync_node import SyncNode
from tf2_msgs.msg import TFMessage

from omx_bridge_app import ros_args

from .isaac_prim import IsaacPrim, IsaacPrimError

VLA_ROOT = os.environ.get("VLA_ROOT", "/vla")
if f"{VLA_ROOT}/sim" not in sys.path:
    sys.path.insert(0, f"{VLA_ROOT}/sim")

import task_config                                        # noqa: E402
from expert import CLOSE, FAILED, HOLD, LIFT, PHASE_NAMES, PickCubeExpert  # noqa: E402
from ik import OMXKinematics                              # noqa: E402
from spawn import sample_cube_pose                        # noqa: E402


class ExpertClient(Node):
    """Reads the twin's joints and the cube's pose; writes commands and cube poses."""

    def __init__(self, profile, cfg):
        super().__init__("vla_expert")
        self.profile = profile
        self.cfg = cfg
        ros = cfg["ros"]
        self.cube_frame = ros["cube_frame"]
        self.cube_prim = f"{cfg.task_root}/cube"
        self.prim = IsaacPrim(self, cfg)

        self._q = None
        self._cube = None                      # (position, yaw)
        self._frames_seen = set()
        self._warned_yaw = False

        self._cmd = self.create_publisher(JointState, "/sync/command", 10)

        ep = profile.endpoint("sim")
        self._ep = ep
        self._canonical_of = {ep.endpoint_joint_name(j): j for j in profile.joints}
        self.create_subscription(JointState, ep.state_topic, self._on_state, 10)
        self.create_subscription(TFMessage, ros["cube_tf_topic"], self._on_tf, 100)

        # Both cameras, kept as the latest frame only. Nothing is written unless
        # somebody asks -- this is for looking at a specific moment, not a
        # recorder; M4's recorder must take its frames from the control tick.
        self._frames = {}
        for name, topic in ros["camera_topics"].items():
            self.create_subscription(
                Image, topic, lambda m, n=name: self._on_image(m, n), 2)

    # ── incoming ───────────────────────────────────────────────────────
    def _on_state(self, msg):
        got = {}
        for name, pos in zip(msg.name, msg.position):
            joint = self._canonical_of.get(name)
            if joint is not None:
                got[joint] = self._ep.endpoint_to_canonical(joint, pos)
        if all(j in got for j in self.profile.joints):
            self._q = np.array([got[j] for j in self.profile.joints], dtype=float)

    def _on_tf(self, msg):
        for t in msg.transforms:
            self._frames_seen.add(t.child_frame_id)
            if t.child_frame_id != self.cube_frame:
                continue
            p, r = t.transform.translation, t.transform.rotation
            self._cube = (np.array([p.x, p.y, p.z]),
                          2.0 * math.atan2(float(r.z), float(r.w)))

    def _on_image(self, msg, name):
        self._frames[name] = msg

    def save_frames(self, tag, out_dir="/vla/data/diag"):
        """Write whatever each camera last sent. Returns the paths written."""
        import os

        from PIL import Image as PILImage

        os.makedirs(out_dir, exist_ok=True)
        written = []
        for name, msg in list(self._frames.items()):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            try:
                arr = arr.reshape(msg.height, msg.width, -1)[:, :, :3]
            except ValueError:
                continue                       # partial frame; skip rather than crash
            path = f"{out_dir}/{tag}_{name}.png"
            PILImage.fromarray(arr).save(path)
            written.append(path)
        return written

    # ── waiting ────────────────────────────────────────────────────────
    def _wait(self, get, timeout, what, hint):
        deadline = time.monotonic() + timeout
        while get() is None:
            if time.monotonic() > deadline:
                print(f"[expert] 等不到{what}。{hint}")
                return False
            time.sleep(0.02)
        return True

    def wait_for_joints(self, timeout=15.0):
        return self._wait(
            lambda: self._q, timeout,
            f"「{self._ep.state_topic}」上的關節狀態",
            "Isaac 開了嗎?Play 按了嗎?ROS_DOMAIN_ID 跟 Isaac 一致嗎?")

    def wait_for_cube(self, timeout=15.0):
        ok = self._wait(
            lambda: self._cube, timeout,
            f"「{self.cfg['ros']['cube_tf_topic']}」上的 {self.cube_frame} frame",
            "場景要是 vla/assets/pick_cube.usd(只有它帶方塊的 TF 節點)。")
        if not ok and self._frames_seen:
            print(f"          該 topic 上看到的 frame:{sorted(self._frames_seen)}")
        return ok

    # ── outgoing ───────────────────────────────────────────────────────
    def joints(self):
        return None if self._q is None else self._q.copy()

    def cube(self):
        return None if self._cube is None else (self._cube[0].copy(), self._cube[1])

    def send(self, canonical):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.profile.joints)
        msg.position = [float(v) for v in canonical]
        self._cmd.publish(msg)

    def go_home(self, gripper_open, settle=3.0, tol=0.02):
        """Command home and wait until the twin is actually there.

        Waiting on the measurement rather than sleeping a fixed time: how long
        the drives take depends on where the last episode left the arm, and
        starting the next episode mid-motion silently truncates it.
        """
        home = np.zeros(len(self.profile.joints))
        home[self.profile.joints.index(self.cfg["gripper"]["joint"])] = gripper_open
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            self.send(home)
            q = self.joints()
            if q is not None and np.abs(q - home).max() < tol:
                return True
            time.sleep(0.05)
        return False

    def wait_for_release(self, timeout=2.0):
        """Wait until the cube is back on the floor before moving it.

        The previous episode ends **holding** the cube. go_home opens the
        gripper, but it returns as soon as the joints are in tolerance -- the
        cube is still falling, and may still be between the fingers. Teleporting
        it then puts it inside the finger geometry, PhysX pushes it straight back
        out, and the reset looks like a failed write.
        """
        floor = self.cfg["cube"]["size"] / 2
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            here = self.cube()
            if here is not None and here[0][2] < floor + 0.01:
                return True
            time.sleep(0.05)
        return False

    def place_cube(self, position, yaw, settle=2.0, still_tol=1e-4):
        """Teleport the cube, then wait until it has stopped moving.

        Verified through TF rather than trusted: a write that quietly does
        nothing leaves the cube where the last episode left it, which reads as a
        policy failure rather than a plumbing failure.

        Waiting for stillness instead of zeroing the velocity: momentum survives
        a teleport, TF carries no velocity, and "has stopped" is the property
        actually wanted anyway.
        """
        self.prim.set(self.cube_prim, "xformOp:translate",
                      [float(v) for v in position])
        try:
            self.prim.set(self.cube_prim, "xformOp:orient",
                          [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        except IsaacPrimError as exc:
            # Position is what the episode needs; yaw only varies how the cube is
            # presented. Losing it silently would quietly narrow the dataset, so
            # say so, once, and carry on.
            if not self._warned_yaw:
                print(f"      ! 方塊 yaw 設不了({exc}) —— 位置有效,朝向固定")
                self._warned_yaw = True

        deadline = time.monotonic() + settle
        arrived = False
        last = None
        while time.monotonic() < deadline:
            time.sleep(0.1)
            here = self.cube()
            if here is None:
                continue
            if not arrived:
                if np.linalg.norm(here[0][:2] - np.asarray(position)[:2]) < 5e-3:
                    arrived, last = True, None
                continue
            if last is not None and np.linalg.norm(here[0] - last) < still_tol:
                return True
            last = here[0]
        return arrived


def run_episode(client, expert, kin, cfg, pos, yaw, snap=None):
    """Reset, run the state machine, return (success, phase, ticks)."""
    grip = cfg["gripper"]
    succ = cfg["success"]
    grasp_z = cfg["cube"]["size"] / 2
    period = 1.0 / cfg["timing"]["fps"]

    client.go_home(grip["open"])
    client.wait_for_release()
    if not client.place_cube(pos, yaw):
        here = client.cube()
        print(f"      ✗ 方塊沒有到位:目標 {np.round(pos, 3)},"
              f" 實際 {np.round(here[0], 3) if here else '讀不到'}")
        return False, FAILED, 0, None, float('nan')

    at, at_yaw = client.cube()
    if not expert.reset(client.joints(), at, at_yaw):
        print(f"      ✗ 規劃不了:{expert.reject_reason}"
              f"   (要求放在 {np.round(pos, 4)})")
        return False, FAILED, 0, None, float('nan')

    ok = False
    held = []
    shot = set()
    lag = float("nan")
    while True:
        phase_before = expert.phase
        measured = kin.fk(client.joints()[:5])[0]
        targets, finished = expert.act(measured)
        if phase_before != expert.phase and expert.phase == CLOSE:
            # How far behind the command is the arm when the fingers start to
            # close? The state machine declares arrival on the **commanded**
            # tool position; the drives are still catching up.
            lag = float(np.linalg.norm(
                expert.tool - kin.fk(client.joints()[:5])[0]))
        if snap and phase_before != expert.phase and expert.phase not in shot:
            # One frame per phase transition: the interesting moments are
            # entering CLOSE (are the fingers around the cube?) and entering
            # LIFT (did it actually get picked up?).
            shot.add(expert.phase)
            client.save_frames(f"{snap}_p{expert.phase}")
        if finished:
            break
        client.send(targets)
        time.sleep(period)
        if expert.phase in (LIFT, HOLD):
            cube = client.cube()[0]
            q = client.joints()
            # Distance to the **pinch point**, from the measured joints -- it is
            # the point the cube is actually held at, and it costs no round trip.
            tool = kin.fk(q[:5])[0]
            if (cube[2] > grasp_z + succ["lift_height"]
                    and np.linalg.norm(cube - tool) < succ["max_ee_distance"]):
                ok = True
                # Where the cube actually ended up, in the gripper's own frame.
                # gripper.grasp_offset says where the fingers pinch; if the two
                # disagree the arm is aiming a few mm off the cube's centre every
                # single time, which is a bias baked into every demonstration.
                held.append(kin.in_ee(q[:5], cube))
    residual = np.mean(held, axis=0) if held else None
    return ok, expert.phase, expert.ticks, residual, lag


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    rclpy.init(args=ros_args(argv, "profile", extra=["-p", "mode:=command",
                                                     "-p", "targets:=sim"]))
    try:
        engine = SyncNode()
    except (ValueError, ProfileError, FileNotFoundError) as exc:
        print(f"[expert] 設定錯誤: {exc}")
        rclpy.shutdown()
        return 1

    profile_path = engine.get_parameter("profile").get_parameter_value().string_value
    cfg = task_config.load()
    kin = OMXKinematics(cfg)
    client = ExpertClient(load_profile(profile_path), cfg)

    client.declare_parameter("episodes", 5)
    client.declare_parameter("holdout", False)
    client.declare_parameter("seed", 0)
    client.declare_parameter("save_frames", False)
    client.declare_parameter("theta_deg", 999.0)   # 999 = 照常隨機取樣
    client.declare_parameter("radius", 0.0)
    episodes = client.get_parameter("episodes").value
    holdout = client.get_parameter("holdout").value
    seed = client.get_parameter("seed").value
    save_frames = client.get_parameter("save_frames").value
    fixed_theta = client.get_parameter("theta_deg").value
    fixed_radius = client.get_parameter("radius").value

    executor = SingleThreadedExecutor()
    executor.add_node(engine)
    executor.add_node(client)

    def spin_quietly():
        try:
            executor.spin()
        except (ExternalShutdownException, rclpy.executors.ShutdownException):
            pass

    spin = threading.Thread(target=spin_quietly, daemon=True)
    spin.start()

    rc = 1
    try:
        if not (client.wait_for_joints() and client.wait_for_cube()
                and client.prim.wait_for_isaac()):
            raise SystemExit(1)

        expert = PickCubeExpert(cfg, kin, seed=seed)
        rng = np.random.default_rng(seed)
        wins, fails, residuals, lags = 0, {}, [], []
        for i in range(episodes):
            pos, yaw, r, th = sample_cube_pose(cfg, rng, holdout)
            if fixed_theta < 900.0:            # 重現某個特定位置
                th = fixed_theta
                r = fixed_radius or r
                a = math.radians(th)
                pos = np.array([r * math.cos(a), r * math.sin(a),
                                cfg["cube"]["size"] / 2])
            snap = f"ep{i + 1:02d}_r{r * 1000:.0f}_t{th:+.0f}" if save_frames else None
            ok, phase, ticks, residual, lag = run_episode(
                client, expert, kin, cfg, pos, yaw, snap)
            if lag == lag:
                lags.append(lag)
            wins += ok
            if residual is not None:
                residuals.append(residual)
            if not ok:
                fails[PHASE_NAMES[phase]] = fails.get(PHASE_NAMES[phase], 0) + 1
            print(f"  第 {i + 1:3d} 集  r={r * 1000:.0f}mm θ={th:+.0f}° "
                  f"{'✅' if ok else '✗ '} 停在「{PHASE_NAMES[phase]}」 {ticks} 個週期"
                  f"   夾合時落後 {lag * 1000:.1f}mm",
                  flush=True)

        if residuals:
            got = np.mean(residuals, axis=0)
            want = np.array(cfg["gripper"]["grasp_offset"])
            n = len(residuals)
            se = np.std(residuals, axis=0) / max(n ** 0.5, 1)
            print(f"\n夾持殘差:方塊被夾住時,實際位置在 end_effector_link 座標系")
            print(f"  實測平均 {np.round(got, 5)}   設定的 grasp_offset {np.round(want, 5)}")
            print(f"  差異     {np.round((got - want) * 1000, 2)} mm"
                  f"   ±{np.round(se * 1000, 2)} 標準誤 (n={n})")
            print("  ⚠️ 專家有注入 ±5mm 路徑點雜訊,單集殘差本來就會散;"
                  "看的是平均與標準誤,不是單集。")
            if np.abs(got - want).max() > 0.002:
                print(f"  → 把 gripper.grasp_offset 改成 [{got[0]:.4f}, {got[1]:.4f}, {got[2]:.4f}]")

        if lags:
            L = np.array(lags) * 1000
            print(f"\n夾合瞬間手臂落後命令: 平均 {L.mean():.1f}mm  最大 {L.max():.1f}mm"
                  f"  (n={len(L)})")
            print("  狀態機是用**命令**位置判定到達的,手臂還在追。夾合的 0.5 秒"
                  "通常夠它追上 —— 落後夠大時就先在方塊上緣合起來。")

        rate = wins / max(episodes, 1)
        print(f"\n成功 {wins}/{episodes} = {rate * 100:.0f}%"
              + (f"   失敗分佈 {fails}" if fails else ""))
        rc = 0 if rate >= 0.9 else 1
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        executor.shutdown()
        engine.destroy_node()
        client.destroy_node()
        spin.join(timeout=2)
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
