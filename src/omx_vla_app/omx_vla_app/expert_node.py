"""Run the scripted expert against Isaac, entirely over ROS.

    ros2 run omx_vla_app expert                       # 5 episodes
    ros2 run omx_vla_app expert --ros-args -p episodes:=20 -p holdout:=true

Isaac is opened by hand and **Play must be running**. Nothing here starts a
simulator and nothing here touches Isaac's Python: the arm is driven through the
same ``/sync/command`` seam ``jog`` uses, and the cube is read and moved over
``tf2_msgs`` -- two stock nodes in the task scene's ActionGraph, no custom
interfaces and no Isaac-side scripting.

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
from sensor_msgs.msg import JointState
from sim_real_bridge.profile import ProfileError, load_profile
from sim_real_bridge.sync_node import SyncNode
from tf2_msgs.msg import TFMessage

from omx_bridge_app import ros_args

VLA_ROOT = os.environ.get("VLA_ROOT", "/vla")
if f"{VLA_ROOT}/sim" not in sys.path:
    sys.path.insert(0, f"{VLA_ROOT}/sim")

import task_config                                        # noqa: E402
from expert import FAILED, HOLD, LIFT, PHASE_NAMES, PickCubeExpert  # noqa: E402
from ik import OMXKinematics                              # noqa: E402


class ExpertClient(Node):
    """Reads the twin's joints and the cube's pose; writes commands and cube poses."""

    def __init__(self, profile, cfg):
        super().__init__("vla_expert")
        self.profile = profile
        self.cfg = cfg
        ros = cfg["ros"]
        self.cube_frame = ros["cube_frame"]

        self._q = None
        self._cube = None                      # (position, yaw)
        self._frames_seen = set()

        self._cmd = self.create_publisher(JointState, "/sync/command", 10)
        self._set_tf = self.create_publisher(TFMessage, ros["cube_set_topic"], 10)

        ep = profile.endpoint("sim")
        self._ep = ep
        self._canonical_of = {ep.endpoint_joint_name(j): j for j in profile.joints}
        self.create_subscription(JointState, ep.state_topic, self._on_state, 10)
        self.create_subscription(TFMessage, ros["cube_tf_topic"], self._on_tf, 100)

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

    def place_cube(self, position, yaw, settle=2.0, still_tol=1e-4):
        """Teleport the cube, then wait until it has stopped moving.

        Returns False if it never arrived -- which is the honest outcome when
        PhysX declines the write, and much easier to read than an episode that
        looks like the policy missed.

        Waiting for stillness instead of zeroing the velocity: TF carries no
        velocity, and "has stopped" is the property actually wanted anyway.
        """
        from geometry_msgs.msg import TransformStamped

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = self.cube_frame
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        t.transform.rotation.w = math.cos(yaw / 2)
        t.transform.rotation.z = math.sin(yaw / 2)

        deadline = time.monotonic() + settle
        arrived = False
        last = None
        while time.monotonic() < deadline:
            if not arrived:
                self._set_tf.publish(TFMessage(transforms=[t]))
            time.sleep(0.1)
            here = self.cube()
            if here is None:
                continue
            if not arrived and np.linalg.norm(here[0][:2] - np.asarray(position)[:2]) < 5e-3:
                arrived = True
                last = None
                continue
            if arrived:
                if last is not None and np.linalg.norm(here[0] - last) < still_tol:
                    return True
                last = here[0]
        return arrived


def sample_cube_pose(cfg, rng, holdout=False):
    """Same distribution as sim/scene.py, without needing the simulator."""
    s = cfg["spawn"]
    band = s["holdout_theta_deg"] if holdout else s["theta_deg"]
    lo, hi = s["holdout_theta_deg"]
    while True:
        th = rng.uniform(*band)
        if holdout or not (lo <= th <= hi):
            break
    r = rng.uniform(*s["radius"])
    yaw = math.radians(rng.uniform(*s["yaw_deg"]))
    a = math.radians(th)
    return (np.array([r * math.cos(a), r * math.sin(a), cfg["cube"]["size"] / 2]),
            yaw, r, th)


def run_episode(client, expert, kin, cfg, pos, yaw):
    """Reset, run the state machine, return (success, phase, ticks)."""
    grip = cfg["gripper"]
    succ = cfg["success"]
    grasp_z = cfg["cube"]["size"] / 2
    period = 1.0 / cfg["timing"]["fps"]

    client.go_home(grip["open"])
    if not client.place_cube(pos, yaw):
        print("      ✗ 方塊沒有移到指定位置 —— Isaac 端的 tf_set 訂閱沒生效")
        return False, FAILED, 0

    at, at_yaw = client.cube()
    if not expert.reset(client.joints(), at, at_yaw):
        return False, FAILED, 0

    ok = False
    while True:
        targets, finished = expert.act()
        if finished:
            break
        client.send(targets)
        time.sleep(period)
        if expert.phase in (LIFT, HOLD):
            cube = client.cube()[0]
            # Distance to the **pinch point**, from the measured joints -- it is
            # the point the cube is actually held at, and it costs no round trip.
            tool = kin.fk(client.joints()[:5])[0]
            if (cube[2] > grasp_z + succ["lift_height"]
                    and np.linalg.norm(cube - tool) < succ["max_ee_distance"]):
                ok = True
    return ok, expert.phase, expert.ticks


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
    episodes = client.get_parameter("episodes").value
    holdout = client.get_parameter("holdout").value
    seed = client.get_parameter("seed").value

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
        if not (client.wait_for_joints() and client.wait_for_cube()):
            raise SystemExit(1)

        expert = PickCubeExpert(cfg, kin, seed=seed)
        rng = np.random.default_rng(seed)
        wins, fails = 0, {}
        for i in range(episodes):
            pos, yaw, r, th = sample_cube_pose(cfg, rng, holdout)
            ok, phase, ticks = run_episode(client, expert, kin, cfg, pos, yaw)
            wins += ok
            if not ok:
                fails[PHASE_NAMES[phase]] = fails.get(PHASE_NAMES[phase], 0) + 1
            print(f"  第 {i + 1:3d} 集  r={r * 1000:.0f}mm θ={th:+.0f}° "
                  f"{'✅' if ok else '✗ '} 停在「{PHASE_NAMES[phase]}」 {ticks} 個週期",
                  flush=True)

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
