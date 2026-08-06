"""Run the scripted expert against Isaac, entirely over ROS.

    ros2 run omx_vla_app expert                       # 5 episodes
    ros2 run omx_vla_app expert --ros-args -p episodes:=20 -p holdout:=true
    ros2 run omx_vla_app record --ros-args -p episodes:=5   # 同上,外加寫出資料

Isaac is opened by hand and **Play must be running**. Nothing here starts a
simulator: the arm is driven through the same ``/sync/command`` seam ``jog`` uses,
the cube is read over ``tf2_msgs`` and moved over Isaac's prim service, and both
come from stock nodes in the task scene's ActionGraph.

The state machine and the kinematics are imported from ``vla/sim/`` unchanged --
there is exactly one expert.

On time alignment: ``state`` and ``action`` are taken from the same iteration of
the control loop, so ``action[t]`` is the command issued at ``state[t]``. Images
cannot be -- they arrive on their own topic at the renderer's pace -- so each
recorded frame carries the image's age instead of pretending it is simultaneous.

⚠️ **Still open:** what the converter should do with a stale image. On this
laptop the cameras publish at ~12 Hz against a 30 Hz control loop, so roughly
every second frame repeats pixels. Deciding that needs the recording rate the
data will actually be collected at.
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
from .recorder import Recorder

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
        self.place_retries = 0
        self._mimic = None

        self._cmd = self.create_publisher(JointState, "/sync/command", 10)

        ep = profile.endpoint("sim")
        self._ep = ep
        self._canonical_of = {ep.endpoint_joint_name(j): j for j in profile.joints}
        self.create_subscription(JointState, ep.state_topic, self._on_state, 10)
        self.create_subscription(TFMessage, ros["cube_tf_topic"], self._on_tf, 100)

        # Latest frame only, written on request. This is for looking at a
        # specific moment, not a recorder.
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
        # gripper_joint_2 is the **passive** finger: no drive, carried by a
        # PhysxMimicJoint at multiplier -1. It is not in the canonical six, so
        # read it straight off the wire. If it does not mirror joint_1 the
        # passive finger is sitting wherever the constraint put it rather than
        # against the cube.
        for name, pos in zip(msg.name, msg.position):
            if name == "gripper_joint_2":
                self._mimic = float(pos)

    def _on_tf(self, msg):
        for t in msg.transforms:
            self._frames_seen.add(t.child_frame_id)
            if t.child_frame_id != self.cube_frame:
                continue
            p, r = t.transform.translation, t.transform.rotation
            self._cube = (np.array([p.x, p.y, p.z]),
                          2.0 * math.atan2(float(r.z), float(r.w)))

    def _on_image(self, msg, name):
        # Arrival time, not msg.header.stamp: Isaac stamps images with
        # **simulation** time by default, which is a different clock from the
        # one the control loop runs on. Subtracting the two gives nonsense.
        self._frames[name] = (msg, time.time())

    def frames(self):
        """Latest ``(msg, arrived_at)`` per camera; None if none has arrived."""
        return {n: self._frames.get(n) for n in self.cfg["ros"]["camera_topics"]}

    def save_frames(self, tag, out_dir="/vla/data/diag"):
        """Write whatever each camera last sent. Returns the paths written."""
        import os

        from PIL import Image as PILImage

        os.makedirs(out_dir, exist_ok=True)
        written = []
        for name, (msg, _) in list(self._frames.items()):
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

    def mimic_error(self):
        """How far the passive finger is from mirroring the driven one, in rad."""
        if self._mimic is None or self._q is None:
            return float("nan")
        gi = self.profile.joints.index(self.cfg["gripper"]["joint"])
        return float(self._mimic + self._q[gi])      # multiplier -1 -> sum is 0

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

        On the measurement rather than a fixed sleep: how long the drives take
        depends on where the last episode left the arm, and starting the next
        episode mid-motion silently truncates it.
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

        The previous episode ends holding it, and ``go_home`` returns as soon as
        the joints are in tolerance -- while the cube may still be between the
        fingers. Teleporting it then puts it inside the finger geometry and
        PhysX pushes it straight back out.
        """
        floor = self.cfg["cube"]["size"] / 2
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            here = self.cube()
            if here is not None and here[0][2] < floor + 0.01:
                return True
            time.sleep(0.05)
        return False

    def _write_pose(self, position, yaw):
        """Write the cube's pose. Returns False if Isaac refused the write."""
        try:
            self.prim.set(self.cube_prim, "xformOp:translate",
                          [float(v) for v in position])
        except IsaacPrimError as exc:
            print(f"      ! 寫入方塊位置失敗:{exc}")
            return False
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
        return True

    def _wait_placed(self, position, settle, still_tol):
        """Did the cube arrive and stop moving?

        Checked through TF rather than trusted: a write that quietly does nothing
        leaves the cube where the last episode left it, which reads as a policy
        failure rather than a plumbing failure. Stillness rather than zeroed
        velocity, because TF carries no velocity and "has stopped" is the
        property actually wanted.
        """
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

    def place_cube(self, position, yaw, settle=2.0, still_tol=1e-4, attempts=3):
        """Teleport the cube where the episode wants it, retrying if it does not go.

        ⚠️ The write occasionally does not take -- about 1 episode in 20, root
        cause unconfirmed, most likely a race between the service applying the
        USD edit and PhysX owning the body. Retrying means **re-issuing the
        write**, not waiting longer: the failure is the write not landing.
        """
        for attempt in range(1, attempts + 1):
            if self._write_pose(position, yaw) and self._wait_placed(
                    position, settle, still_tol):
                self.place_retries += attempt - 1
                if attempt > 1:
                    print(f"      · 方塊第 {attempt} 次嘗試才放到位")
                return True
        self.place_retries += attempts - 1
        return False


def run_episode(client, expert, kin, cfg, pos, yaw, snap=None, rec=None):
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
        return False, FAILED, 0, None, float('nan'), (float('nan'),) * 3

    at, at_yaw = client.cube()
    if not expert.reset(client.joints(), at, at_yaw):
        print(f"      ✗ 規劃不了:{expert.reject_reason}"
              f"   (要求放在 {np.round(pos, 4)})")
        return False, FAILED, 0, None, float('nan'), (float('nan'),) * 3

    ok = False
    held = []
    mimic = []
    shot = set()
    lag = float("nan")
    grasp_pitch = grasp_tilt = float("nan")
    while True:
        phase_before = expert.phase
        q_now = client.joints()
        # Resync a stateful solver to the real arm. The analytic one has no
        # state and ignores this; MoveIt's position-only IK is decided almost
        # entirely by its seed, and left to feed off its own output it drifts.
        if hasattr(kin, "seed"):
            kin.seed = q_now[:5]
        measured = kin.fk(q_now[:5])[0]
        targets, finished = expert.act(measured)
        if phase_before != expert.phase and expert.phase == CLOSE:
            # How far behind the command the arm is when the fingers start to
            # close. Should stay a few mm; it was ~14 before the settle gate.
            lag = float(np.linalg.norm(
                expert.tool - kin.fk(client.joints()[:5])[0]))
            R = kin.link5_frame(client.joints()[:5])[1]
            grasp_pitch = math.degrees(math.asin(-(R @ np.array([1., 0, 0]))[2]))
            grasp_tilt = math.degrees(math.asin(abs((R @ np.array([0, 1., 0]))[2])))
        if snap and phase_before != expert.phase and expert.phase not in shot:
            # One frame per phase transition. The two that matter are entering
            # CLOSE (are the fingers around the cube?) and entering LIFT (did it
            # actually get picked up?).
            shot.add(expert.phase)
            client.save_frames(f"{snap}_p{expert.phase}")
        if finished:
            break

        # The recorded pair comes from **this** iteration: `state` was measured
        # to produce `targets`, so action[t] really is the command issued at
        # state[t]. Re-subscribing to the two topics instead would pair them by
        # arrival time and teach the policy the transport jitter.
        state = client.joints()
        client.send(targets)
        if rec is not None:
            rec.frame(state, targets, client.frames(), time.time())
        time.sleep(period)
        if expert.phase in (LIFT, HOLD):
            cube = client.cube()[0]
            q = client.joints()
            # Height plus distance to the pinch point. Deliberately not
            # "do the fingers look like they touch": the decomposed colliders
            # are fatter than the rendered mesh and are not drawn, so a firm
            # grasp shows a visible gap on both sides.
            tool = kin.fk(q[:5])[0]
            if (cube[2] > grasp_z + succ["lift_height"]
                    and np.linalg.norm(cube - tool) < succ["max_ee_distance"]):
                ok = True
                # Where the cube ended up, in the gripper's own frame -- see the
                # printout at the end of a run for what this can and cannot say.
                held.append(kin.in_ee(q[:5], cube))
                mimic.append(client.mimic_error())
    residual = np.mean(held, axis=0) if held else None
    mimic_err = float(np.nanmean(mimic)) if mimic else float('nan')
    return ok, expert.phase, expert.ticks, residual, lag,\
        (grasp_pitch, grasp_tilt, mimic_err)


def main(argv=None, record=False):
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
    client = ExpertClient(load_profile(profile_path), cfg)

    client.declare_parameter("episodes", 5)
    client.declare_parameter("holdout", False)
    client.declare_parameter("seed", 0)
    client.declare_parameter("save_frames", False)
    client.declare_parameter("theta_deg", 999.0)   # 999 = 照常隨機取樣
    client.declare_parameter("radius", 0.0)
    client.declare_parameter("out", "/vla/data/raw")
    # Three solvers, same interface, chosen at run time so they can be compared
    # by success rate rather than argued about:
    #   moveit         MoveIt's KDL plugin with ROBOTIS's own omx_f config
    #   analytic       sim/ik.py -- closed form, gripper held pointing down
    #   position_only  a damped-least-squares stand-in for MoveIt, no install
    client.declare_parameter("ik", "moveit")
    which = client.get_parameter("ik").value
    analytic = OMXKinematics(cfg, position_only=(which == "position_only"))
    if which == "moveit":
        from .moveit_ik import MoveItKinematics
        kin = MoveItKinematics(cfg, OMXKinematics(cfg))
    elif which in ("analytic", "position_only"):
        kin = analytic
    else:
        print(f"[expert] 不認得的 ik:={which}(要 moveit / analytic / position_only)")
        raise SystemExit(2)
    print(f"[expert] IK = {which}")
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

        rec = Recorder(cfg, client.get_parameter("out").value) if record else None
        expert = PickCubeExpert(cfg, kin, seed=seed)
        rng = np.random.default_rng(seed)
        wins, fails, residuals, lags, geoms = 0, {}, [], [], []
        for i in range(episodes):
            pos, yaw, r, th = sample_cube_pose(cfg, rng, holdout)
            if fixed_theta < 900.0:            # 重現某個特定位置
                th = fixed_theta
                r = fixed_radius or r
                a = math.radians(th)
                pos = np.array([r * math.cos(a), r * math.sin(a),
                                cfg["cube"]["size"] / 2])
            snap = f"ep{i + 1:02d}_r{r * 1000:.0f}_t{th:+.0f}" if save_frames else None
            if rec is not None:
                rec.begin(i + 1, {"seed": seed, "episode": i + 1,
                                  "holdout": bool(holdout),
                                  "cube": {"r": float(r), "theta_deg": float(th),
                                           "yaw_rad": float(yaw),
                                           "requested": [float(v) for v in pos]}})
            ok, phase, ticks, residual, lag, geom = run_episode(
                client, expert, kin, cfg, pos, yaw, snap, rec)
            geoms.append(geom)
            if rec is not None:
                rec.end(ok)
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
            got = np.mean(residuals, axis=0) * 1000
            want = np.array(cfg["gripper"]["grasp_offset"]) * 1000
            n = len(residuals)
            se = np.std(residuals, axis=0) * 1000 / max(n ** 0.5, 1)
            print(f"\n夾持殘差(方塊在 end_effector_link 座標系,mm):"
                  f" {np.round(got, 2)} ±{np.round(se, 2)}"
                  f"   設定 {np.round(want, 2)}   (n={n})")
        G = np.array([g for g in geoms if g[0] == g[0]])
        if len(G):
            print(f"夾合瞬間夾爪俯仰角: 平均 {G[:,0].mean():.1f}°  範圍 "
                  f"{G[:,0].min():.1f}~{G[:,0].max():.1f}°   "
                  f"開合軸偏離水平 最大 {G[:,1].max():.1f}°")
            m = G[:, 2][~np.isnan(G[:, 2])]
            if len(m):
                print(f"被動指(gripper_joint_2)未跟上驅動指: 平均 {np.degrees(m).mean():+.2f}°"
                      f"  最大 {np.degrees(np.abs(m)).max():.2f}°")
        if lags:
            L = np.array(lags) * 1000
            print(f"夾合瞬間手臂落後命令(mm):平均 {L.mean():.1f}  最大 {L.max():.1f}")
        if client.place_retries:
            print(f"方塊放置重試 {client.place_retries} 次 / {episodes} 集")

        if rec is not None:
            print(rec.summary())

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


def record_main(argv=None):
    """Same run, plus a raw dump of every successful episode."""
    return main(argv, record=True)


if __name__ == "__main__":
    sys.exit(main())
