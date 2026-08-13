"""The scripted expert: a waypoint state machine, and the ROS node that runs it.

    ros2 run data_collection expert                       # 5 episodes, nothing written
    ros2 run data_collection expert --ros-args -p episodes:=20 -p holdout:=true
    ros2 run data_collection record --ros-args -p episodes:=500   # 同上,外加寫出 data/raw

Isaac is opened by hand and **Play must be running**. Nothing here starts a
simulator: the arm is driven through the same ``/sync/command`` seam ``jog`` uses,
the cube is read over ``tf2_msgs`` and moved over Isaac's prim service, and both
come from stock nodes in the task scene's ActionGraph. That is the whole reason
this is a plain ROS node rather than an Isaac script -- it does not care whether
the arm on the other end is simulated, and M7 is a parameter change.

``expert`` and ``record`` are the same program; ``record`` only attaches a
``Recorder``. Keeping them one file is deliberate -- a demonstration that was
never written down and one that was must be produced by identical code, or the
dataset documents a program that no longer exists.

On time alignment: ``state`` and ``action`` are taken from the same iteration of
the control loop, so ``action[t]`` is the command issued at ``state[t]``. Images
cannot be -- they arrive on their own topic at the renderer's pace -- so each
recorded frame carries the image's age instead of pretending it is simultaneous.
``ml/convert.py`` reports the distribution of those ages; read it before
training on a dump.
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
from omx_bridge_app.isaac_prim import IsaacPrim, IsaacPrimError
from omx_vla_app import task_config
from omx_vla_app.ik import OMXKinematics
from omx_vla_app.spawn import instruction_for, sample_cube_pose, sample_scene

from .recorder import Recorder


APPROACH, DESCEND, CLOSE, LIFT, HOLD, DONE, FAILED = range(7)
PHASE_NAMES = ["接近", "下降", "夾合", "抬升", "保持", "完成", "失敗"]


class PickCubeExpert:
    """A waypoint state machine, stepped once per control tick (30 Hz).

    Tool-space interpolation rather than joint-space: the gripper then travels in
    straight lines at a controlled speed, which is both easier to imitate and
    what the eventual policy's action space describes.
    """

    def __init__(self, cfg: task_config.Config, kin: OMXKinematics, seed: int = 0):
        self.cfg = cfg
        self.kin = kin
        self.e = cfg["expert"]
        self.n = self.e["noise"]
        self.rng = np.random.default_rng(seed)
        self.gi = cfg.joints.index(cfg["gripper"]["joint"])
        self.dt = 1.0 / cfg["timing"]["fps"]

    def _reject(self, why, cube_pos):
        """Record why an episode could not be planned, then decline it.

        A bare False makes every planning failure look the same in the log, and
        they are not: "out of reach" and "the cube is not where it was put" want
        opposite fixes.
        """
        self.reject_reason = f"{why} (方塊在 {np.round(cube_pos, 4)})"
        return False

    # ── planning ───────────────────────────────────────────────────────
    def reset(self, q_now, cube_pos, cube_yaw):
        """Plan an episode. Returns False when the cube is simply unreachable."""
        grip = self.cfg["gripper"]
        grasp_z = self.cfg["cube"]["size"] / 2
        jit = lambda s: self.rng.normal(0.0, s)  # noqa: E731
        self.reject_reason = None

        # Aim slightly off, on purpose, and differently every episode.
        gx = cube_pos[0] + jit(self.n["waypoint_xy"])
        gy = cube_pos[1] + jit(self.n["waypoint_xy"])
        self.grasp_at = np.array([gx, gy, grasp_z])

        # Chosen once and frozen: re-choosing per waypoint lets the solution flip
        # elbow mid-trajectory, a discontinuity in the recorded action.
        if self.kin.solve(self.grasp_at, cube_yaw) is None:
            return self._reject("抓取點無解", cube_pos)
        self.elbow = max(
            ("up", "down"),
            key=lambda e: (lambda q: -1e9 if q is None else self.kin.fk(q)[1][1:, 2].min())(
                self.kin.ik(self.grasp_at, cube_yaw, elbow=e)),
        )
        if self.kin.ik(self.grasp_at, cube_yaw, elbow=self.elbow) is None:
            return self._reject("選定的肘姿抓取點無解", cube_pos)

        prefer = max(self.e["hover_floor"],
                     self.e["hover_prefer"] + jit(self.n["hover_height"]))
        found = self.kin.hover(self.grasp_at, cube_yaw,
                               prefer=prefer, floor=self.e["hover_floor"])
        if found is None:
            return self._reject("懸停點無解", cube_pos)
        self.hover_at, _ = found
        self.q_hover = self.kin.ik(self.hover_at, cube_yaw, elbow=self.elbow)
        if self.q_hover is None:
            return self._reject("懸停姿態無解", cube_pos)
        # Same reachability treatment as the hover: the far edge of the annulus
        # can plan a lift it cannot perform, and failing at plan time costs one
        # rejected episode instead of one truncated demonstration.
        need = self.cfg["success"]["lift_height"]
        found = self.kin.hover(self.grasp_at, cube_yaw,
                               prefer=need + self.e["lift_extra"],
                               floor=need + self.e["lift_margin"])
        if found is None:
            return self._reject("抬升點無解", cube_pos)
        self.lift_at, _ = found

        self.yaw = cube_yaw
        # The first leg is **joint space**, not tool space. At home the gripper
        # points forward; that same point with the gripper turned to face down
        # puts the wrist 0.38 m from a shoulder with 0.28 m of planar reach, so
        # every tool-space step of the transit would be unsolvable.
        self.q_start = np.asarray(q_now, dtype=float)[:5].copy()
        self.phase = APPROACH
        self.tool = self.hover_at.copy()
        self.gripper = grip["open"]
        self.timer = 0
        self.ticks = 0
        self.close_for = max(1, self.e["close_ticks"]
                             + int(round(jit(self.n["close_ticks"]))))
        self.settle = 0
        self.settle_timeouts = 0
        return True

    # ── stepping ───────────────────────────────────────────────────────
    def _step_joints(self):
        """Joint-space leg: walk from the start pose toward the hover pose."""
        delta = self.q_hover - self.q_start
        span = float(np.abs(delta).max())
        stride = self.e["joint_speed"] * self.dt
        if span <= stride:
            self.q_start = self.q_hover.copy()
            self.phase = DESCEND
            return self.q_hover
        self.q_start = self.q_start + delta / span * stride
        return self.q_start

    def _settled(self, goal, measured):
        """Has the **arm** reached ``goal``, not just the command?

        The command is an interpolation this class owns; the arm is a set of PD
        drives chasing it, and it lands about 14 mm behind against a 25 mm cube.
        Closing the fingers on the command means closing them above the cube.

        Callers that cannot measure pass ``None`` and skip the wait. A timeout
        releases the phase either way, so an arm that never converges cannot hang
        the episode; ``settle_timeouts`` counts those.
        """
        if measured is None:
            return True
        if np.linalg.norm(goal - np.asarray(measured)) <= self.e["settle_tol"]:
            self.settle = 0
            return True
        self.settle += 1
        if self.settle >= self.e["settle_max_ticks"]:
            self.settle = 0
            self.settle_timeouts += 1
            return True
        return False

    def _step_tool(self, measured=None):
        """Tool-space leg: straight lines at a controlled speed."""
        goal = {DESCEND: self.grasp_at, CLOSE: self.grasp_at,
                LIFT: self.lift_at, HOLD: self.lift_at}[self.phase]
        speed = self.e["descend_speed"] if self.phase == DESCEND else self.e["speed"]
        delta = goal - self.tool
        dist = float(np.linalg.norm(delta))
        stride = speed * self.dt
        self.tool = goal.copy() if dist <= stride else self.tool + delta / dist * stride
        arrived = dist <= self.e["arrive_tol"]

        if self.phase == DESCEND and arrived and self._settled(goal, measured):
            self.phase, self.timer = CLOSE, 0
        elif self.phase == CLOSE:
            self.gripper = self.cfg["gripper"]["grasp"]
            self.timer += 1
            if self.timer >= self.close_for:
                self.phase = LIFT
        elif self.phase == LIFT and arrived and self._settled(goal, measured):
            self.phase, self.timer = HOLD, 0
        elif self.phase == HOLD:
            # Longer than hold_steps on purpose: success needs that many
            # **consecutive** frames and the count restarts whenever the cube
            # swings past max_ee_distance, so an exact hold leaves no slack.
            self.timer += 1
            if self.timer >= self.cfg["success"]["hold_steps"] + self.e["hold_margin"]:
                self.phase = DONE
        return self.kin.ik(self.tool, self.yaw, elbow=self.elbow)

    def act(self, measured=None):
        """The next joint-target vector, and whether the episode is over.

        ``measured`` is the tool position the arm is **actually** at, in world
        coordinates -- pass ``kin.fk(q[:5])[0]`` of the measured joints. Given
        it, the phases that must not start early wait for the arm rather than
        for the command. See ``_settled``.

        Returns ``(targets, finished)``. ``targets`` is exactly the payload that
        would go on ``/sync/command``: the six canonical joints, radians, USD
        frame, in profile order.
        """
        self.ticks += 1
        if self.ticks > self.e["max_ticks"]:
            self.phase = FAILED
        if self.phase in (DONE, FAILED):
            return None, True

        q = (self._step_joints() if self.phase == APPROACH
             else self._step_tool(measured))
        if q is None:
            self.phase = FAILED
            return None, True

        out = np.zeros(len(self.cfg.joints), dtype=np.float32)
        out[:5] = q + self.rng.normal(0.0, self.n["joint_jitter"], 5)
        out[self.gi] = self.gripper
        return out, False


class ExpertClient(Node):
    """Reads the twin's joints and the cube's pose; writes commands and cube poses."""

    def __init__(self, profile, cfg):
        super().__init__("vla_expert")
        self.profile = profile
        self.cfg = cfg
        ros = cfg["ros"]
        # ⚠️ Works against both scenes. The three-object task lists them under
        # `objects`; the frozen single-cube spec (task/pick_cube_1obj.task.yaml,
        # which the M5/M6 checkpoints were trained against) has no such key and
        # falls back to the one frame it does name. Keeping one code path means
        # the old models stay evaluable without a git checkout.
        self.object_keys = ([o["key"] for o in cfg["objects"]]
                            if "objects" in cfg.raw else [ros["cube_frame"]])
        self.object_prims = {k: f"{cfg.task_root}/{k}" for k in self.object_keys}
        self.target = self.object_keys[0]      # set per episode by place_objects
        self.prim = IsaacPrim(self, ros["get_attribute_service"],
                              ros["set_attribute_service"])

        self._q = None
        self._sim_t = None                     # the twin's clock -- see wait_sim
        self._objs = {}                        # key -> (position, yaw)
        self._frames_seen = set()
        self._warned_yaw = False
        self.place_retries = 0
        self.loop_overruns = 0                 # ticks that outran the sim clock
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
        # The twin's own clock. /joint_states is published from the scene's
        # on_playback_tick and stamped by isaac_read_simulation_time, so this
        # advances once per simulation tick -- see wait_sim for why the control
        # loop runs on it instead of on wall clock.
        self._sim_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
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
            if t.child_frame_id not in self.object_prims:
                continue
            p, r = t.transform.translation, t.transform.rotation
            self._objs[t.child_frame_id] = (np.array([p.x, p.y, p.z]),
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
        """Block until ``get()`` stops returning None.

        ⚠️ Ready means *not None*, never truthiness: the predicates here return
        the thing waited for (a joint array, a pose), and ``not ndarray`` raises.
        A predicate that answers with a plain bool therefore has to return None
        for "not ready" -- ``False is not None``, so a bool False reads as ready
        and turns the whole wait into a silent no-op. ``_ready`` wraps that.
        """
        deadline = time.monotonic() + timeout
        while get() is None:
            if time.monotonic() > deadline:
                print(f"[expert] 等不到{what}。{hint}")
                return False
            time.sleep(0.02)
        return True

    @staticmethod
    def _ready(flag):
        """None until ``flag`` is true -- the shape ``_wait`` expects of a bool."""
        return True if flag else None

    def wait_for_joints(self, timeout=15.0):
        return self._wait(
            lambda: self._q, timeout,
            f"「{self._ep.state_topic}」上的關節狀態",
            "Isaac 開了嗎?Play 按了嗎?ROS_DOMAIN_ID 跟 Isaac 一致嗎?")

    def wait_for_cube(self, timeout=15.0):
        """Block until **every** object has been seen on TF.

        ⚠️ Waiting for all of them, not the first: a scene rebuilt with only
        some of the objects publishes a partial tree, and an episode that starts
        anyway would place a distractor it can never read back -- which surfaces
        much later as an unexplained placement failure.
        """
        want = set(self.object_keys)
        ok = self._wait(
            lambda: self._ready(want <= set(self._objs)), timeout,
            f"「{self.cfg['ros']['cube_tf_topic']}」上的 {', '.join(sorted(want))} frame",
            f"Isaac 載的場景要跟任務規格一致({self.cfg['paths']['scene_usd']})。")
        if not ok:
            missing = sorted(want - set(self._objs))
            print(f"          缺少的 frame:{missing}")
            if self._frames_seen:
                print(f"          該 topic 上看到的:{sorted(self._frames_seen)}")
        return ok

    def wait_sim(self, dt, timeout=5.0):
        """Block until the **simulation** clock has advanced by ``dt`` seconds.

        This is the control loop's pacing, and wall clock cannot do the job.

        ⚠️ **Isaac's run loop is not rate-limited under headless streaming.**
        Measured on an RTX Pro 6000: the simulation advances 2.42 s per second
        of wall clock. ``rateLimitEnabled``, ``rateLimitFrequency``,
        ``useFixedTimeStepping`` and ``useFastMode`` were all passed on the
        command line and all had **no effect** -- headless has no swapchain
        present for the limiter to hang off. A slower card hides the bug rather
        than avoiding it: when rendering is itself the bottleneck the ratio
        lands near 1.0x by accident, and the same code looks correct.

        With ``time.sleep(1/30)`` the loop therefore issued one command per
        **80 ms of simulated time** instead of 33 ms. That is not a small
        timing error -- it moves the seed that position-only IK is decided by
        (§5: the null space has 2 DOF and no selection criterion), and the grasp
        pitch drifts out of the range where a 5-DOF arm has a solution at all:

            accidentally rate-limited   20/20, pitch 32-44°
            unthrottled (2.42x)         11/20, pitch -12..72° (mean 13°),
                                        9 episodes rejected as "抓取點無解"

        Waiting on the twin's clock makes the loop rate correct regardless of
        how fast the renderer happens to be, which is also what lets recordings
        from different hardware be merged into one dataset.

        Returns False if the clock stopped advancing -- paused, or Isaac gone --
        rather than hanging the episode forever.
        """
        start = self._sim_t
        if start is None:
            # No state yet. The caller is inside wait_for_joints' territory;
            # fall back rather than spin forever on a clock that is not running.
            time.sleep(dt)
            return False
        if (self._sim_t - start) >= dt:
            # The tick's own work (IK, mostly) already took longer than a
            # control period of simulated time, so there is nothing to wait for
            # and the loop is running at whatever rate the solver allows. That
            # is the wall-clock failure mode again, arrived at from the other
            # side, and it is silent unless counted -- the run summary prints it.
            self.loop_overruns += 1
            return True
        deadline = time.monotonic() + timeout
        while (self._sim_t - start) < dt:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.0005)
        return True

    # ── outgoing ───────────────────────────────────────────────────────
    def joints(self):
        return None if self._q is None else self._q.copy()

    def mimic_error(self):
        """How far the passive finger is from mirroring the driven one, in rad."""
        if self._mimic is None or self._q is None:
            return float("nan")
        gi = self.profile.joints.index(self.cfg["gripper"]["joint"])
        return float(self._mimic + self._q[gi])      # multiplier -1 -> sum is 0

    def cube(self, key=None):
        """Pose of one object. Defaults to **this episode's target**.

        ⚠️ Defaulting to the target rather than to a fixed name is what keeps
        `run_episode` and `eval.py` unchanged: they ask "where is the thing I am
        supposed to pick up", and with one object on the table that is the same
        question as before.
        """
        got = self._objs.get(key or self.target)
        return None if got is None else (got[0].copy(), got[1])

    def objects(self):
        """Every object's pose -- for the recorder, which logs all of them."""
        return {k: (v[0].copy(), v[1]) for k, v in self._objs.items()}

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

    def _write_pose(self, key, position, yaw):
        """Write one object's pose. Returns False if Isaac refused the write."""
        prim = self.object_prims[key]
        try:
            self.prim.set(prim, "xformOp:translate",
                          [float(v) for v in position])
        except IsaacPrimError as exc:
            print(f"      ! 寫入 {key} 位置失敗:{exc}")
            return False
        try:
            self.prim.set(prim, "xformOp:orient",
                          [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        except IsaacPrimError as exc:
            # Position is what the episode needs; yaw only varies how the cube is
            # presented. Losing it silently would quietly narrow the dataset, so
            # say so, once, and carry on.
            if not self._warned_yaw:
                print(f"      ! {key} 的 yaw 設不了({exc}) —— 位置有效,朝向固定")
                self._warned_yaw = True
        return True

    def _wait_placed(self, placements, settle, still_tol):
        """Did **every** object arrive and stop moving?

        Checked through TF rather than trusted: a write that quietly does nothing
        leaves the cube where the last episode left it, which reads as a policy
        failure rather than a plumbing failure. Stillness rather than zeroed
        velocity, because TF carries no velocity and "has stopped" is the
        property actually wanted.
        """
        deadline = time.monotonic() + settle
        arrived, last = {}, {}
        while time.monotonic() < deadline:
            time.sleep(0.1)
            for key, want in placements.items():
                here = self.cube(key)
                if here is None:
                    continue
                if key not in arrived:
                    if np.linalg.norm(here[0][:2] - np.asarray(want[0])[:2]) < 5e-3:
                        arrived[key] = True
                    continue
                prev = last.get(key)
                if prev is not None and np.linalg.norm(here[0] - prev) < still_tol:
                    arrived[key] = "still"
                last[key] = here[0]
            if len(arrived) == len(placements) and all(
                    v == "still" for v in arrived.values()):
                return True
        return len(arrived) == len(placements)

    def place_objects(self, placements, target=None, settle=2.0,
                      still_tol=1e-4, attempts=3):
        """Teleport every object where the episode wants it.

        ``placements`` is ``{key: (position, yaw)}``; ``target`` names the object
        this episode is about and becomes what bare ``cube()`` returns.

        ⚠️ The write occasionally does not take -- about 1 episode in 20, root
        cause unconfirmed, most likely a race between the service applying the
        USD edit and PhysX owning the body. Retrying means **re-issuing every
        write**, not waiting longer: the failure is the write not landing.

        ⚠️ All objects are re-placed every episode, including ones that did not
        move. A distractor left where the previous episode's arm nudged it makes
        the clutter distribution drift over a long recording -- slowly, and in a
        way nothing would flag.
        """
        if target is not None:
            self.target = target
        for attempt in range(1, attempts + 1):
            wrote = all(self._write_pose(k, p, y) for k, (p, y) in placements.items())
            if wrote and self._wait_placed(placements, settle, still_tol):
                self.place_retries += attempt - 1
                if attempt > 1:
                    print(f"      · 物體第 {attempt} 次嘗試才放到位")
                return True
        self.place_retries += attempts - 1
        return False

    def place_cube(self, position, yaw, **kw):
        """Single-object shim -- what the frozen 1-object task and eval.py call."""
        return self.place_objects({self.object_keys[0]: (position, yaw)},
                                  target=self.object_keys[0], **kw)


def run_episode(client, expert, kin, cfg, pos, yaw, snap=None, rec=None,
                placements=None, target=None):
    """Reset, run the state machine, return (success, phase, ticks).

    ``placements``/``target`` carry the multi-object scene. Without them this is
    the single-object task unchanged, which is what keeps the frozen
    ``pick_cube_1obj`` spec working off the same code.
    """
    grip = cfg["gripper"]
    succ = cfg["success"]
    grasp_z = cfg["cube"]["size"] / 2
    period = 1.0 / cfg["timing"]["fps"]

    client.go_home(grip["open"])
    client.wait_for_release()
    placed = (client.place_objects(placements, target=target) if placements
              else client.place_cube(pos, yaw))
    if not placed:
        here = client.cube()
        print(f"      ✗ 方塊沒有到位:目標 {np.round(pos, 3)},"
              f" 實際 {np.round(here[0], 3) if here else '讀不到'}")
        return False, FAILED, 0, None, float('nan'), (float('nan'),) * 3

    at, at_yaw = client.cube()
    # Reseed before planning, for the same reason the loop below reseeds every
    # tick (§5: the null space has 2 DOF and no selection criterion, so the
    # answer is decided almost entirely by the seed).
    #
    # ⚠️ ``expert.reset`` solves IK for the grasp point, and without this line
    # the seed is still whatever the **previous episode's last tick** left --
    # the arm up in the air holding a cube. A bad ending then poisons the next
    # episode's plan; that episode never enters the loop, so the seed is never
    # refreshed, and the same stale seed fails again. The failures come in
    # blocks, not singly: episodes 14-16 of a 20-episode run went down together
    # behind one anomalous episode 13, and 9 in a row before the loop was fixed.
    #
    # The arm is at home by this point, so seeding from the measurement also
    # makes each episode's plan independent of how the last one ended.
    if hasattr(kin, "seed"):
        kin.seed = client.joints()[:5]
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
        # The twin's clock, not the wall's -- see ExpertClient.wait_sim. On a
        # machine where the simulation outruns real time, sleeping here is what
        # silently turned 20/20 into 11/20.
        client.wait_sim(period)
        if expert.phase in (LIFT, HOLD):
            cube = client.cube()[0]
            q = client.joints()
            # Height plus distance to the pinch point -- not "do the fingers
            # look like they touch". Any convex approximation of the curved
            # finger is fatter than the rendered mesh and is not drawn, so a
            # firm grasp still shows a gap.
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
    #   analytic       omx_vla_app/ik.py -- closed form, gripper pointing down
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
            placements = target = None
            if "objects" in cfg.raw:
                placements, target, r, th = sample_scene(cfg, rng, holdout)
                pos, yaw = placements[target]
            else:
                pos, yaw, r, th = sample_cube_pose(cfg, rng, holdout)
            if fixed_theta < 900.0:            # 重現某個特定位置
                th = fixed_theta
                r = fixed_radius or r
                a = math.radians(th)
                pos = np.array([r * math.cos(a), r * math.sin(a),
                                cfg["cube"]["size"] / 2])
            snap = f"ep{i + 1:02d}_r{r * 1000:.0f}_t{th:+.0f}" if save_frames else None
            if rec is not None:
                meta = {"seed": seed, "episode": i + 1, "holdout": bool(holdout),
                        "cube": {"r": float(r), "theta_deg": float(th),
                                 "yaw_rad": float(yaw),
                                 "requested": [float(v) for v in pos]}}
                if placements:
                    # ⚠️ The instruction and the target key go in the dump, not
                    # just the target's pose. convert.py turns the instruction
                    # into the dataset's `task` column, and without the key
                    # there is no way to score "did it pick the right one"
                    # after the fact.
                    meta["target"] = target
                    meta["instruction"] = instruction_for(cfg, target)
                    meta["objects"] = {k: {"requested": [float(v) for v in q[0]],
                                           "yaw_rad": float(q[1])}
                                       for k, q in placements.items()}
                rec.begin(i + 1, meta)
            ok, phase, ticks, residual, lag, geom = run_episode(
                client, expert, kin, cfg, pos, yaw, snap, rec, placements, target)
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
            tag = f" [{target}]" if target else ""
            print(f"  第 {i + 1:3d} 集{tag}  r={r * 1000:.0f}mm θ={th:+.0f}° "
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
        if client.loop_overruns:
            # Not cosmetic: every overrun tick is one where the arm advanced
            # further than a control period, which is exactly what wait_sim
            # exists to prevent. A handful at phase changes is fine; a large
            # fraction means the solver, not the clock, is now setting the rate.
            print(f"⚠️ 控制迴圈追不上模擬的 tick 數:{client.loop_overruns}"
                  f" —— 這些 tick 前進得比一個控制週期多,IK 種子會漂")

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
