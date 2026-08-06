"""Scripted pick-place expert: the state machine that produces the demonstrations.

Pure numpy -- no Isaac, no ROS -- so the same class runs wherever the episode is
being driven from. It is driven by ``omx_vla_app.expert_node``:

    ros2 run omx_vla_app expert --ros-args -p episodes:=20

The cube's pose comes from the simulator and IK is solved against it. That only
ever happens while *generating* data: what gets recorded as the observation is
images plus joint state, so the cube's coordinates never enter the dataset.

The trajectory is deliberately noisy (``expert.noise`` in the task yaml). A clean
scripted expert produces a nearly single-mode distribution, and a student trained
on it scores well by replaying an average trajectory while ignoring the image.
Widening the distribution is what forces the image to carry information.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))
import task_config  # noqa: E402
from ik import OMXKinematics  # noqa: E402

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
