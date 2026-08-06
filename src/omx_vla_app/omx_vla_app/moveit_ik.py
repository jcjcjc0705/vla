"""MoveIt's own IK for this arm, behind the same interface as ``sim/ik.py``.

ROBOTIS ships a moveit_config for ``omx_f``; the image bakes it at ``/moveit``.
This uses it directly -- the KDL plugin, that ``kinematics.yaml``, that SRDF --
so "MoveIt IK" here means MoveIt's, not a reimplementation.

**In-process, via ``MoveItPy``.** Calling ``/compute_ik`` on a running
``move_group`` would put a service round trip inside a 30 Hz control loop; this
loads the solver into the node instead. Nothing plans and nothing executes:
trajectory interpolation and ``/sync/command`` stay exactly as they were, so the
recorded actions mean the same thing whichever solver produced them.

``kinematics.yaml`` sets ``position_only_ik: True``, which is not a shortcut --
the arm is 5-DOF and cannot reach an arbitrary 6-DOF pose. The consequence is
real and worth knowing before choosing this solver: **the gripper's orientation
is whatever falls out.** Measured at the grasp, it lands around 32-44 degrees of
pitch rather than pointing down, and the grasp still succeeds. What it does not
give is control of ``joint5``, which is the joint that lines the fingers up with
the cube's faces; that ends up wherever the seed left it.

Interface parity with ``OMXKinematics`` is deliberate -- ``fk``, ``ik``,
``solve``, ``hover``, ``in_ee``, ``link5_frame`` -- so the expert takes either
without knowing which.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import yaml
from geometry_msgs.msg import Pose


class MoveItKinematics:
    """MoveIt's KDL solver, wrapped to look like ``OMXKinematics``.

    FK is delegated to the analytic model rather than to MoveIt: it is exact,
    already verified against Isaac to 0.024 mm, and needed on every tick for the
    settle gate. Only the *inverse* direction comes from MoveIt here.
    """

    def __init__(self, cfg, analytic, config_dir="/moveit",
                 urdf=None, group="arm", tip="end_effector_link",
                 collision_retries=8, seed_spread=0.4):
        from moveit.planning import MoveItPy

        self.cfg = cfg
        self.analytic = analytic
        self.group = group
        self.tip = tip
        self.reach = analytic.reach
        self.ee_local = analytic.ee_local
        self.tool_local = analytic.tool_local

        d = pathlib.Path(config_dir)
        # The MoveIt copy, whose mesh paths are absolute. The one in assets/ has
        # relative paths, which MoveIt resolves as URLs and silently gives up on
        # -- leaving the robot with no collision geometry.
        urdf_path = pathlib.Path(urdf) if urdf else (
            d / "omx_f.urdf" if (d / "omx_f.urdf").exists() else pathlib.Path(cfg.robot_urdf))
        self._mi = MoveItPy(
            node_name="omx_moveit_ik",
            config_dict={
                "robot_description": urdf_path.read_text(),
                "robot_description_semantic": (d / "omx_f.srdf").read_text(),
                "robot_description_kinematics":
                    yaml.safe_load((d / "kinematics.yaml").read_text()),
                # A pipeline has to be named even though nothing plans here.
                "planning_pipelines": {"pipeline_names": ["ompl"]},
                "ompl": {"planning_plugins": ["ompl_interface/OMPLPlanner"]},
            },
            provide_planning_service=False,
        )
        self.model = self._mi.get_robot_model()

        from moveit.core.robot_state import RobotState
        self._state = RobotState(self.model)
        self._state.set_to_default_values()
        self._n = len(cfg.joints) - 1          # the 5 arm joints

        # ⚠️ Position-only IK has a 2-DOF null space and no criterion for picking
        # within it, so the answer is decided almost entirely by the seed. Left
        # to chain off its own previous answer the solver drifts, and the drift
        # is chaotic: changing the collider approximation -- which cannot touch
        # kinematics -- moved the mean grasp pitch from 44 to 7 degrees, because
        # it changed how many ticks each phase took and therefore how many times
        # the seed had been fed back into itself.
        #
        # Callers set `seed` from the **measured** joints each tick. That is both
        # what a controller actually does and what makes a run reproducible.
        self.seed = None

        # Collision checking uses the real meshes and ROBOTIS's own allowed-
        # collision matrix from the SRDF -- not an approximation of either.
        # Needed because nothing else catches a self-intersecting pose: Isaac's
        # articulation ships with enabledSelfCollisions = False, and
        # moveit_py's set_from_ik exposes no validity callback (the official
        # avoid_collisions flag lives on the /compute_ik service, which would
        # put a round trip inside the control loop).
        #
        # Rejecting a colliding solution outright would be too strict: with
        # position-only IK the 2-DOF null space usually holds a clear pose, and
        # the solution returned is decided almost entirely by the seed. So
        # perturb the seed and ask again -- that searches the null space, which
        # is what avoid_collisions does internally.
        self._scene = None
        self._retries = collision_retries
        self._spread = seed_spread
        self._rng = np.random.default_rng(0)
        self.rejected = 0
        try:
            from moveit.core.planning_scene import PlanningScene
            self._scene = PlanningScene(self.model)
        except Exception as exc:                      # noqa: BLE001
            print(f"[moveit_ik] 建不出 PlanningScene({exc}) —— 不做自碰撞檢查")

    def _collides(self, q):
        if self._scene is None:
            return False
        st = self._scene.current_state
        st.set_joint_group_positions(self.group, np.asarray(q, dtype=float))
        st.update()
        from moveit.core.collision_detection import CollisionRequest, CollisionResult
        req, res = CollisionRequest(), CollisionResult()
        self._scene.check_self_collision(req, res, st,
                                         self._scene.allowed_collision_matrix)
        return bool(res.collision)

    # ── forward: delegated ─────────────────────────────────────────────
    def fk(self, q):
        return self.analytic.fk(q)

    def link5_frame(self, q):
        return self.analytic.link5_frame(q)

    def in_ee(self, q, world_point):
        return self.analytic.in_ee(q, world_point)

    def in_link5(self, q, world_point):
        return self.analytic.in_link5(q, world_point)

    # ── inverse: MoveIt ────────────────────────────────────────────────
    def ik(self, target, cube_yaw=0.0, elbow="up", pitch=math.pi / 2, iters=4,
           timeout=0.02):
        """Joint angles putting the **pinch point** on ``target``.

        ``cube_yaw``, ``elbow`` and ``pitch`` are accepted and ignored: with
        position-only IK none of them can be requested. They stay in the
        signature so the two solvers remain interchangeable.

        Seeded from ``self.seed`` when the caller keeps it fed with measured
        joint positions; otherwise from the previous answer, which drifts.

        MoveIt solves for a *link*, so the pinch offset is removed first -- the
        request is for wherever ``end_effector_link`` has to be for the fingers
        to close on ``target``. Which orientation the solver returns is not known
        until it returns, so the offset is applied using the previous solution's
        orientation and then corrected once.
        """
        target = np.asarray(target, dtype=float)
        base = np.asarray(self.seed if self.seed is not None
                          else self._state.get_joint_group_positions(self.group),
                          dtype=float)[:self._n]
        # EE -> pinch, **not** link5 -> pinch. Subtracting tool_local instead
        # leaves the 91.9 mm end_effector_joint offset in, which shows up as the
        # arm settling that far from the cube.
        pinch = self.tool_local - self.ee_local

        for attempt in range(self._retries + 1):
            q = base if attempt == 0 else base + self._rng.normal(0, self._spread, self._n)
            self._state.set_joint_group_positions(self.group,
                                                  np.append(q, 0.0)[:len(base) + 1][:self._n])
            ok = True
            for _ in range(2):                # settle the pinch offset
                R = self.analytic.link5_frame(q)[1]
                want = target - R @ pinch
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = (float(v) for v in want)
                pose.orientation.w = 1.0
                if not self._state.set_from_ik(self.group, pose, self.tip, timeout):
                    ok = False
                    break
                q = np.asarray(self._state.get_joint_group_positions(self.group),
                               dtype=float)[:self._n]
            if not ok:
                continue
            if not self._collides(q):
                return q
            self.rejected += 1
        return None

    # ── the two helpers the expert uses, unchanged in behaviour ────────
    def solve(self, target, cube_yaw=0.0, min_z=0.01, **kw):
        q = self.ik(target, cube_yaw, **kw)
        if q is None:
            return None
        return q if self.fk(q)[1][1:, 2].min() >= min_z else None

    def hover(self, target, cube_yaw=0.0, prefer=0.12, floor=0.05, step=0.005):
        target = np.asarray(target, dtype=float)
        h = prefer
        while h >= floor - 1e-9:
            at = np.array([target[0], target[1], target[2] + h])
            q = self.solve(at, cube_yaw)
            if q is not None:
                return at, q
            h -= step
        return None
