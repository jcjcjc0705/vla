"""MoveIt's IK, adapted to the interface ``sim/ik.py`` presents.

The solver itself lives in the bridge image (``omx_bridge_app.moveit_kin``) --
it is this arm's kinematics, not this task's, and ``ik_target`` uses the same one.
What is here is only the adaptation:

* **FK comes from the analytic model**, not from MoveIt. It is exact, already
  verified against Isaac to 0.024 mm, and it is called on every tick by the
  settle gate. Only the *inverse* direction comes from MoveIt.
* the pinch offset is taken from ``task/pick_cube.task.yaml`` rather than from
  the solver's default, so the two solvers aim at the same point by construction.
* ``solve`` and ``hover`` -- the two helpers the expert calls.

``kinematics.yaml`` sets ``position_only_ik: True``, which is not a shortcut --
the arm is 5-DOF and cannot reach an arbitrary 6-DOF pose. The consequence is
real and worth knowing before choosing this solver: **the gripper's orientation
is whatever falls out.** Measured at the grasp it lands around 32-44 degrees of
pitch rather than pointing down, and the grasp still succeeds. What it does not
give is control of ``joint5``, the joint that lines the fingers up with the
cube's faces; that ends up wherever the seed left it.
"""
from __future__ import annotations

import numpy as np
from omx_bridge_app.moveit_kin import MoveItKin


class MoveItKinematics(MoveItKin):
    """``MoveItKin`` with the analytic model's forward kinematics and helpers."""

    def __init__(self, cfg, analytic, config_dir="/moveit", **kw):
        self.analytic = analytic
        # The task's own measurement, so both solvers aim at the same point.
        super().__init__(config_dir=config_dir,
                         pinch=analytic.tool_local - analytic.ee_local, **kw)
        self.cfg = cfg
        self.reach = analytic.reach
        self.ee_local = analytic.ee_local
        self.tool_local = analytic.tool_local

    # ── forward: delegated ─────────────────────────────────────────────
    def fk(self, q):
        return self.analytic.fk(q)

    def tip_frame(self, q):
        return self.analytic.link5_frame(q)

    def link5_frame(self, q):
        return self.analytic.link5_frame(q)

    def in_ee(self, q, world_point):
        return self.analytic.in_ee(q, world_point)

    def in_link5(self, q, world_point):
        return self.analytic.in_link5(q, world_point)

    # ── the two helpers the expert uses ────────────────────────────────
    def solve(self, target, cube_yaw=0.0, min_z=0.01, **kw):
        q = self.ik(target, **kw)
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
