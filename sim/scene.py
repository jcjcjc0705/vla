"""The pick-cube episode: reset, step, observe, decide success.

Import this **after** ``app.start()`` -- the isaacsim modules it needs only exist
once the SimulationApp is up.

Everything numeric comes from ``task/pick_cube.task.yaml``. Joint order comes
from ``sim_real_bridge.profile`` via task_config.py, so the vector this produces is
already a valid ``/sync/command`` payload: same six joints, same order, radians,
USD frame. That is the seam that lets the eventual policy drive the real arm
without reinterpreting anything.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_config  # noqa: E402


class PickCubeScene:
    def __init__(self, cfg: task_config.Config, with_cameras: bool = True):
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.stage import open_stage

        self.cfg = cfg
        self.task_root = cfg.task_root
        t = cfg["timing"]
        self.render_every = t["render_every"]

        if not cfg.scene_usd.exists():
            raise FileNotFoundError(
                f"{cfg.scene_usd} 不存在 —— 先跑 sim/build_scene.py"
            )
        open_stage(str(cfg.scene_usd))

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=t["physics_dt"],
            rendering_dt=t["physics_dt"] * self.render_every,
        )
        self.arm = self.world.scene.add(
            SingleArticulation(prim_path=cfg.robot_root, name="omx_f")
        )
        self.cube = self.world.scene.add(
            SingleRigidPrim(prim_path=f"{self.task_root}/cube", name="cube")
        )
        self.ee = SingleXFormPrim(prim_path=f"{cfg.robot_root}/link5/end_effector_link")

        self.world.reset()

        # The articulation carries 7 DOFs (gripper_joint_2 is a real PhysX mimic
        # joint, not decoration). We only ever command the 6 canonical ones.
        self.dof_index = [self.arm.dof_names.index(j) for j in cfg.joints]
        self.n_dof = len(self.arm.dof_names)
        self._targets = np.zeros(self.n_dof, dtype=np.float32)

        self.cameras = self._make_cameras() if with_cameras else {}
        self._rng = np.random.default_rng(0)
        self._hold = 0

    def _make_cameras(self):
        from isaacsim.sensors.camera import Camera

        specs = self.cfg["cameras"]
        cams = {}
        for name, path in (
            ("front", f"{self.task_root}/cam_front"),
            ("wrist", f"{specs['wrist']['parent']}/cam_wrist"),
        ):
            cam = Camera(
                prim_path=path,
                resolution=tuple(specs[name]["resolution"]),
                frequency=self.cfg["timing"]["fps"],
            )
            cam.initialize()
            cams[name] = cam
        return cams

    # ── episode ────────────────────────────────────────────────────────
    def sample_cube_pose(self, seed: int, holdout: bool = False):
        """Cube pose for one episode, drawn from the spawn annulus.

        ``holdout`` draws from the theta band deliberately kept out of training.
        Evaluating there is what distinguishes a policy that looks at the image
        from one that has memorised an average trajectory.
        """
        s = self.cfg["spawn"]
        rng = np.random.default_rng(seed)
        band = s["holdout_theta_deg"] if holdout else s["theta_deg"]
        if not holdout:
            # Training draws from the full sweep minus the held-out band.
            lo, hi = s["holdout_theta_deg"]
            while True:
                th = rng.uniform(*band)
                if not (lo <= th <= hi):
                    break
        else:
            th = rng.uniform(*band)
        r = rng.uniform(*s["radius"])
        yaw = math.radians(rng.uniform(*s["yaw_deg"]))
        th = math.radians(th)
        pos = np.array([r * math.cos(th), r * math.sin(th), self.cfg["cube"]["size"] / 2])
        quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])  # w,x,y,z
        return pos, quat

    def reset(self, seed: int = 0, holdout: bool = False, cube_pose=None):
        """Home the arm, place the cube, let it settle."""
        self.world.reset()
        self.arm.set_joint_positions(np.zeros(self.n_dof))
        self.arm.set_joint_velocities(np.zeros(self.n_dof))
        self._targets = np.zeros(self.n_dof, dtype=np.float32)

        pos, quat = cube_pose if cube_pose is not None else self.sample_cube_pose(seed, holdout)
        self.place_cube(pos, quat)
        for _ in range(self.cfg["timing"]["settle_steps"]):
            self.world.step(render=False)
        self._hold = 0
        return self.observe(render=False)

    def place_cube(self, position, orientation=None):
        """Put the cube somewhere and kill its momentum."""
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.cube.set_world_pose(position=np.asarray(position, dtype=float),
                                 orientation=np.asarray(orientation, dtype=float))
        self.cube.set_linear_velocity(np.zeros(3))
        self.cube.set_angular_velocity(np.zeros(3))

    # ── act ────────────────────────────────────────────────────────────
    def set_targets(self, canonical):
        """Command the six canonical joints, in radians (== the USD frame).

        This vector is exactly what would go on ``/sync/command``.

        ``apply_action`` sets the **drive targets** and lets the PD drives track
        them, which is what a position command means. ``set_joint_positions``
        would teleport the joints instead -- no contact forces, no dynamics, and
        a grasp test that always "succeeds".

        Only the six canonical indices are addressed, so the passive mimic joint
        is left to PhysX rather than being commanded behind its own constraint.
        """
        from isaacsim.core.utils.types import ArticulationAction

        q = np.asarray(canonical, dtype=np.float32)
        if q.shape != (len(self.cfg.joints),):
            raise ValueError(f"要 {len(self.cfg.joints)} 個關節,拿到 {q.shape}")
        self._targets = q
        self.arm.apply_action(
            ArticulationAction(joint_positions=q, joint_indices=np.array(self.dof_index))
        )

    def step(self, render: bool = False):
        self.world.step(render=render)

    # ── observe ────────────────────────────────────────────────────────
    def joint_positions(self):
        """Measured joint positions -- not the commanded targets.

        Recording the command as the state would teach the policy an identity
        map: excellent training loss, useless behaviour.
        """
        return np.asarray(self.arm.get_joint_positions(), dtype=np.float32)[self.dof_index]

    def observe(self, render: bool = True):
        obs = {"state": self.joint_positions()}
        if render and self.cameras:
            obs["images"] = {
                name: cam.get_rgba()[:, :, :3] for name, cam in self.cameras.items()
            }
        return obs

    def cube_pose(self):
        pos, quat = self.cube.get_world_pose()
        return np.asarray(pos), np.asarray(quat)

    def ee_position(self):
        return np.asarray(self.ee.get_world_pose()[0])

    # ── succeed ────────────────────────────────────────────────────────
    def success(self):
        """Lifted, still held, and stayed that way.

        ``held`` is what rejects a cube that was merely flicked into the air;
        requiring consecutive steps is what rejects a momentary bounce. A single
        instantaneous check passes for both of those, which is why neither term
        is optional.
        """
        s = self.cfg["success"]
        cube = self.cube_pose()[0]
        lifted = cube[2] > self.cfg["cube"]["size"] / 2 + s["lift_height"]
        held = np.linalg.norm(cube - self.ee_position()) < s["max_ee_distance"]
        self._hold = self._hold + 1 if (lifted and held) else 0
        return self._hold >= s["hold_steps"]

    def diagnose(self):
        """Why success() is or is not firing -- for tuning, not for the reward."""
        s = self.cfg["success"]
        cube = self.cube_pose()[0]
        return {
            "cube_z": float(cube[2]),
            "lift_needed": self.cfg["cube"]["size"] / 2 + s["lift_height"],
            "ee_dist": float(np.linalg.norm(cube - self.ee_position())),
            "max_ee_dist": s["max_ee_distance"],
            "hold": self._hold,
            "hold_needed": s["hold_steps"],
        }
