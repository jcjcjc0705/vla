"""Run the scripted expert inside Isaac Sim's GUI, on a stage you opened yourself.

Sibling of gui_probe.py: it attaches to the running session instead of creating a
SimulationApp, so you keep your viewport, your camera, and your Play button. The
expert and the kinematics are the same modules the headless runs use -- there is
no second copy of the logic here.

The GUI drives physics, not this file, so the expert is advanced from a physics
step callback rather than a loop.

    1. bash ~/isaac_sim_5.1/isaac-sim.sh
    2. File -> Open -> vla/assets/pick_cube.usd
    3. press Play          (nothing moves until physics is running)
    4. Window -> Script Editor, then paste:

        exec(open('/home/jcjcjc/Desktop/screamlab/OMX_arm/vla/sim/gui_expert.py').read())

    5. one line at a time:

        run()            # random cube position, watch it pick
        run(seed=7)      # a specific one, repeatable
        run(holdout=True)  # from the region held out of training
        stop()           # abort the current episode

Output goes to the Console window. ASCII only -- the Script Editor mangles the
rest.
"""
import sys

VLA_SIM = "/home/jcjcjc/Desktop/screamlab/OMX_arm/vla/sim"
if VLA_SIM not in sys.path:
    sys.path.insert(0, VLA_SIM)

import math

import numpy as np
import omni.physx
import omni.usd
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction

import task_config
from expert import DONE, FAILED, HOLD, LIFT, PHASE_NAMES, PickCubeExpert
from ik import OMXKinematics

_cfg = task_config.load()
_kin = OMXKinematics(_cfg)
_stage = omni.usd.get_context().get_stage()
_arm = SingleArticulation(prim_path=_cfg.robot_root, name="gui_expert_arm")
_arm.initialize()
_cube = SingleRigidPrim(prim_path=f"{_cfg.task_root}/cube", name="gui_expert_cube")
_cube.initialize()
_ee = SingleXFormPrim(prim_path=f"{_cfg.robot_root}/link5/end_effector_link")
_idx = np.array([_arm.dof_names.index(j) for j in _cfg.joints])

_expert = PickCubeExpert(_cfg, _kin)
_sub = None
_state = {"n": 0, "ok": False}

# ASCII names, because the Script Editor console cannot print the real ones.
_PHASE_ASCII = ["approach", "descend", "close", "lift", "hold", "done", "FAILED"]


def _q():
    return np.asarray(_arm.get_joint_positions())[_idx]


def _succeeded():
    """Same rule as scene.success(), minus the consecutive-frame counter."""
    s = _cfg["success"]
    cube = np.asarray(_cube.get_world_pose()[0])
    ee = np.asarray(_ee.get_world_pose()[0])
    lifted = cube[2] > _cfg["cube"]["size"] / 2 + s["lift_height"]
    return bool(lifted and np.linalg.norm(cube - ee) < s["max_ee_distance"])


def _on_step(dt):
    """Advance the expert once per control tick; the GUI supplies the substeps."""
    _state["n"] += 1
    if _state["n"] % _cfg["timing"]["render_every"]:
        return

    targets, finished = _expert.act()
    if finished:
        phase = _PHASE_ASCII[_expert.phase]
        print(f"episode over: {phase}, {_expert.ticks} ticks, "
              f"{'SUCCESS' if _state['ok'] else 'no grasp'}")
        stop()
        return

    _arm.apply_action(ArticulationAction(joint_positions=targets, joint_indices=_idx))
    if _expert.phase in (LIFT, HOLD) and _succeeded():
        _state["ok"] = True


def run(seed=None, holdout=False):
    """Place the cube somewhere and let the expert pick it up.

    The arm starts from wherever it is -- the expert's first leg is a joint-space
    move, so there is no need to home it first.
    """
    global _sub
    stop()
    if seed is None:
        seed = np.random.randint(0, 10000)

    s = _cfg["spawn"]
    rng = np.random.default_rng(seed)
    band = s["holdout_theta_deg"] if holdout else s["theta_deg"]
    while True:
        th = rng.uniform(*band)
        lo, hi = s["holdout_theta_deg"]
        if holdout or not (lo <= th <= hi):
            break
    r = rng.uniform(*s["radius"])
    yaw = math.radians(rng.uniform(*s["yaw_deg"]))
    pos = np.array([r * math.cos(math.radians(th)), r * math.sin(math.radians(th)),
                    _cfg["cube"]["size"] / 2])

    _cube.set_world_pose(position=pos,
                         orientation=np.array([math.cos(yaw / 2), 0.0, 0.0,
                                               math.sin(yaw / 2)]))
    _cube.set_linear_velocity(np.zeros(3))
    _cube.set_angular_velocity(np.zeros(3))

    if not _expert.reset(_q(), pos, yaw):
        print(f"seed {seed}: cube at r={r:.3f} th={th:+.0f} is unreachable, "
              "nothing to run")
        return
    _state.update(n=0, ok=False)
    _sub = omni.physx.get_physx_interface().subscribe_physics_step_events(_on_step)
    print(f"seed {seed}: cube r={r * 1000:.0f}mm th={th:+.0f}deg yaw={math.degrees(yaw):+.0f}deg"
          f"  -- press Play if nothing moves")


def stop():
    """Abort whatever is running and hand control back."""
    global _sub
    if _sub is not None:
        _sub.unsubscribe()
        _sub = None


print("gui_expert loaded.  run()   run(seed=7)   run(holdout=True)   stop()")
print("Play must be running for physics steps to fire.")
