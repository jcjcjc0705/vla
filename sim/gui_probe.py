"""Helpers to run inside Isaac Sim's Script Editor, on a stage you opened yourself.

Unlike everything else in sim/, this does **not** create a SimulationApp and does
not open a stage -- it attaches to the running GUI session. That makes it the
right tool for questions best answered by looking: where the fingers actually
are, how wide they open, whether a cube fits between them.

    1. bash ~/isaac_sim_5.1/isaac-sim.sh          (or isaac-sim.streaming.sh)
    2. File -> Open -> vla/assets/pick_cube.usd
    3. press Play  (physics must be running for the joints to move)
    4. Window -> Script Editor, then paste:

        exec(open('/home/jcjcjc/Desktop/screamlab/OMX_arm/vla/sim/gui_probe.py').read())

    5. call the helpers, one line at a time:

        where()                # print fingers / EE / cube, in mm
        grip(0.6)              # open      (positive = open)
        grip(0.0)              # closed
        cube(0.27)             # put the cube at x=0.27 on the finger centre line
        cube(0.27, z=0.22)     # ... at a different height
        squeeze()              # close on whatever is there, then report

Output goes to the Console window (Window -> Console).
"""
import numpy as np
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from pxr import Usd, UsdGeom

import omni.usd

_stage = omni.usd.get_context().get_stage()
_arm = SingleArticulation(prim_path="/omx_f", name="probe_arm")
_arm.initialize()
_cube = SingleRigidPrim(prim_path="/World/task/cube", name="probe_cube")
_cube.initialize()
_ee = SingleXFormPrim(prim_path="/omx_f/link5/end_effector_link")

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1"]
_idx = [_arm.dof_names.index(j) for j in JOINTS]
_GI = 5                                  # gripper_joint_1 within JOINTS
_bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])


def _mm(v):
    return np.round(np.asarray(v) * 1000, 1)


def where():
    """Print where everything is, in millimetres.

    ASCII only: the Script Editor console mangles non-Latin text.
    """
    _bbox.Clear()
    q = np.asarray(_arm.get_joint_positions())[_idx]
    print(f"\ngripper_joint_1 = {q[_GI]:+.4f} rad")
    for link in ("link6", "link7"):
        r = _bbox.ComputeWorldBound(_stage.GetPrimAtPath(f"/omx_f/{link}")).ComputeAlignedRange()
        lo, hi = _mm(r.GetMin()), _mm(r.GetMax())
        print(f"  {link}  x[{lo[0]:7.1f},{hi[0]:7.1f}]  "
              f"y[{lo[1]:7.1f},{hi[1]:7.1f}]  z[{lo[2]:7.1f},{hi[2]:7.1f}]")
    ee_p, ee_q = _ee.get_world_pose()
    cube_p = _cube.get_world_pose()[0]
    print(f"  EE    {_mm(ee_p)}   quat(wxyz) {np.round(ee_q, 4)}")
    print(f"  cube  {_mm(cube_p)}")

    # The offset that matters is in the EE's own frame: the world-space
    # difference depends on how the arm happens to be posed right now, but the
    # grasp point is fixed relative to the gripper.
    w, x, y, z = [float(v) for v in ee_q]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    local = R.T @ (np.asarray(cube_p) - np.asarray(ee_p))
    print(f"  cube - EE : world {_mm(np.asarray(cube_p) - np.asarray(ee_p))} mm"
          f"   EE-local {_mm(local)} mm   <-- this is the number to record")
    print("  (link bboxes span the whole finger incl. the mount near the pivot,")
    print("   so they do not locate the gripping face; the EE-local offset does)")


def grip(value):
    """Drive gripper_joint_1 to ``value`` radians. Positive opens."""
    q = np.asarray(_arm.get_joint_positions())[_idx].astype(np.float32)
    q[_GI] = value
    _arm.apply_action(ArticulationAction(joint_positions=q, joint_indices=np.array(_idx)))
    print(f"gripper -> {value:+.3f} rad  (needs Play running to move)")


def cube(x, y=-0.0016, z=0.2106):
    """Place the cube and stop it moving. Defaults sit on the finger centre line."""
    _cube.set_world_pose(position=np.array([x, y, z], dtype=float))
    _cube.set_linear_velocity(np.zeros(3))
    _cube.set_angular_velocity(np.zeros(3))
    print(f"cube -> {_mm([x, y, z])} mm")


def squeeze(closed=0.0, hold=None):
    """Close the gripper, then report whether the cube stayed put.

    Run it after ``cube(...)``. Let a second or two of simulation pass between
    the call and reading the result -- this does not step the sim itself, the
    GUI does.
    """
    before = np.asarray(_cube.get_world_pose()[0]).copy()
    grip(closed)
    print(f"closing... wait a second, then where() or result({before[2]:.4f})")
    return before


def result(z_before):
    """Compare the cube's height now against before the squeeze."""
    z = float(_cube.get_world_pose()[0][2])
    drop = (z_before - z) * 1000
    print(f"drop {drop:+.1f} mm  ->  {'HELD' if drop < 15 else 'DROPPED'}")
    return drop


print(__doc__.split("    1.")[0])
print("available: where()  grip(rad)  cube(x[,y,z])  squeeze()  result(z_before)")
where()
