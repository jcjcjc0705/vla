# Design notes

Why the system is shaped the way it is. Most of what follows was learned by hitting the problem first;
the entries marked ⚠️ are the ones that failed *silently*, which is why they are written down.

The primary engineering record is Chinese: [zh/ENGINEERING_BRIEF.md](zh/ENGINEERING_BRIEF.md) and
[zh/PROGRESS_NOTES.md](zh/PROGRESS_NOTES.md). This is the distilled English version.

---

## The problem

Make an OpenManipulator-X follower (`omx_f`, 5-DOF arm + gripper) pick a cube off a table in Isaac Sim.

**Hard requirement: the policy may not be told where the cube is.** It has to find it in the image.

- The scripted expert solves IK against the cube's ground-truth pose — that happens **only during data
  generation**
- The policy receives images and joint states, nothing else — coordinates never enter the model

Standard privileged-expert → visuomotor-student. Success is defined in simulation: the cube is lifted
and held.

---

## Process topology

### The Python version split decides the whole file layout

- **Isaac Sim 5.1 is CPython 3.11** (`kit/python/bin/python3.11`; every extscache entry is `cp311`)
- **LeRobot ≥ 0.6 requires Python ≥ 3.12**
- **`rclpy` cannot be imported from Isaac's interpreter either** — ROS 2 Jazzy is 3.12

So torch, LeRobot and rclpy cannot be installed into Isaac's interpreter. **Multiple processes are
forced, not chosen.**

| Process | Python | Runs |
|---|---|---|
| Isaac (native on host) | 3.11 | Simulation. ActionGraph speaks ROS |
| `omx_vla_ctrl` container | 3.12 (Jazzy) | Expert, recorder, `jog`, `ik_target` — all rclpy |
| `omx_vla` container | 3.12 | torch + LeRobot: convert, train, eval |

The boundary between Isaac and the control layer is **ROS, not a hand-rolled RPC**. Isaac's ROS 2
bridge already uses Jazzy, so both sides speak the same messages with no bridging layer.

### ⚠️ Two containers, because MoveIt and LeRobot cannot coexist

LeRobot 0.6.x requires `numpy>=2.0,<2.3`. `moveit_py` is a compiled C++ binding, and numpy 2.0 broke
the C ABI. Measured 2026-08-07:

```
numpy 2.2.6    expert with ik:=moveit     -> Segmentation fault
numpy 1.26.4   same command               -> 20/20
numpy 2.2.6    same command, ik:=analytic -> fine
```

**It imports cleanly and segfaults at run time, with no message.** Do not use an import check to decide
whether MoveIt is available — that mistake has already been made once here. The line is drawn with two
images rather than trusting anyone to set `PYTHONPATH` correctly:

```
omx_vla_image      numpy 2.x    convert / train / eval / ik:=analytic
omx_bridge_image   numpy 1.x    ik:=moveit, jog, ik_target, monitor
```

Both containers may run at once, but **only one process may publish `/sync/command`**.

### Isaac runs no project code

Isaac's ActionGraph does four things, all stock nodes:

| Node | Purpose |
|---|---|
| `ROS2PublishJointState` / `ROS2SubscribeJointState` | `/joint_states` out, `/joint_command` in |
| `ROS2PublishTransformTree` | Object poses out over TF |
| `ROS2ServicePrim` | Read/write prim attributes from outside — **the only way to place the cube** |
| `IsaacCreateRenderProduct` + `ROS2CameraHelper` | Camera images |

Episode reset, randomization, the success test and the state machine are all Python in the container,
operating Isaac through those four. You open Isaac yourself, load the scene, press Play, then
`ros2 run` in the container — no standalone script.

⚠️ For headless rendering use `isaacsim.core.nodes.IsaacCreateRenderProduct`, **not**
`IsaacCreateViewport` / `IsaacGetViewportRenderProduct` — those need a viewport that does not exist
headless.

### Everything goes over ROS

Recording, the expert and keyboard control all publish `/sync/command` → `sync_node` →
`/joint_command`.

An earlier design recorded data by driving the articulation directly inside the Isaac process, for
timestamp alignment. It was replaced because:

- **Reset does not have to happen on the Isaac side.** `ROS2ServicePrim`'s `set_prim_attribute` can
  write `xformOp:translate` from outside and the cube really moves. The assumption that a pure-ROS path
  did not exist was simply wrong.
- **An in-process path needs rclpy to share the bridge**, and rclpy cannot enter Isaac's 3.11.
- **Timestamp alignment became recorded rather than guaranteed.** `state` and `action` come from the
  same control-loop iteration, so `action[t]` really is the command issued at `state[t]`. Images cannot
  do this — they arrive at their own rendering cadence — so every frame records *which* image was used
  and *how old it was*, and the converter decides what to do about stale frames. Pretending they were
  simultaneous is worse than recording the lag.

⚠️ An image's `header.stamp` is Isaac's **simulation** clock, which is not the control loop's clock.
Subtracting them yields numbers on the order of a billion seconds. Use wall-clock arrival time, which
is what `recorder.py` does.

---

## Kinematics

**Lula was never used and should not be revisited** — it lives in Isaac's interpreter while the control
code lives in a container. There are two interchangeable solvers:

| Solver | Where | Role |
|---|---|---|
| **MoveIt** (primary) | `omx_bridge_app.moveit_kin.MoveItKin` in the image | ROBOTIS' official config, KDL plugin |
| **Analytic** (control) | `OMXKinematics` in `src/omx_vla_app/ik.py` | Closed form, verified to 0.024 mm against Isaac |

Switch with `ros2 run data_collection expert --ros-args -p ik:=moveit|analytic|position_only`. Both
reach 20/20.

### The 5-DOF limitation, and why it does not matter here

`joint1` is base yaw; `joint2/3/4` are three **parallel** pitch axes forming a planar 3R chain inside
the vertical plane chosen by `joint1`; `joint5` is forearm roll.

- Gives 3 position DOF plus the approach pitch within that plane
- **Cannot give independent end-effector yaw** — the approach azimuth is always `atan2(y, x)`
- **But the limitation disappears for top-down grasps**: with the pitch at 90° pointing down, `joint5`'s
  roll axis becomes vertical, so it *becomes* gripper yaw. Vertical grasps at any `(x, y)` and any cube
  orientation are fully reachable.

**Keep the task top-down.** Asking for a pose the arm cannot reach returns `None` — or worse, a
plausible-looking wrong pose.

⚠️ `position_only_ik: True` is ROBOTIS' official setting, not a shortcut: a 5-DOF arm cannot hit an
arbitrary 6-DOF pose, so the solver asks only for position and lets orientation fall where it may. The
grasp pitch lands at 32–44° — it reaches in at an angle, and that works.

⚠️ **The null space has 2 DOF and no selection criterion**, so the answer is almost entirely determined
by the seed. Letting the solver continue from its own previous answer drifts, and drifts chaotically:
once, changing only the finger collider approximation — which cannot touch kinematics — moved the mean
grasp pitch from 44° to 7°.

⚠️ **Reset the IK seed to the measured joint values before `expert.reset()`.** The per-tick seed reset
lives *inside* the control loop while planning happens *outside* it. Without this line, planning uses
whatever pose the previous episode ended in, and failures arrive in blocks (runs of 3 and of 9 were
observed) because a failed episode never enters the loop, so the seed never updates.

**If reimplementing the analytic solver**: the `joint2 → joint3` offset vector is
`(0.0415, 0, 0.11315)`, which is **20.15° off vertical** — OpenManipulator-X's classic elbow offset.
Omitting that constant biases every solution. `ik.py` reads it from the URDF rather than hard-coding it.

**Self-collision**: MoveIt checks solutions before returning and perturbs the seed to re-solve on a hit
(8 attempts by default). Isaac's articulation has `enabledSelfCollisions = False`, so nothing else will
ever complain about the arm passing through itself.

---

## ⚠️ The control loop runs on simulation time

Isaac does **not** rate-limit under headless streaming — measured at 2.42× real time, drifting with GPU
load. `time.sleep(1/30)` therefore lets ~80 ms of simulation time elapse per tick; the seed trajectory
shifts, and position-only IK drifts out of its solvable range.

**11/20 with wall clock, 19/20 with simulation time.** Use `ExpertClient.wait_sim()`.

⚠️ All four kit rate-limit parameters (`rateLimitEnabled`, `rateLimitFrequency`,
`useFixedTimeStepping`, `useFastMode`) were tried and none work under headless streaming. That path is
closed.

---

## Scene, cameras and randomization

Everything below lives in `task/*.task.yaml`. **No magic numbers in code.**

### Randomization and success

- Cube: **2.5 cm, 15 g**, with a custom `PhysicsMaterial(static_friction=1.0, dynamic_friction=0.9)` —
  the default 0.2 is too slippery
- Spawn: `r ∈ U[0.16, 0.24]` m, `θ ∈ U[−50°, 50°]`, `yaw ∈ U[−45°, 45°]`
- Held-out bands: `[−35°, −20°]` and `[+20°, +35°]` — **inside** the range, not at the edges, and two of
  them (see [RESULTS.md](RESULTS.md#what-is-being-measured) for why)
- Success: lift `0.05` m, cube within `max_ee_distance 0.06` m of the gripper, `hold_steps 15`
  consecutive cycles. **Both conditions are needed** — distance rejects a cube knocked away, the hold
  rejects a bounce
- `spawn.py` is the only sampler, seeded via `np.random.default_rng(seed)`. **Do not use
  `omni.replicator`** — it is built for SDG capture loops and fights the control loop

Scene geometry: Z-up, `metersPerUnit = 1.0`, ground at z=0. End effector at home is
`(0.3129, −0.0016, 0.2107)`; shoulder at `(−0.011, 0, 0.0975)`; maximum reach ≈ **0.40 m**, so the
spawn annulus has ample margin. `physicsScene`: GPU dynamics **off**, TGS, 32 position iterations,
CCD on.

### Five cameras, chosen at training time

Four environment cameras, one per quadrant, geometrically symmetric — all 0.50 m from the workspace
centre, 0.304 m height difference, horizontal azimuth ±146.9° (rear) and ±33.1° (front) — plus a wrist
camera on `link5`. `rear_right` is the primary one, with intrinsics matched to a RealSense D455 so that
a future sim-to-real swap does not require retraining.

⚠️ **`ros.camera_topics` is the single source of truth for the camera list.** `build_scene.py`,
`recorder.py`, `convert.py` and `eval.py` all iterate it; none of them hard-codes a camera name. The
cost of violating that is documented in [RESULTS.md](RESULTS.md#five-cameras-sent-to-a-three-camera-model).

⚠️ **Rendering is 320×240 because of a crash, not for performance.** Rendering at 640×480 with five
cameras made Isaac crash roughly every 8 minutes:

```
assertion failed: 'm_owner == pthread_self()'
  in carb::thread::detail::BaseMutex<false>::unlock()

only two modules in the crash stack:
  omni.graph.action.RationalTimeSyncGate
  isaacsim.ros2.bridge.ROS2PublishImage
```

Five cameras publishing through one time-sync gate trips a threading error inside Isaac. That is
Isaac's own bug, not a scene-data problem (those present as parse errors or missing geometry, never a
mutex assertion). The only available mitigation is shrinking the contention window — 75 % fewer pixels
per publish. Training data is unaffected: the recorder was already downscaling to `record_resolution`.

Measured with all five: 71 Hz wall / **29.6 Hz simulation time** against a 30 Hz control loop, with
0.854–0.921 unique images per frame — adding three cameras did not make images staler.

### Procedural textures are load-bearing

A plain scene poses two independent problems for a vision-only policy. A plain floor offers no landmark
for localization, and **a plain cube is rotationally symmetric on screen** — so `yaw ±45°` randomization
does not exist in pixels at all. Two episodes differing only in yaw produce identical images mapped to
different expert actions, which is label noise no amount of data can resolve.

- The cube became a **Mesh** (procedural `UsdGeom.Cube` has no UVs), with a different asymmetric pattern
  on each of its six faces. Physics is unchanged: `boundingCube` is an *exact* approximation for a box,
  and grasp residuals measured `[-7.69, -0.34, 2.56]` versus `[-7.48, -0.53, 2.19]` before the change,
  10/10 either way.
- ⚠️ `subdivisionScheme` **must** be set to `none`, or the renderer applies Catmull-Clark and turns the
  cube into a sphere. **This happens only on the render side** — physics still behaves like a box, so
  every camera sees a sphere while the numbers look perfectly normal.
- The table is a **new visual quad**, not the floor. The real ground plane is 100 m across with UVs
  spanning only 0–1, so a texture on it would give the 0.5 m workspace about five pixels. Physics still
  uses the ground; the quad is raised 0.5 mm to avoid z-fighting.

---

## USD facts that cost time

### ⚠️ `omx_f.usd` must be sublayered, never `AddReference`d

The file is flattened: 16 `Flattened_Prototype_1..16` prim specs at the root, plus root-level
`/visuals`, `/colliders` and `/meshes`. `/omx_f/link6/collisions` internally references
`</Flattened_Prototype_5>`.

**Referencing the default prim `/omx_f` leaves all of those dangling and silently loses every visual
and collision mesh.** Use a sublayer:

```python
stage = Usd.Stage.CreateNew(out)
stage.GetRootLayer().subLayerPaths = ["<relative path>/omx_f.usd"]
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetDefaultPrim(stage.GetPrimAtPath("/omx_f"))
```

Verify by checking that the composed stage has the same prim count as the original (~235 under
traverse).

### ⚠️ The mimic joint stiffness — the only real blocker in building the scene

`/omx_f/joints/gripper_joint_2` has an **empty** `apiSchemas`: it has no drive at all and follows
`gripper_joint_1` purely through a `PhysxMimicJoint:rotZ` constraint. The factory
`naturalFrequency = 25.0` is far too soft at a 1/120 s step — under load the passive finger is pushed
open and lags badly, so **only one finger is actually gripping**.

**Fix: `naturalFrequency` 25 → 1000**, as an override in the task yaml. This is also more faithful: the
real arm's second finger is a rigid geared linkage, not a spring.

⚠️ **This failure mode is completely immune to tuning `gripper_joint_1`'s drive stiffness.** Raising it
from 625 to 20000 (32×) and reducing the cube from 15 g to 5 g changed the drop distance by not one
millimetre — the driven finger always had enough force. **If grip failures reappear, check whether the
passive finger is keeping up before touching the drive again.**

The mimic joint occupies a real articulation DOF, so `/joint_states` publishes **7** names, not 6.

### Gripper force is a parameter, not a risk

All drives are `type = acceleration`, stiffness 625, damping 40. With
`gripper_joint_1`'s `JointEquivalentInertia = 4.51e-5`, torque ≈ `0.028 × err` N·m; at a 0.3 rad
overshoot that is ~8.5 mN·m, giving a fingertip normal force of about **0.14 N** — enough for roughly
**25–30 g**. The cube is **15 g**, inside budget with ~2× margin. And this is simulation: cube mass,
friction, size and drive stiffness are all numbers you set.

### ⚠️ The one thing a lighter cube cannot fix

The finger colliders use `physics:approximation = convexHull`. Measured across both finger meshes, the
flatness ratio is only **3.7 : 2.1 : 1** (chunky parts, not thin plates) and only **3–6 %** of vertices
lie on the two outermost faces — so the convex hull does **not** preserve a flat gripping surface. Bad
contact geometry makes the cube slip or rotate out regardless of weight.

**Tested, and left on `convexHull`:**

- Both approximations over-approximate — a union of convex chunks necessarily contains the original
  mesh, so decomposition just relocates the fat
- Functionally equivalent: 20/20 under both solvers
- Decomposition dropped camera rate from 12.0 to 9.4 Hz and **broke visual diagnosis** (the chunks are
  fatter than the mesh and are not rendered), and looking at the scene was the most effective debugging
  channel available

If contact geometry ever does become the bottleneck, the next step is explicit flat pads, **not** a
different approximation.

### Do not edit `omx_f.usd` from this repository

It is part of the sim↔real calibration chain that `jog` / `to_sim` / `to_real` depend on. All
task-specific additions belong in the scene USD as overrides. To change the arm itself, change it in
`omx_bridge_image`, tag, and rebuild the image.

---

## Dataset format

Calibrated against LeRobot 0.6.1. Implementation: `ml/convert.py`.

```python
features = {
  "observation.state": {"dtype": "float32", "shape": (6,),
      "names": ["joint1","joint2","joint3","joint4","joint5","gripper_joint_1"]},
  "action":            {"dtype": "float32", "shape": (6,), "names": [...same...]},
  # the camera subset comes from --cameras; nothing is hard-coded
  "observation.images.front_left": {"dtype": "image", "shape": (3, 240, 320),
      "names": ["channels","height","width"]},
  ...
}
ds = LeRobotDataset.create(repo_id=..., fps=30, features=features, root=...,
                           robot_type="omx_f", use_videos=False)
for f in frames: ds.add_frame({**f, "task": "pick up the cube"})
ds.save_episode()
ds.finalize()      # required -- without it the parquet footer is never written
```

⚠️ Three things that fail in confusing ways:

- **`"image"`, not `"video"`.** With `use_videos=False`, `dataset_metadata.py` explicitly raises
  "features contain video keys but use_videos is False".
- **`shape` must be a `tuple`.** `validate_feature_numpy_array` compares `ndarray.shape != expected_shape`,
  and a list never compares equal — the error message then prints two shapes that look identical.
- **`finalize()` is not optional.**

Semantics that matter more than they look:

- **`observation.state` is the measured joint position**, not the commanded one. Storing commands
  teaches ACT the identity mapping: beautiful training loss, a policy that does nothing.
- **`action` is an absolute joint target**, not a delta. Absolute matches `/sync/command`'s semantics,
  which is why deploying a policy was a five-line change.
- **Only successful episodes enter the dataset.** Behavior cloning has no mechanism to exploit
  failures; failures stay in `data/raw/` for debugging.
- `fps = 30`; physics at 1/120, rendering and recording every 4th step.
- **`use_videos=False`.** 486 episodes × 3 cameras of 320×240 PNG = 20 GB. Storing PNG rather than MP4
  moves video decoding out of the dataloader, which at this scale is the real bottleneck — see
  [RESULTS.md](RESULTS.md#the-dataloader-decoded-300-pngs-per-sample-and-threw-them-away).

⚠️ **Normalization is not inside the policy.** A checkpoint ships `policy_preprocessor_*` /
`policy_postprocessor_*`, and the correct call sequence is `pre(obs) → policy.select_action → post(action)`,
built with `make_pre_post_processors(...)`. Calling `select_action` directly feeds unnormalized radians
and pixels and returns un-denormalized actions — **the arm still moves and nothing raises an error.**
The preprocessor includes `to_batch_processor` and `device_processor`, so observations must be given
**unbatched and on the CPU**.

### Raw dump first, dataset second

`recorder.py` writes `data/raw/ep_XXXXX/{frames.npz, img_*.png, meta.json}`; `convert.py` builds the
dataset from it. Two reasons, both of which have already paid off:

- **The raw PNGs are lossless.** If the dataset layer ever moves to MP4, that is lossy H.264 — only a
  lossless raw layer makes "changing format is a re-convert" actually true.
- **The scene has changed twice** (cameras added, cube geometry changed), invalidating recorded data
  both times. The raw dump separates "change how the dataset presents the data" from "change the
  scene"; only the latter requires re-recording.

LeRobot moved v2 → v2.1 → v3.0 within a year. Changing format is re-running `convert.py`, not
re-recording 500 episodes.

---

## Invariants

Violating any of these produces a failure that is hard to trace.

1. **Never edit `omx_f.usd` from this repo** — it is part of the sim↔real calibration chain.
2. **Never `AddReference` `omx_f.usd`** — sublayer it.
3. **Joint order has exactly one source**: `import sim_real_bridge.profile`. Do not copy the constants.
4. **`action` must be a valid `/sync/command` payload** — 6 joints, same order, radians, USD frame,
   absolute.
5. **`observation.state` is measured, not commanded.**
6. **Install torch alone from the cu128 index**, and never let a `requirements.txt` pin it. Blackwell is
   sm_120 and needs torch ≥ 2.7 + CUDA 12.8; a cu118/cu121 wheel fails later as
   `no kernel image is available`. ⚠️ Isaac's bundled `environment.yml` pins `cuda-toolkit=11.8` and
   must not be used to build an environment for torch.
7. **Do not upgrade Isaac Sim.** Pinned at `5.1.0-rc.19`. 6.0 is an Early Developer Release, its public
   import paths moved (every API verified here would need re-checking), and 5.1 is already proven on
   this Blackwell card.
8. **Only successful episodes enter the dataset.**
9. **The control loop uses simulation time, not wall clock.** 11/20 vs 19/20.
10. **Reset the IK seed to measured joint values before `expert.reset()`.**
11. **MoveIt and LeRobot cannot share a Python environment**, and an import check will not tell you.
12. **Do not locate the repo root by counting `.parent` levels.** `src/omx_vla_app` is mounted at two
    paths at once in the container, so a fixed count lands in the wrong one. Read `VLA_ROOT`.
13. **Mount one ROS package per compose line; never cover `/workspaces/src` wholesale.** The image was
    built with `colcon build --symlink-install`, so `install/` → `build/` → `src/`; covering the
    directory turns the image's own packages into dangling symlinks, and it presents as an ImportError
    that looks like a broken image.
