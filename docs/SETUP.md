# Setup and usage

## ⚠️ Read this first: what this repository is and is not

This repository holds the **task layer** — scene generation, the scripted expert, the dataset
pipeline, training and held-out evaluation. It does **not** hold the robot.

Everything that describes this particular arm — `omx_f.usd`, the URDF, the meshes, the joint-order
profile, the sync engine and the MoveIt configuration — lives inside a separate Docker image, and that
is deliberate. The alternative, a copy of those files per machine, was tried: it produces checkouts
that silently disagree with the image the containers are actually running, and the symptom is *"the
scene looks right but the numbers are wrong"*, which is the most expensive class of bug in this
project. There is one source of truth, and it is the image.

**Consequence for anyone outside the original lab: those images are in a private registry, so cloning
this repository alone will not give you a running system.** This document states precisely what the
image has to provide so that the gap is a known quantity rather than a surprise. Reference copies of
both Dockerfiles are in [`../docker/images/`](../docker/images/).

---

## The image chain

```
sim_real_bridge_image     robot-agnostic sync engine; ships no robot data
└── omx_bridge_image      this arm: profile, USD/URDF/meshes, MoveIt, operator tools   (numpy 1.x)
    └── omx_vla_image     + torch cu128, LeRobot, GR00T, TensorBoard                   (numpy 2.x)
```

`omx_vla_image` is built **on top of** the control image rather than on a plain CUDA base for one
reason: `eval.py` has to run a policy and publish `/sync/command` **in the same process**, so torch and
rclpy must share an interpreter.

### What the base image must provide

If you are rebuilding this stack against your own arm, this is the contract the task layer depends on.

| Provided at | What | Used by |
|---|---|---|
| ROS 2 package `sim_real_bridge` | The sync engine (`SyncNode`) and **`profile.py`** — the single source of truth for joint order | `eval.py`, `recorder.py`, `convert.py`, `task_config` |
| ROS 2 package `omx_bridge_app` | `jog`, `monitor`, `ik_target`, and `moveit_kin.MoveItKin` | `expert.py` (`ik:=moveit`), manual driving |
| ROS 2 package `isaac_ros2_messages` | Isaac's prim-service interface, **rebuilt from source for Jazzy** (Isaac ships it compiled for Humble) | Placing the cube via `ROS2ServicePrim` |
| `/profile/omx_f.profile.yaml` | Endpoint calibration: joint names, order, per-joint sign and offset | Everything that touches `/sync/command` |
| `/assets/omx_f/` | `omx_f.usd`, `omx_f.urdf`, meshes | `build_scene.py`, analytic IK, host-side Isaac |
| `/moveit/` | ROBOTIS' `moveit_config`, with mesh paths rewritten absolute | `MoveItKin` |
| Base environment | ROS 2 Jazzy, Python 3.12, **numpy 1.x** | MoveIt's C++ bindings |

The learning image adds, in this order (the order matters — see
[invariant 6](DESIGN.md#invariants)): numpy 2.x, **torch + torchvision from the cu128 index alone**,
`lerobot[dataset,training]`, `lerobot[groot]`, `tensorboard`.

`isaac/fetch_assets.sh` copies the USD, meshes and profile out of the image to `.image_cache/` so that
host-side Isaac can read them. It takes `OMX_BRIDGE_IMAGE` from the environment, so pointing it at your
own image is a one-variable change.

⚠️ Re-run `fetch_assets.sh` after pulling a new base image. The cache is an unversioned copy and will
not update itself.

---

## Hardware and software the results were measured on

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell, 96 GB (sm_120) |
| Driver | 580.159.03 |
| OS | Ubuntu 24.04.4, system Python 3.12.3 |
| Isaac Sim | **5.1.0-rc.19**, installed natively, headless + WebRTC streaming |
| torch | 2.11.0+cu128 |
| LeRobot | 0.6.1 |
| Real robot | **not involved at any point** |

⚠️ **Blackwell requires torch ≥ 2.7 with CUDA 12.8.** Many robotics `requirements.txt` files still pin
cu118/cu121; installing one fails much later as `no kernel image is available`. Install torch alone
from the cu128 index first, then everything else. Verify by **running a kernel**, not by printing a
version:

```python
import torch
print(torch.__version__, torch.cuda.get_device_capability())   # expect (12, 0)
a = torch.randn(4096, 4096, device="cuda"); (a @ a).sum().item()
```

GR00T fine-tuning at batch 24 with the vision tower unfrozen measured 2.03 B trainable parameters,
61.6 GB of VRAM and 1.18 s/step at 320×320.

---

## One-time setup

```bash
# 1. Point at your Isaac install and pick a DDS domain
#    docker/compose/.env holds ROS_DOMAIN_ID; isaac_python.sh reads ISAAC_SIM_PATH
export ISAAC_SIM_PATH=/path/to/isaac_sim_5.1

# 2. Pull the robot's data out of the image for host-side Isaac
bash isaac/fetch_assets.sh                      # -> .image_cache/

# 3. Author the scene (does not start Isaac; authors USD directly, takes seconds)
bash isaac/isaac_python.sh isaac/build_scene.py --force     # -> assets/pick_cube.usd
```

⚠️ `ROS_DOMAIN_ID` must match **in Isaac, in both containers, and in `.env`**. A mismatch means neither
side discovers the other, and it looks exactly like a broken scene.

Then start Isaac natively, open `assets/pick_cube.usd`, and press **Play**. Isaac runs no project code;
everything else happens in the containers.

---

## The pipeline

### 1. Collect demonstrations — control container

```bash
bash docker/scripts/ctrl.sh                     # omx_bridge_image: numpy 1.x + MoveIt

ros2 run data_collection expert                                      # 5 episodes, writes nothing
ros2 run data_collection expert --ros-args -p episodes:=20 -p holdout:=true
ros2 run data_collection record --ros-args -p episodes:=500          # -> data/raw/
ros2 run omx_bridge_app jog sim                                      # drive it by hand
```

Parameters: `episodes`, `holdout`, `seed`, `ik` (`moveit` | `analytic` | `position_only`), `theta_deg`,
`radius`, `save_frames`, `out`. Both solvers reach 20/20.

⚠️ Only one process may publish `/sync/command` at a time. Before starting, check both containers:

```bash
docker exec omx_vla      pgrep -f 'data_collection|omx_vla'
docker exec omx_vla_ctrl pgrep -f 'data_collection|omx_vla'
```

### 2–5. Convert, audit, train, evaluate — learning container

```bash
bash docker/scripts/vla.sh                      # omx_vla_image: numpy 2.x + torch + LeRobot

# 2. raw dump -> LeRobotDataset v3.0
python3 ml/convert.py --raw data/raw --out data/lerobot_3cam \
    --cameras front_left,front_right,wrist --repo-id <id>
#   --resize N       both models see identical pixels
#   --pad-square     letterbox to square; required for a fair GR00T comparison
#   --force          rebuild an existing output directory

# 3. check the dataset before spending GPU hours on it
python3 ml/audit.py --raw data/raw_3obj --dataset data/lerobot_3obj

# 4. train  (parameters live in ml/train.yaml, with the reasoning beside each number)
python3 ml/train.py --profile=act_3obj --tensorboard
python3 ml/train.py --steps=100                 # one-off override; the CLI always wins

# 5. evaluate on held-out poses
python3 ml/eval.py --checkpoint outputs/act_v1/checkpoints/best/pretrained_model \
    --episodes 20 --holdout --seed 0
python3 ml/eval.py --checkpoint ... --record data/eval_rec --frame-every 10   # save rollout frames

# sweep every checkpoint of a run: 3 rounds x 20 episodes, new seed per round
bash ml/eval_sweep.sh outputs/groot_v1
EPISODES=30 SEEDS="0 1 2 3" bash ml/eval_sweep.sh outputs/act_v1
```

⚠️ **Use `ml/train.py`, not the bare `lerobot-train` CLI, for this dataset.** It applies the
chunked-action dataloader fix; without it the GPU idles at 96 % and a 60 k-step run takes about seven
days instead of about three hours. No error is reported either way. Details:
[RESULTS.md](RESULTS.md#the-dataloader-decoded-300-pngs-per-sample-and-threw-them-away).

---

## Configuration

**`ml/train.yaml`** holds every training profile. `active:` selects the default; `--profile=` overrides
it. Keys are LeRobot flag names (`policy.tune_visual` → `--policy.tune_visual`), and each number carries
the reasoning for its value — including, where it applies, a note that the reasoning was later
overturned.

Profiles present: `act_v1`, `groot_v1`, `groot_v2`, `groot_v3` (single-object, all measured),
`act_3obj`, `groot_3obj`, `groot_3obj_lr4` (three-object, unfinished).

**`task/*.task.yaml`** holds the scene, randomization, camera list, success criteria and the USD
overrides. **No magic numbers in code.** Two specs are kept:

- `pick_cube.task.yaml` — three objects, language-conditioned (current)
- `pick_cube_1obj.task.yaml` — **frozen**, so the four measured models stay reproducible

⚠️ Switching tasks requires switching **both** the scene USD and the task yaml — the camera list and
resolution are read from the yaml, not from the scene. Mismatching them fails silently: one such
mismatch scored 0/480 across eight checkpoints and looked exactly like a bad model. Image size is the
exception; `eval.py` reads that from the checkpoint's `input_features`.

Re-measuring the four single-object models:
[RESULTS.md](RESULTS.md#reproducing-these-four-measurements).

---

## Storage

Not small. On the original machine:

| | |
|---|---|
| `data/raw/` (486 eps, 5 cameras, lossless PNG) | 37 GB |
| `data/raw_3obj/` (1 868 eps) | 136 GB |
| `data/lerobot_3cam/` (486 eps / 100 096 frames) | 20 GB |
| `data/lerobot_3obj/` (1 868 eps / 384 147 frames) | 55 GB |
| One GR00T checkpoint | 24–27 GB |
| One ACT checkpoint | 591 MB |

`data/`, `outputs/`, `assets/*.usd`, `assets/textures/`, `.image_cache/` and `.hf_cache/` are all
gitignored — every one of them is a reproducible product. GR00T's weights land in `.hf_cache/` via
`HF_HOME` so that the download survives `docker compose down`.
