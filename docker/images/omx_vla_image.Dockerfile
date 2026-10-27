FROM registry.screamtrumpet.csie.ncku.edu.tw/pochun/omx_bridge_image:latest

# Declared so CI's `docker build --build-arg THREADS=$(nproc)` does not warn.
# Nothing here compiles; every install below is a wheel.
ARG THREADS=4

SHELL ["/bin/bash", "-c"]

# This image is the learning layer, and it is deliberately the **only** place
# torch exists. The chain is:
#
#   sim_real_bridge_image   the robot-agnostic sync engine
#   └── omx_bridge_image    this arm's data: profile, assets, MoveIt, jog
#       └── omx_vla_image   training / fine-tuning / inference   ← here
#
# It inherits ROS 2 Jazzy and rclpy from below, which is the reason it is built
# on top of omx_bridge_image rather than on a plain CUDA base: `eval.py` has to
# run a policy and publish `/sync/command` **in one process**, so torch and
# rclpy must share an interpreter. Training does not need ROS, but carrying it
# costs one shared layer and saves maintaining a second image.

# ── the apt-owned packages lerobot wants to replace ────────────────────────
# ROS's apt dependencies put these in /usr/lib/python3/dist-packages, and a
# **debian-installed** distribution has no RECORD file, so pip cannot uninstall
# it: `error: uninstall-no-record-file`. Left to itself the lerobot install dies
# on the first one it meets.
#
# `--ignore-installed` sidesteps it by writing to
# /usr/local/lib/python3.12/dist-packages, which precedes the distro path on
# sys.path -- the new copies are the ones that get imported, the apt ones stay
# on disk untouched for apt's sake.
#
# ⚠️ Scoped to **this list** rather than applied to the lerobot install as a
# whole. A blanket `--ignore-installed` would reinstall every dependency
# including torch, and torch would come from the default index without cu128 --
# exactly the failure invariant 6 exists to prevent.
#
# ⚠️⚠️ **MoveIt does not work in this image, and cannot be made to.**
#
# numpy 2.0 broke the C ABI and `moveit_py` is a compiled binding, so raising
# numpy to satisfy lerobot breaks MoveIt. Measured on 2026-08-07:
#
#   numpy 2.2.6   `ros2 run omx_vla_app expert` (ik:=moveit)  -> Segmentation fault
#   numpy 1.26.4  same command                                 -> 20/20
#   numpy 2.2.6   same command with ik:=analytic               -> fine
#
# ⚠️ It **imports cleanly and segfaults at run time**, with no message. Do not
# trust an import check here -- an earlier version of this file did, concluded
# MoveIt was fine, and was wrong.
#
# This is a binary-compatibility fact, not a configuration problem: the only
# real fix is rebuilding moveit against numpy 2.x, which is out of scope. The
# line is therefore drawn between two images rather than left to whoever
# remembers to set PYTHONPATH:
#
#   omx_vla_image      numpy 2.x   convert / train / eval / ik:=analytic
#   omx_bridge_image   numpy 1.x   ik:=moveit, jog, ik_target, monitor
#
# `eval.py` uses the analytic solver, so inference is unaffected. The compose
# files for both live in the vla repo (docker-compose-vla.yml and
# docker-compose-ctrl.yml).
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed \
        --root-user-action=ignore \
        "numpy>=2.0.0,<2.3.0" packaging pillow setuptools pyyaml psutil

# ── torch, from the cu128 index, split across several layers ───────────────
# ⚠️ **The split is not cosmetic -- a single-layer install cannot be pushed.**
# Installing torch in one RUN produces one 7.1 GB layer (~3.5 GB compressed),
# and pushing a blob that large to the GitLab registry outlives the auth token:
# every smaller layer uploaded fine, that one retried five times and then failed
# with `unauthorized: authentication required`. Splitting keeps the largest blob
# near 1.7 GB uncompressed.
#
# The CUDA runtime packages are installed first, **pinned and with --no-deps**,
# so the final `torch` install finds its dependencies already satisfied and only
# adds torch itself. Grouped by size, largest first, so no group dominates.
#
# ⚠️ These pins must match what the torch build actually wants. If they drift,
# nothing breaks -- pip simply reinstalls the right versions in the torch layer,
# and that layer becomes large again. To refresh the list after a torch upgrade:
#
#     docker run --rm <image> pip3 list --format=freeze | grep -E '^(nvidia|triton)'
#
# ⚠️ Invariant 6 still holds: torch's own version is **not** pinned anywhere.
# What is pinned here is the CUDA runtime it depends on, which is a different
# thing -- these are the packages torch would have pulled anyway.
ARG PIP_ARGS="--no-cache-dir --break-system-packages --root-user-action=ignore --index-url https://download.pytorch.org/whl/cu128"

RUN pip3 install ${PIP_ARGS} --no-deps nvidia-cudnn-cu12==9.19.0.56
RUN pip3 install ${PIP_ARGS} --no-deps nvidia-cublas-cu12==12.8.4.1
RUN pip3 install ${PIP_ARGS} --no-deps \
        nvidia-cusparselt-cu12==0.7.1 \
        nvidia-cusolver-cu12==11.7.3.90 \
        nvidia-nccl-cu12==2.28.9
RUN pip3 install ${PIP_ARGS} --no-deps \
        nvidia-cusparse-cu12==12.5.8.93 \
        nvidia-cufft-cu12==11.3.3.83 \
        nvidia-cuda-nvrtc-cu12==12.8.93 \
        nvidia-nvshmem-cu12==3.4.5 \
        nvidia-curand-cu12==10.3.9.90 \
        nvidia-nvjitlink-cu12==12.8.93 \
        nvidia-cuda-cupti-cu12==12.8.90 \
        nvidia-cuda-runtime-cu12==12.8.90 \
        nvidia-cufile-cu12==1.13.1.3 \
        nvidia-nvtx-cu12==12.8.90
RUN pip3 install ${PIP_ARGS} --no-deps triton==3.6.0

# torch itself. Its CUDA dependencies are in place above, so this layer holds
# torch + torchvision and the handful of pure-Python packages they need.
RUN pip3 install ${PIP_ARGS} torch torchvision

# ── the training stack ─────────────────────────────────────────────────────
# `dataset` brings the LeRobotDataset writer/reader, `training` brings
# accelerate. Both are optional extras in lerobot 0.6.x: a bare
# `pip install lerobot` imports fine and then fails at the first dataset or
# training call, which is a confusing place to discover a missing dependency.
#
# Deliberately **after** torch: lerobot depends on torch, and letting it resolve
# that dependency itself would pull a default-index build without cu128.
RUN pip3 install --no-cache-dir --break-system-packages \
        --root-user-action=ignore \
        "lerobot[dataset,training]"

# ── GR00T ──────────────────────────────────────────────────────────────────
# M6 fine-tunes nvidia/GR00T-N1.7-3B, which lerobot 0.6.1 exposes as an ordinary
# policy (`--policy.type=groot`). The extra pulls transformers (the Qwen3-VL
# backbone), peft, diffusers, timm, dm-tree and decord.
#
# ⚠️ Verified not to disturb the pinned stack before adding it here: installed
# into a throwaway container from this image, torch stayed 2.11.0+cu128 and
# numpy stayed 2.2.6. Nothing in lerobot's metadata pins torch, and the version
# already present satisfies the transitive `torch>=...` constraints, so pip
# leaves it alone. **Re-check that after any lerobot bump** -- a default-index
# torch here is exactly what invariant 6 exists to prevent, and it fails much
# later as `no kernel image is available`.
#
# Its own layer, after the training stack: it is the largest optional addition
# and keeping it separate means a GR00T change does not rebuild torch.
RUN pip3 install --no-cache-dir --break-system-packages \
        --root-user-action=ignore \
        "lerobot[groot]"

# ── metric logging ─────────────────────────────────────────────────────────
# lerobot 0.6.1 logs to W&B and nowhere else. This project runs on a shared
# machine with no W&B account, so `vla/ml/train.py --tensorboard` substitutes a
# TensorBoard writer at the same seam (see vla/ml/tb_logger.py). That needs the
# `tensorboard` package for torch.utils.tensorboard.SummaryWriter; without it
# the flag exits with an explanation rather than training un-logged.
#
# Its own layer: it is small, it changes far less often than lerobot, and
# keeping it out of the line above avoids re-resolving the training stack.
RUN pip3 install --no-cache-dir --break-system-packages \
        --root-user-action=ignore \
        tensorboard

# ── build-time guard ───────────────────────────────────────────────────────
# Catches the one mistake that is otherwise invisible until a training run dies
# on a Blackwell card: something in the resolution above quietly replacing the
# cu128 build with a default-index one.
#
# ⚠️ A CUDA **kernel** cannot be run here (the builder has no GPU), so this only
# checks the build tag. `docker run --gpus all ...` in the README is the real
# check -- printing a version proves nothing about whether kernels launch.
#
# ⚠️ MoveIt is deliberately **not** checked. It imports fine and segfaults at
# run time under numpy 2.x (see the top of this file), so an import assertion
# here would produce false confidence, which is exactly what happened once.
# What is checked is the part inference actually needs: rclpy, the message
# types and the profile.
RUN python3 -c "\
import torch, numpy, lerobot, transformers; \
from torch.utils.tensorboard import SummaryWriter; \
from lerobot.policies import get_policy_class; \
assert torch.__version__.endswith('+cu128'), 'torch is not the cu128 build: ' + torch.__version__; \
assert numpy.__version__.startswith('2.'), 'numpy was downgraded: ' + numpy.__version__; \
get_policy_class('act'); get_policy_class('groot'); \
print('torch', torch.__version__, '| numpy', numpy.__version__, '| lerobot', lerobot.__version__, '| transformers', transformers.__version__)" && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source ${ROS2_WS}/install/setup.bash && \
    python3 -c "\
import rclpy; from sensor_msgs.msg import JointState; from tf2_msgs.msg import TFMessage; \
from sim_real_bridge.profile import load_profile; \
load_profile('/profile/omx_f.profile.yaml'); \
print('inference-side ROS layer OK under numpy 2.x (MoveIt intentionally not checked)')"

WORKDIR ${ROS2_WS}
ENTRYPOINT ["/ros_entrypoint.bash"]
CMD ["bash", "-l"]
