FROM registry.screamtrumpet.csie.ncku.edu.tw/pochun/sim_real_bridge_image:latest

# Consumed by CI's `docker build --build-arg THREADS=$(nproc)`; declared so the
# arg is used (no build warning), even though colcon build here is a single pkg.
ARG THREADS=4

SHELL ["/bin/bash", "-c"]

# ── The OMX follower's endpoint calibration ────────────────────────────────
# sim_real_bridge_image is the invariant core and deliberately ships no robot
# data. This is that data: which joints exist, in what order, and the sign /
# offset that turns each endpoint's frame into the canonical one. Baked at
# /profile so every compose `command` can pass the same path.
COPY profile /profile

# ── The simulation model ───────────────────────────────────────────────────
# Carried so the image is self-describing -- one artifact holds everything that
# is specific to this arm. Isaac itself runs NATIVELY, not in this container, so
# it reads these from a checkout of this repo rather than from here. Pull them
# out with:  docker run --rm -v "$PWD:/out" <image> cp -r /assets /out
COPY assets /assets

# ── MoveIt, for IK against this arm's official configuration ───────────────
# Only the pieces an in-process solver needs, not the `moveit` meta package:
# 242 MB rather than a gigabyte, and nothing here runs move_group.
#
# ROBOTIS ships a moveit_config for omx_f; /moveit is that config, verbatim.
# Its `position_only_ik: True` is not a shortcut -- this arm is 5-DOF and cannot
# hit an arbitrary 6-DOF pose, so the official solver asks only for position.
RUN apt update && apt install -y \
        ros-${ROS_DISTRO}-moveit-py \
        ros-${ROS_DISTRO}-moveit-kinematics \
        ros-${ROS_DISTRO}-moveit-ros-planning \
        ros-${ROS_DISTRO}-moveit-planners-ompl && \
    rm -rf /var/lib/apt/lists/*

COPY moveit /moveit

# MoveIt needs the meshes to load, and it resolves `filename` as a URL: the
# URDF's relative `meshes/omx_f/*.stl` becomes a hostname lookup that fails, and
# the robot silently ends up with **no collision geometry at all** -- which means
# self-collision checking passes everything.
#
# Rewritten here rather than in assets/: an absolute path is only correct for one
# filesystem, and the original has to keep working from a host checkout, where
# Isaac and sim/ik.py read it.
RUN sed -E 's#filename="meshes/#filename="file:///assets/omx_f/meshes/#g' \
        /assets/omx_f/omx_f.urdf > /moveit/omx_f.urdf && \
    grep -c 'file:///assets' /moveit/omx_f.urdf

# ── The operator tools, and the interfaces one of them needs ───────────────
# omx_bridge_app: jog and monitor (keyboard TUI, deviation compare), and
# ik_target, which drives a target prim in Isaac. Built against the core's public
# interface only -- they publish /sync/command and read /sync/state, and never
# reach into sim_real_bridge's internals.
#
# isaac_ros2_messages is Isaac's prim-service interface, rebuilt from source
# rather than taken from the Isaac install: Isaac ships it compiled for Humble
# and this is Jazzy, so the binary is unusable while the .srv files are not.
COPY src ${ROS2_WS}/src
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source ${ROS2_WS}/install/setup.bash && \
    cd ${ROS2_WS} && \
    colcon build --symlink-install \
        --packages-select isaac_ros2_messages omx_bridge_app

WORKDIR ${ROS2_WS}
ENTRYPOINT ["/ros_entrypoint.bash"]
CMD ["bash", "-l"]
