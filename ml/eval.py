"""Drive the twin from a trained checkpoint -- the last seam of M3.

Run it in the **omx_vla** container:

    bash docker/scripts/vla.sh
    python3 ml/eval.py --checkpoint outputs/act_m3/checkpoints/last/pretrained_model

That container is ``omx_vla_image`` -- ``omx_bridge_image`` plus torch and
lerobot -- and it exists precisely for this file: the policy needs torch,
``/sync/command`` needs rclpy, and they have to live in the **same process**.

⚠️ Not the control container. Its numpy is 1.x because MoveIt needs it, and
lerobot requires 2.x. That split is why there are two compose files.

⚠️ This uses the **analytic** kinematics, not MoveIt -- only for the forward
kinematics the success test needs. That is what keeps inference on the numpy 2.x
side of the split.

**This is ``data_collection.expert.run_episode`` with a policy in the expert's
place.**
The reset, the cube placement, the settle behaviour and the success test are
the same code reached through the same seam, because an evaluation that
measures something other than what was demonstrated measures nothing. The
policy sees only images and joint states -- the cube's coordinates are used to
place it and to score it, never to drive it.

⚠️ **A 200-step checkpoint is supposed to fail.** M3's acceptance is that the
arm moves *because of* a checkpoint: that every seam carries a tensor of the
right shape from the dataset through the trainer to the wire. Success at the
task is M5's problem.
"""
from __future__ import annotations

import argparse
import sys
import threading

import numpy as np
import rclpy
import torch
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from sim_real_bridge.profile import load_profile
from sim_real_bridge.sync_node import SyncNode

from data_collection.expert import ExpertClient
from omx_bridge_app import ros_args
from omx_vla_app import task_config
from omx_vla_app.ik import OMXKinematics
from omx_vla_app.spawn import sample_cube_pose


def observation(client, cfg):
    """The policy's input: images and joint state, and nothing else.

    ⚠️ The cube's pose is deliberately absent. The whole point of the project is
    that the policy finds the cube in the image; feeding it coordinates here
    would produce a number that looks like success and means nothing.

    Returned **unbatched and on CPU**. lerobot 0.6.1 runs observations through a
    processor pipeline before the policy sees them, and that pipeline already
    contains ``to_batch_processor`` and ``device_processor``. Adding a batch
    dimension or calling ``.to(device)`` here would double up on both.
    """
    q = client.joints()
    frames = client.frames()
    obs = {"observation.state": torch.from_numpy(q.astype(np.float32))}
    for name, got in frames.items():
        if got is None:
            return None                      # no image yet -- caller waits
        msg, _ = got
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            arr = arr.reshape(msg.height, msg.width, -1)[:, :, :3]
        except ValueError:
            return None                      # partial frame
        want = tuple(cfg["cameras"]["record_resolution"])
        if (arr.shape[1], arr.shape[0]) != want:
            from PIL import Image as PILImage
            arr = np.asarray(PILImage.fromarray(arr).resize(
                want, PILImage.BILINEAR))
        # CHW float in [0,1] -- what LeRobotDataset hands the trainer, so the
        # normalizer's MEAN_STD statistics are in those units. Feeding uint8
        # here is silent train/serve skew: it runs, the numbers are wrong.
        obs[f"observation.images.{name}"] = \
            torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return obs


def run_episode(client, policy, pre, post, kin, cfg, pos, yaw, max_steps):
    """Reset, let the policy drive, score it the way the expert is scored."""
    grip, succ = cfg["gripper"], cfg["success"]
    grasp_z = cfg["cube"]["size"] / 2
    period = 1.0 / cfg["timing"]["fps"]

    client.go_home(grip["open"])
    client.wait_for_release()
    if not client.place_cube(pos, yaw):
        return False, 0, "方塊沒有到位"

    policy.reset()                            # clears the action chunk queue
    held = 0
    for step in range(max_steps):
        obs = observation(client, cfg)
        if obs is None:
            client.wait_sim(period)
            continue
        # ⚠️ The pipeline is not optional. In lerobot 0.6.1 normalization lives
        # in these processors, not inside the policy, so calling select_action
        # directly feeds raw radians and raw pixels into a network trained on
        # standardised ones. The arm still moves -- that is what makes it worth
        # a warning rather than a crash.
        with torch.no_grad():
            action = post(policy.select_action(pre(obs)))
        # Absolute joint targets in canonical order -- the same payload the
        # expert publishes, which is why M7 is a one-line change and not a
        # milestone.
        client.send(np.asarray(action.squeeze(0).cpu(), dtype=float))
        client.wait_sim(period)

        cube = client.cube()
        if cube is None:
            continue
        tool = kin.fk(client.joints()[:5])[0]
        if (cube[0][2] > grasp_z + succ["lift_height"]
                and np.linalg.norm(cube[0] - tool) < succ["max_ee_distance"]):
            held += 1
            if held >= succ["hold_steps"]:
                return True, step, "成功"
        else:
            held = 0
    return False, max_steps, "逾時"


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    ap = argparse.ArgumentParser(description="用訓練好的 checkpoint 驅動手臂")
    ap.add_argument("--checkpoint", required=True, help="pretrained_model 目錄")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", action="store_true", help="只在保留區取樣")
    ap.add_argument("--device", default="cuda")
    known, rest = ap.parse_known_args(argv[1:])

    # ⚠️ targets:=sim is not a default, it is a guard. This machine has no real
    # arm; the expert hard-codes the same thing for the same reason.
    rclpy.init(args=ros_args([argv[0]] + rest, "profile",
                             extra=["-p", "mode:=command", "-p", "targets:=sim"]))
    engine = SyncNode()
    profile_path = engine.get_parameter("profile").get_parameter_value().string_value
    cfg = task_config.load()
    client = ExpertClient(load_profile(profile_path), cfg)
    kin = OMXKinematics(cfg)

    if known.device == "cuda" and not torch.cuda.is_available():
        # The container needs the GPU reservation in docker-compose-vla.yml.
        # Without it lerobot quietly falls back to CPU and the failure surfaces
        # much later as a driver error from .to("cuda").
        print("[eval] ⚠️ 容器裡看不到 GPU。docker-compose-vla.yml 要有 "
              "deploy.resources.reservations.devices,改完 down/up 一次。"
              "\n       先用 CPU 跑(會慢,推論可能追不上模擬)。")
        known.device = "cpu"

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    print(f"[eval] 載入 {known.checkpoint}  (device={known.device})")
    policy = ACTPolicy.from_pretrained(known.checkpoint).to(known.device).eval()
    # Rebuilt from the checkpoint, not reconstructed by hand: the normalizer
    # carries the dataset's own mean/std, and a policy fed with anyone else's
    # statistics produces plausible-looking nonsense.
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=known.checkpoint,
        preprocessor_overrides={"device_processor": {"device": known.device}},
    )

    executor = SingleThreadedExecutor()
    executor.add_node(engine)
    executor.add_node(client)

    def spin_quietly():
        try:
            executor.spin()
        except (ExternalShutdownException, rclpy.executors.ShutdownException):
            pass

    spin = threading.Thread(target=spin_quietly, daemon=True)
    spin.start()

    rc = 1
    try:
        if not (client.wait_for_joints() and client.wait_for_cube()
                and client.prim.wait_for_isaac()):
            raise SystemExit(1)
        rng = np.random.default_rng(known.seed)
        wins = 0
        for i in range(known.episodes):
            pos, yaw, r, th = sample_cube_pose(cfg, rng, known.holdout)
            ok, steps, why = run_episode(
                client, policy, pre, post, kin, cfg, pos, yaw,
                cfg["timing"]["max_episode_steps"])
            wins += ok
            print(f"  第 {i + 1:3d} 集  r={r * 1000:.0f}mm θ={th:+.0f}° "
                  f"{'✅' if ok else '✗ '} {why} ({steps} 步)", flush=True)
        print(f"\n成功 {wins}/{known.episodes}")
        if client.loop_overruns:
            print(f"⚠️ 推論趕不上模擬的 tick 數:{client.loop_overruns}"
                  " —— 策略每一步前進得比一個控制週期多")
        rc = 0
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        executor.shutdown()
        engine.destroy_node()
        client.destroy_node()
        spin.join(timeout=2)
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
