"""``data/raw/`` -> a LeRobotDataset. Runs in the **omx_vla** container.

    bash docker/scripts/vla.sh                  # omx_vla_image: torch + lerobot
    python3 ml/convert.py                       # data/raw -> data/lerobot
    python3 ml/convert.py --raw data/raw_other --force

Not Isaac's 3.11 (no lerobot there) and not the control container (its numpy is
pinned to 1.x by MoveIt -- see docker/compose/docker-compose-ctrl.yml).

The raw dump is deliberately format-free (see ``recorder.py``): LeRobot moved
v2 -> v2.1 -> v3.0 inside a year and GR00T wants a different variant again, so a
format change is a re-run of this file rather than a re-recording of 200
episodes. That is the whole reason this step exists as a separate program.

Three things here are load-bearing and were verified against lerobot 0.6.1
rather than assumed:

* **``dtype`` is ``"image"``, not ``"video"``.** ``dataset_metadata.py`` raises
  outright if any feature is ``"video"`` while ``use_videos=False``, and PNGs are
  what this dataset wants -- 200 episodes x ~150 frames x 2 cameras at 320x240
  keeps video decode out of the dataloader, which at this scale is the real
  bottleneck.
* **``shape`` must be a tuple.** ``validate_feature_numpy_array`` compares it to
  ``ndarray.shape`` with ``!=``, so a list never matches and every frame is
  rejected with a shape error that names two identical-looking shapes.
* **Images may be handed over channel-last.** ``validate_feature_image_or_video``
  accepts ``(H, W, C)`` as well as ``(C, H, W)``, so the PNG goes in as read and
  no transpose is needed.

**The joint order is not written down here.** It comes from
``sim_real_bridge.profile`` via ``task_config``, the same single source the
recorder and the ROS nodes use, and every episode's ``meta.json`` is checked
against it. A dump recorded under a different profile is a hard error, not a
warning: silently reordering a column produces a dataset that trains fine and
means nothing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sim"))

import task_config  # noqa: E402

TASK_STRING = "pick up the cube"


class ConvertError(RuntimeError):
    """Something about the raw dump does not match the config -- say which."""


def episode_dirs(raw_root: Path):
    eps = sorted(p for p in raw_root.glob("ep_*") if (p / "frames.npz").exists())
    if not eps:
        raise ConvertError(
            f"{raw_root} 裡沒有任何含 frames.npz 的 ep_* 目錄。\n"
            "先錄資料:ros2 run omx_vla_app record --ros-args -p episodes:=N")
    return eps


def load_episode(ep_dir: Path, joints, cameras):
    """One episode as ``(rows, meta)``, images already expanded per frame.

    Returns ``rows`` as a list of dicts ready for ``add_frame``.

    Images are stored once per unique camera timestamp and referenced by index,
    so expanding them is this function's main job. The cache matters: at 30 Hz
    against a slower camera the same PNG is referenced by several consecutive
    ticks, and decoding it once per tick is the difference between a minute and
    ten.
    """
    meta = json.loads((ep_dir / "meta.json").read_text())
    npz = np.load(ep_dir / "frames.npz")

    got = list(meta.get("joints", []))
    if got != list(joints):
        raise ConvertError(
            f"{ep_dir.name} 的關節順序與現在的 profile 不一致。\n"
            f"  dump: {got}\n  現在: {list(joints)}\n"
            "這份 raw 是用不同的 profile 錄的 —— 重錄,或 checkout 回當時的 tag。")

    # Same idea for cameras: the recorder writes one entry per camera it saw, so
    # a dump made before a camera was added (or renamed) has a different set.
    # Caught here rather than as a KeyError three lines down, because the fix is
    # "re-record" and the error should say so.
    got_cams = set(meta.get("images", {}))
    if got_cams and got_cams != set(cameras):
        raise ConvertError(
            f"{ep_dir.name} 的相機組合與現在的設定不一致。\n"
            f"  dump: {sorted(got_cams)}\n  現在: {sorted(cameras)}\n"
            "相機是在錄製當下決定的 —— 加了或改名之後,舊的 raw 不能用,要重錄。")

    state, action = npz["state"], npz["action"]
    n = len(state)
    if action.shape != state.shape:
        raise ConvertError(f"{ep_dir.name}: state {state.shape} 與 action "
                           f"{action.shape} 形狀不符")
    if state.shape[1] != len(joints):
        raise ConvertError(f"{ep_dir.name}: state 有 {state.shape[1]} 欄,"
                           f"profile 有 {len(joints)} 個關節")

    # Where every camera has an image. The subscription is live before the
    # episode starts, so in practice this is frame 0, but the first ticks of the
    # very first episode can land before anything has arrived.
    idx = {c: npz[f"{c}_idx"] for c in cameras}
    age = {c: npz[f"{c}_age"] for c in cameras}
    have = np.all([idx[c] >= 0 for c in cameras], axis=0)
    if not have.any():
        raise ConvertError(f"{ep_dir.name}: 沒有任何一幀是每台相機都有影像的")
    start = int(np.argmax(have))
    if not have[start:].all():
        # A gap in the middle is not a startup artifact -- a camera stopped
        # publishing mid-episode. Dropping those frames would leave a hole in a
        # trajectory that is about to be learned as a continuous one.
        missing = int((~have[start:]).sum())
        raise ConvertError(
            f"{ep_dir.name}: 第 {start} 幀之後還有 {missing} 幀缺影像 —— "
            "不是啟動殘留,是相機中途停過。這集不能用。")

    cache: dict[tuple[str, int], np.ndarray] = {}

    def image(cam, i):
        key = (cam, int(i))
        if key not in cache:
            path = ep_dir / f"img_{cam}_{int(i):05d}.png"
            if not path.exists():
                raise ConvertError(f"{ep_dir.name}: 缺 {path.name}"
                                   "(frames.npz 指到不存在的影像)")
            cache[key] = np.asarray(Image.open(path).convert("RGB"))
        return cache[key]

    rows = []
    for t in range(start, n):
        row = {
            # ⚠️ state is the **measured** joint position and action is the
            # command issued at that same tick. Storing the command as state
            # would teach an identity map: high training scores, no behaviour.
            "observation.state": state[t].astype(np.float32),
            "action": action[t].astype(np.float32),
            "task": TASK_STRING,
        }
        for cam in cameras:
            row[f"observation.images.{cam}"] = image(cam, idx[cam][t])
        rows.append(row)

    ages = np.concatenate([age[c][start:] for c in cameras])
    return rows, meta, ages


def build_features(joints, cameras, resolution):
    w, h = resolution
    names = list(joints)
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),),
                              "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            # "image" not "video" -- see the module docstring.
            "dtype": "image",
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }
    return features


def main(argv=None):
    ap = argparse.ArgumentParser(description="raw dump -> LeRobotDataset")
    ap.add_argument("--raw", default="data/raw", help="raw dump 的根目錄")
    ap.add_argument("--out", default="data/lerobot", help="dataset 輸出目錄")
    ap.add_argument("--repo-id", default="screamlab/omx_pick_cube")
    ap.add_argument("--force", action="store_true", help="輸出目錄已存在就砍掉重建")
    ap.add_argument("--limit", type=int, default=0, help="只轉前 N 集(除錯用)")
    args = ap.parse_args(argv)

    cfg = task_config.load()
    joints = list(cfg.joints)
    cameras = list(cfg["ros"]["camera_topics"])
    resolution = tuple(cfg["cameras"]["record_resolution"])
    fps = cfg["timing"]["fps"]

    raw_root = (REPO_ROOT / args.raw) if not Path(args.raw).is_absolute() \
        else Path(args.raw)
    out = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() \
        else Path(args.out)

    eps = episode_dirs(raw_root)
    if args.limit:
        eps = eps[:args.limit]

    print(f"raw      : {raw_root}  ({len(eps)} 集)")
    print(f"out      : {out}")
    print(f"joints({len(joints)}): {', '.join(joints)}")
    print(f"cameras  : {', '.join(cameras)}  @ {resolution[0]}x{resolution[1]}")
    print(f"fps      : {fps}")

    if out.exists():
        if not args.force:
            raise ConvertError(f"{out} 已存在。要重建就加 --force。")
        shutil.rmtree(out)

    # Imported here rather than at module scope: the checks above should fail in
    # milliseconds, and importing lerobot costs seconds.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = build_features(joints, cameras, resolution)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=features, root=out,
        robot_type="omx_f",
        # PNG rather than MP4: this moves video decode out of the dataloader,
        # which at 200 episodes is where the time actually goes.
        use_videos=False,
    )

    total, all_ages = 0, []
    for ep_dir in eps:
        rows, meta, ages = load_episode(ep_dir, joints, cameras)
        for row in rows:
            ds.add_frame(row)
        ds.save_episode()
        total += len(rows)
        all_ages.append(ages)
        print(f"  {ep_dir.name}  {len(rows):4d} 幀"
              f"  (raw {meta['frames']})", flush=True)

    # ⚠️ Without this the parquet footer is never written and the dataset does
    # not load back -- with no error at write time.
    ds.finalize()

    ages = np.concatenate(all_ages)
    ages = ages[~np.isnan(ages)]
    period = 1.0 / fps
    print(f"\n寫出 {len(eps)} 集 / {total} 幀 -> {out}")
    if len(ages):
        stale = float((ages > period).mean()) * 100
        print(f"影像年齡: 平均 {ages.mean() * 1000:.2f} ms  最大 "
              f"{ages.max() * 1000:.2f} ms  控制週期 {period * 1000:.2f} ms")
        print(f"          超過一個控制週期的比例 {stale:.1f}%")
        if stale > 50:
            print("  ⚠️ 過半的幀用的是比控制週期還舊的影像 —— 相機比控制迴圈慢,"
                  "資料裡有重複像素。先確認算繪速率,不要直接拿去訓練。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ConvertError, task_config.ConfigError) as exc:
        print(f"[convert] {exc}")
        sys.exit(1)
