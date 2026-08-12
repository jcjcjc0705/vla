"""Check a converted dataset before anyone trains on it.

    python3 ml/audit.py --raw data/raw_3obj --dataset data/lerobot_3obj

Everything here is something that has actually gone wrong in this project, or
would have been invisible until it wasted a training run:

* **One distinct instruction.** The three-object task's whole point is that the
  language carries information. A recorder bug that froze the target would leave
  a dataset that trains fine and measures nothing.
* **A polluted holdout.** M5's number only means something because the held-out
  theta bands contain zero training episodes. That was verified by counting the
  486 episodes on disk, not by trusting the sampler, and it should stay that way.
* **Objects closer than the minimum separation.** A pair that overlaps makes an
  episode where the expert's own demonstration knocks a distractor over.
* **Stale images.** The camera runs at its own rate; the converter reports the
  distribution, and a long tail means the policy is sometimes acting on pixels
  several control periods old.

Exit code is 1 if any hard check fails, so a script can gate on it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "omx_vla_app"))

from omx_vla_app import task_config  # noqa: E402
from omx_vla_app.spawn import holdout_bands, is_holdout  # noqa: E402


def audit_raw(raw: Path, cfg):
    """Checks that need the dump: instructions, targets, holdout, separation."""
    out, hard = [], True
    metas = sorted(raw.glob("ep_*/meta.json"))
    if not metas:
        return [f"✗ {raw} 裡沒有 ep_*/meta.json"], False
    out.append(f"raw           {len(metas)} 集  ({raw})")

    ins, tgt, seps, thetas, nobj = Counter(), Counter(), [], [], Counter()
    for f in metas:
        d = json.loads(f.read_text())
        ins[d.get("instruction", "(無)")] += 1
        tgt[d.get("target", "(無)")] += 1
        objs = d.get("objects") or {}
        nobj[len(objs)] += 1
        ps = [v["requested"][:2] for v in objs.values()]
        if len(ps) > 1:
            seps.append(min(math.dist(a, b) for i, a in enumerate(ps)
                            for b in ps[i + 1:]))
        c = d.get("cube") or {}
        if "theta_deg" in c:
            thetas.append(float(c["theta_deg"]))

    out.append("")
    out.append(f"指令種類      {len(ins)}")
    for k, v in ins.most_common():
        out.append(f"   {v:5d} 集  {k!r}")
    if len(ins) < 2:
        out.append("✗ 只有一種指令 —— 語言那一維是常數,這個 dataset 測不了語言接地")
        hard = False

    out.append("")
    out.append(f"目標分布      {dict(tgt)}")
    if tgt:
        lo, hi = min(tgt.values()), max(tgt.values())
        skew = hi / max(lo, 1)
        out.append(f"   最多/最少 = {skew:.2f}" +
                   ("" if skew < 1.3 else "   ⚠️ 偏斜,某個物體的示範明顯少"))

    if nobj:
        out.append(f"每集物體數    {dict(nobj)}")
        if len(nobj) > 1:
            out.append("⚠️ 集與集之間物體數不一致")

    if seps:
        want = cfg["spawn"].get("min_separation", 0)
        out.append("")
        out.append(f"物體最小間距  {min(seps)*1000:.0f} ~ {max(seps)*1000:.0f} mm"
                   f"  (要求 ≥ {want*1000:.0f})")
        if min(seps) < want - 1e-6:
            out.append(f"✗ 有 {sum(1 for s in seps if s < want - 1e-6)} 集低於下限")
            hard = False

    if thetas:
        bands = holdout_bands(cfg)
        bad = sum(is_holdout(t, bands) for t in thetas)
        out.append("")
        out.append(f"目標 theta    {min(thetas):+.0f}° ~ {max(thetas):+.0f}°")
        out.append(f"落在保留帶    {bad} / {len(thetas)}"
                   + ("   ✓ 乾淨" if bad == 0 else ""))
        if bad:
            out.append("✗ 訓練資料污染了保留帶 —— 保留區的成績會失去意義")
            hard = False
    return out, hard


def audit_dataset(root: Path):
    """Checks that need the converted dataset: shapes, dtypes, task column."""
    out, hard = [], True
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("screamlab/omx_pick_cube", root=str(root))
    out.append(f"dataset       {ds.num_episodes} 集 / {ds.num_frames} 幀"
               f" / {ds.fps} fps  ({root})")
    if ds.num_episodes == 0:
        return out + ["✗ dataset 是空的"], False

    s = ds[0]
    shapes = {k.rsplit(".", 1)[-1]: tuple(v.shape)
              for k, v in s.items() if "images" in k}
    out.append(f"影像          {shapes}")
    edges = {v[1:] for v in shapes.values()}
    if len(edges) != 1:
        out.append("✗ 相機之間的影像尺寸不一致 —— GR00T 的 np.stack 會直接爆")
        hard = False
    elif len(set(next(iter(edges)))) != 1:
        out.append("⚠️ 影像不是正方形 —— GR00T 會非等比拉伸,方塊的 yaw 會被扭曲")

    out.append(f"state/action  {tuple(s['observation.state'].shape)}"
               f" / {tuple(s['action'].shape)}")
    # ⚠️ state must be measured and action commanded; identical columns would
    # teach an identity map that scores well and does nothing.
    same = float((s["observation.state"] - s["action"]).abs().max())
    out.append(f"state≠action  最大差 {same:.4f}"
               + ("   ✗ 兩者相同,會學成恆等映射" if same < 1e-9 else ""))
    if same < 1e-9:
        hard = False

    # Sample the task column across episodes rather than trusting frame 0.
    idx = np.linspace(0, ds.num_frames - 1, min(300, ds.num_frames)).astype(int)
    tasks = Counter(ds[int(i)]["task"] for i in idx)
    out.append(f"task 欄位     抽 {len(idx)} 幀,{len(tasks)} 種")
    for k, v in tasks.most_common():
        out.append(f"   {v:4d} 幀  {k!r}")
    if len(tasks) < 2:
        out.append("✗ dataset 的 task 欄位是常數 —— GR00T 讀不到任何資訊")
        hard = False
    return out, hard


def main(argv=None):
    ap = argparse.ArgumentParser(description="訓練前檢查 dataset")
    ap.add_argument("--raw", default="data/raw_3obj")
    ap.add_argument("--dataset", default="data/lerobot_3obj")
    ap.add_argument("--task", default=None, help="任務規格(預設用當前的)")
    a = ap.parse_args(argv)

    cfg = task_config.load(Path(a.task)) if a.task else task_config.load()
    lines, ok = [], True
    for fn, arg in ((audit_raw, Path(a.raw)), (audit_dataset, Path(a.dataset))):
        try:
            got, good = fn(arg, cfg) if fn is audit_raw else fn(arg)
        except Exception as exc:                              # noqa: BLE001
            got, good = [f"✗ {fn.__name__} 失敗:{type(exc).__name__}: {exc}"], False
        lines += got + [""]
        ok &= good

    print("\n".join(lines))
    print("=" * 60)
    print("✅ 全部檢查通過,可以開始訓練" if ok else
          "🛑 有硬性檢查沒過 —— 上面標 ✗ 的先處理,不要直接訓練")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
