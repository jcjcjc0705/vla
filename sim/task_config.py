"""載入 pick_cube.task.yaml,解析路徑,取得 canonical 關節順序。

跑在 **Isaac 的 CPython 3.11** 下(`./python.sh`),所以這裡只能用純 Python +
PyYAML —— 不要 import torch、numpy 以外的重東西,也不要 import rclpy。

關節順序**不在這裡定義**。它從 `sim_real_bridge.profile` 讀,那是全鏈唯一的真相:
同一份 profile 同時被這個模擬器、轉檔器、以及日後 M7 的 ROS node 使用。抄一份常數
就是製造第二個真相,遲早會對不上。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_FILE = REPO_ROOT / "task" / "pick_cube.task.yaml"


class ConfigError(RuntimeError):
    """設定或相依路徑有問題 —— 訊息要能直接告訴使用者去改哪裡。"""


def _resolve(candidates, must_contain, what):
    """依序試 candidates,回傳第一個存在且含有 must_contain 的目錄。

    路徑在筆電與伺服器上不同,又不想散落 machine-specific 的設定,所以列一組候選
    讓它自己找。找不到時的錯誤訊息要包含試過哪些,不然使用者無從下手。
    """
    tried = []
    for c in candidates:
        p = Path(os.path.expanduser(c))
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        tried.append(str(p))
        if (p / must_contain).exists():
            return p
    raise ConfigError(
        f"找不到 {what}。試過:\n  " + "\n  ".join(tried)
        + f"\n每個都必須含有 {must_contain}。"
        f"\n請改 {TASK_FILE} 的 paths 區塊,或把 repo clone 到並排的位置。"
    )


class Config:
    """task yaml 的存取器,外加解析好的路徑與關節順序。"""

    def __init__(self, task_file: Path = TASK_FILE):
        if not task_file.exists():
            raise ConfigError(f"找不到 task 檔:{task_file}")
        self.task_file = task_file
        self.raw = yaml.safe_load(task_file.read_text())

        paths = self.raw["paths"]
        self.omx_bridge = _resolve(
            paths["omx_bridge"], paths["robot_usd"], "omx_bridge_image"
        )
        # Inside omx_bridge_image, sim_real_bridge is an **installed** ROS package
        # on PYTHONPATH, not a directory to be found. Only fall back to searching
        # when the import is not already available (which is the native-Isaac case).
        self.bridge_root = None
        try:
            import sim_real_bridge.profile  # noqa: F401
        except ImportError:
            self.bridge_root = _resolve(
                paths["sim_real_bridge"], "sim_real_bridge/profile.py", "sim_real_bridge"
            )

        self.robot_usd = self.omx_bridge / paths["robot_usd"]
        self.robot_urdf = self.omx_bridge / paths["robot_urdf"]
        self.profile_path = self.omx_bridge / paths["profile"]
        self.scene_usd = REPO_ROOT / paths["scene_usd"]

        self.joints = self._load_joints()

    def _load_joints(self):
        """canonical 關節順序,來自 sim_real_bridge —— 全鏈唯一的真相。"""
        if self.bridge_root is not None and str(self.bridge_root) not in sys.path:
            sys.path.insert(0, str(self.bridge_root))
        try:
            from sim_real_bridge.profile import load_profile
        except ImportError as exc:                      # noqa: BLE001
            where = self.bridge_root or "已安裝的 ROS 套件"
            raise ConfigError(
                f"從 {where} import sim_real_bridge.profile 失敗:{exc}\n"
                "它是純 Python + yaml,不該有相依問題 —— 容器內請先 source "
                "install/setup.bash,原生環境請檢查 clone 是否完整。"
            ) from exc
        return list(load_profile(str(self.profile_path)).joints)

    # ── task yaml 的區塊 ────────────────────────────────────────────────
    def __getitem__(self, key):
        return self.raw[key]

    @property
    def task_root(self):
        return self.raw["scene"]["task_root"]

    @property
    def robot_root(self):
        return self.raw["scene"]["robot_root"]

    def summary(self) -> str:
        c, s = self.raw["cube"], self.raw["spawn"]
        return "\n".join([
            f"task     : {self.task_file}",
            f"robot usd: {self.robot_usd}",
            f"scene usd: {self.scene_usd}  ({'已存在' if self.scene_usd.exists() else '尚未產生'})",
            f"bridge   : {self.bridge_root or '已安裝的 sim_real_bridge 套件'}",
            f"joints({len(self.joints)}): {', '.join(self.joints)}",
            f"cube     : {c['size'] * 1000:.0f} mm, {c['mass'] * 1000:.0f} g, "
            f"摩擦 {c['static_friction']}",
            f"spawn    : r={s['radius']} m, theta={s['theta_deg']}°, "
            f"保留區 {s['holdout_theta_deg']}°",
        ])


def load(task_file: Path = TASK_FILE) -> Config:
    return Config(task_file)


if __name__ == "__main__":
    try:
        print(load().summary())
    except ConfigError as exc:
        print(f"[config] {exc}")
        sys.exit(1)
