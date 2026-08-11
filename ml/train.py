"""``lerobot-train`` with the chunked-action dataloader fix and a config file.

    python3 ml/train.py                     # 跑 ml/train.yaml 的 active 那一組
    python3 ml/train.py --steps=100         # 臨時覆蓋一個值
    python3 ml/train.py --profile=act       # 換一組,不用改檔

**Parameters live in ``ml/train.yaml``, not in your shell history.** That file
carries the reasoning next to each number -- why batch is 24, why steps is 16000
-- which a pasted command line cannot. Anything on the command line still wins,
so one-off experiments do not need an edit.

**Use this instead of the ``lerobot-train`` CLI for this dataset.** Apart from
the config file, every flag is passed straight through.

Why it has to exist: see ``fast_chunk_patch``. Short version -- ACT's 100-step
action chunk makes lerobot decode ~300 PNGs per sample and discard them. On this
dataset, measured at batch 64 / 4 workers::

    lerobot-train        updt_s 0.441   data_s 9.575   smp/s   6
    ml/train.py          updt_s 0.172   data_s 0.012   smp/s 348

60k steps: about **7 days** versus about **3 hours**. The bare CLI still works
and reports no error -- the GPU just idles at 96%.

⚠️ **This forces ``dataloader_multiprocessing_context=fork``**, and that is not
cosmetic. lerobot defaults it to ``spawn``, and a spawned worker re-imports
everything from scratch, so it never sees a patch applied in this process. The
first attempt at this fix looked like it did nothing for exactly that reason:
``ds[i]`` was 49x faster in the parent while ``data_s`` did not move at all,
because none of the work happens in the parent. Pass the flag yourself to
override; a patched parent with spawned workers is the one combination that
silently wastes the whole speedup.

⚠️ ``--policy.push_to_hub=false`` is still required (lerobot rejects the config
without it, complaining about ``repo_id`` in a message that sounds unrelated).

**``--tensorboard``** writes scalars to ``<output_dir>/tb``. lerobot only knows
how to log to W&B, so this borrows that seam -- see ``ml/tb_logger.py``.

    python3 ml/train.py --tensorboard ... --output_dir=outputs/act_v1
    tensorboard --logdir outputs/act_v1 --bind_all      # 另一個終端

⚠️ ``tensorboard`` is pip-installed into the running container, which does not
survive ``docker compose down``. Add it to omx_vla_image's Dockerfile to keep it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fast_chunk_patch  # noqa: E402

_CTX = "--dataloader_multiprocessing_context"
_CONFIG = HERE / "train.yaml"


def _has(argv, flag):
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _profile_args(argv):
    """``ml/train.yaml`` -> CLI 旗標,命令列已經給過的就跳過。

    Keys map to lerobot flags one-to-one (``policy.tune_visual`` ->
    ``--policy.tune_visual``), so a new knob needs a yaml line and no code.

    ⚠️ Skipping what the caller already passed, rather than relying on "last one
    wins", keeps precedence independent of how the parser resolves duplicates.
    """
    import yaml

    if not _CONFIG.exists():
        return argv, None
    cfg = yaml.safe_load(_CONFIG.read_text()) or {}

    name = None
    for a in list(argv):
        if a.startswith("--profile="):
            name = a.split("=", 1)[1]
            argv.remove(a)
    name = name or cfg.get("active")
    profiles = cfg.get("profiles") or {}
    if name not in profiles:
        raise SystemExit(
            f"{_CONFIG} 裡沒有 profile '{name}'。可用的:{sorted(profiles)}")

    extra = []
    for key, val in (profiles[name] or {}).items():
        if key == "tensorboard":
            if val and "--tensorboard" not in argv:
                argv.append("--tensorboard")
            continue
        flag = f"--{key}"
        if _has(argv, flag):
            continue                      # 命令列優先
        extra.append(f"{flag}={'true' if val is True else 'false' if val is False else val}")
    return extra + argv, name


if __name__ == "__main__":
    fast_chunk_patch.apply()
    argv, profile = _profile_args(sys.argv[1:])
    if profile:
        print(f"[train] profile: {profile}  ({_CONFIG})")

    want_tb = "--tensorboard" in argv
    if want_tb:
        argv = [a for a in argv if a != "--tensorboard"]

    if not _has(argv, _CTX):
        # fork, so the workers inherit the patched class. See the warning above.
        argv = [f"{_CTX}=fork", *argv]

    if want_tb:
        import tb_logger

        tb_logger.install()
        # ⚠️ These two only get the logger past lerobot's construction gate
        # (`if cfg.wandb.enable and cfg.wandb.project`). The class behind the
        # name is now TensorBoardLogger -- wandb is never imported and nothing
        # is uploaded. Anything the caller passed explicitly still wins.
        if not _has(argv, "--wandb.enable"):
            argv = ["--wandb.enable=true", *argv]
        if not _has(argv, "--wandb.project"):
            argv = ["--wandb.project=tensorboard", *argv]

    sys.argv = [sys.argv[0], *argv]
    from lerobot.scripts.lerobot_train import main

    sys.exit(main())
