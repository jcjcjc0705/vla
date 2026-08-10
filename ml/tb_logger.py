"""TensorBoard for ``lerobot-train``, which only knows how to log to W&B.

Used by ``ml/train.py --tensorboard``; there is no reason to import this
directly.

**How it attaches.** lerobot's training loop never mentions TensorBoard, but
every metric it reports goes through one object::

    if wandb_logger:
        wandb_logger.log_dict(train_tracker.to_dict(), step)

So this supplies an object with that same shape and substitutes it for
``WandBLogger`` in the train module. Nothing in the loop changes, and every
metric lerobot decides to report -- including policy sub-losses like
``l1_loss`` and ``kld_loss``, which the tracker aggregates on its own -- comes
through for free. Re-implementing the loop's logging would have meant tracking
that list by hand forever.

⚠️ **The substitution needs ``wandb.enable=true`` to get past the gate.**
The construction site is::

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)

so ``ml/train.py`` sets those two flags and swaps the class. **wandb itself is
never imported and nothing is uploaded** -- the name is the only thing borrowed.
If you see ``wandb.enable=true`` in the effective config of a ``--tensorboard``
run, that is this, not a stray upload.

**Metric names** are passed through as lerobot emits them, with ``mode`` as the
prefix: ``train/loss``, ``train/grad_norm``, ``eval/eval_loss``. Non-numeric
values are skipped rather than coerced -- a string in a scalar plot is worse
than a missing one.
"""
from __future__ import annotations

import logging
from pathlib import Path

_SKIP = {"step", "steps", "epoch", "samples"}


class TensorBoardLogger:
    """Duck-typed stand-in for ``lerobot.common.wandb_utils.WandBLogger``."""

    def __init__(self, cfg):
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(cfg.output_dir) / "tb"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        logging.info(f"TensorBoard -> {self.log_dir}")
        logging.info(f"  tensorboard --logdir {Path(cfg.output_dir)} --bind_all")
        try:
            # One row of hyperparameters, so runs are comparable in the UI later.
            flat = {k: v for k, v in _flatten(cfg.to_dict()).items()
                    if isinstance(v, (int, float, str, bool))}
            self._writer.add_text("config", _as_markdown(flat), 0)
        except Exception as exc:                                  # noqa: BLE001
            # Never let bookkeeping kill a training run that is otherwise fine.
            logging.warning(f"TensorBoard 寫設定失敗(不影響訓練): {exc}")

    def log_dict(self, d, step=None, mode="train", custom_step_key=None):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is not None:
            step = d.get(custom_step_key)
        if step is None:
            return
        for k, v in d.items():
            if k in _SKIP or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"{mode}/{k}", v, step)
        self._writer.flush()

    def log_policy(self, checkpoint_dir):
        """W&B uploads the checkpoint here. TensorBoard has nowhere to put it."""

    def log_video(self, video_path, step, mode="train"):
        """Same -- videos stay on disk; the path is already in the log."""


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _as_markdown(flat):
    rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(flat.items()))
    return f"| key | value |\n| --- | --- |\n{rows}"


def install() -> Path | None:
    """Swap ``TensorBoardLogger`` in for ``WandBLogger`` in the train module."""
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"--tensorboard 需要 tensorboard 套件,但匯入失敗:{exc}\n"
            "容器裡先 `pip install tensorboard`。⚠️ 那是暫時的,容器重建就沒了 —— "
            "長久的做法是加進 omx_vla_image 的 Dockerfile。"
        ) from exc

    from lerobot.scripts import lerobot_train

    lerobot_train.WandBLogger = TensorBoardLogger
    return None
