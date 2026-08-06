"""Write one episode to a format-agnostic raw dump.

    data/raw/ep_00001/
        frames.npz     state, action, per-camera image index and age, timestamps
        img_front_*.png
        img_wrist_*.png
        meta.json

Raw rather than LeRobot directly: LeRobot moved v2 -> v2.1 -> v3.0 inside a year,
and GR00T wants a v2 variant plus ``meta/modality.json`` while ACT wants v3.0.
Keeping a format-free dump makes a format change a re-run of the converter
instead of a re-recording of 200 episodes.

**Time alignment.** ``state`` and ``action`` are taken from the control tick that
produced them, so ``action[t]`` really is the command issued at ``state[t]``.
Images cannot be: they arrive on their own topic at the renderer's pace, so each
frame records **which** image was current and **how old it was**. Nothing is
resampled or interpolated here -- the converter decides what to do with a stale
frame, and it can only decide that if the staleness was written down.

Images are stored once per unique camera timestamp and referenced by index. At a
30 Hz control loop and a 12 Hz camera that is 2-3 ticks per image; writing a PNG
per tick would triple the size to store the same pixels.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class Recorder:
    """One directory per episode, written only if the episode succeeded."""

    def __init__(self, cfg, root="/vla/data/raw"):
        self.cfg = cfg
        self.root = Path(root)
        self.kept = 0
        self.dropped = 0
        self._ep = None

    # ── episode ────────────────────────────────────────────────────────
    def begin(self, index, meta):
        self._dir = self.root / f"ep_{index:05d}"
        self._meta = dict(meta)
        self._rows = []
        self._images = {}          # camera -> {stamp_ns: (index, ndarray)}
        self._ep = index

    def frame(self, state, action, images, t_wall):
        """One control tick.

        ``images`` maps a camera name to ``(msg, arrived_at)``, or None if
        nothing has arrived for it yet.
        """
        row = {"t": t_wall,
               "state": np.asarray(state, dtype=np.float32),
               "action": np.asarray(action, dtype=np.float32)}
        for name, got in images.items():
            row[f"{name}_idx"] = -1
            row[f"{name}_age"] = np.float32("nan")
            if got is None:
                continue
            msg, arrived = got
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            store = self._images.setdefault(name, {})
            if stamp not in store:
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                try:
                    arr = arr.reshape(msg.height, msg.width, -1)[:, :, :3]
                except ValueError:
                    continue                   # partial frame; leave this tick blank
                store[stamp] = (len(store), self._shrink(arr))
            row[f"{name}_idx"] = store[stamp][0]
            # Measured from arrival rather than from the header: Isaac stamps
            # with simulation time, which is not the control loop's clock.
            row[f"{name}_age"] = np.float32(t_wall - arrived)
        self._rows.append(row)

    def _shrink(self, arr):
        """Down to ``cameras.record_resolution`` before it is stored.

        The render resolution is chosen for looking at; the dataset resolution is
        chosen for training. At 640x480 an episode is ~47 MB, so 200 of them
        would be 9 GB of pixels a policy never sees at that size anyway.
        """
        want = tuple(self.cfg["cameras"]["record_resolution"])
        if (arr.shape[1], arr.shape[0]) == want:
            return arr
        from PIL import Image as PILImage
        return np.asarray(PILImage.fromarray(arr).resize(want, PILImage.BILINEAR))

    def end(self, success):
        """Keep the episode only if it succeeded. Returns the directory or None.

        Failed episodes are discarded rather than labelled: a demonstration set
        is supposed to demonstrate. Their statistics are still counted, because a
        drop rate is worth watching.
        """
        if self._ep is None:
            return None
        if not success or not self._rows:
            self.dropped += 1
            self._ep = None
            return None

        from PIL import Image as PILImage

        self._dir.mkdir(parents=True, exist_ok=True)
        for name, store in self._images.items():
            for _, (idx, arr) in sorted(store.items()):
                PILImage.fromarray(arr).save(self._dir / f"img_{name}_{idx:05d}.png")

        cols = {k: np.stack([r[k] for r in self._rows])
                for k in self._rows[0] if k != "t"}
        cols["t"] = np.array([r["t"] for r in self._rows], dtype=np.float64)
        np.savez_compressed(self._dir / "frames.npz", **cols)

        ages = np.concatenate([cols[f"{n}_age"] for n in self._images]) if self._images \
            else np.array([np.nan])
        self._meta.update(
            frames=len(self._rows),
            joints=list(self.cfg.joints),
            fps=self.cfg["timing"]["fps"],
            images={n: len(s) for n, s in self._images.items()},
            resolution=list(self.cfg["cameras"]["record_resolution"]),
            # Recorded, not assumed: the camera runs slower than the control
            # loop on this machine, and the converter needs to know by how much.
            image_age_s={"mean": float(np.nanmean(ages)),
                         "max": float(np.nanmax(ages))},
            recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        (self._dir / "meta.json").write_text(
            json.dumps(self._meta, indent=2, ensure_ascii=False))
        self.kept += 1
        self._ep = None
        return self._dir

    def summary(self):
        return f"錄下 {self.kept} 集,丟棄 {self.dropped} 集(失敗的不留) -> {self.root}"
