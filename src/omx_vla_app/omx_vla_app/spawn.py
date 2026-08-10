"""Where the cube goes each episode -- the one sampler, shared by every caller.

Pure Python + numpy, no Isaac and no ROS. One copy on purpose: three copies of a
distribution is three chances for the training set and the evaluation set to stop
meaning what they say.

**The held-out bands sit inside the sweep, not at its edge, and there are two of
them.** That combination is what makes M5's acceptance criterion mean anything:

* A policy that ignores the image replays roughly the *mean* trajectory, so a
  band in the middle is the one place such a policy passes.
* A band at the edge tests extrapolation, and then a failure has two possible
  causes -- "not looking" and "never saw that part of the workspace" -- which is
  the distinction M5 exists to make.

Two symmetric interior bands are far from the mean (a trajectory-replaying policy
fails there) while bracketed by training data on both sides (a policy that reads
the image can interpolate), and they leave the training distribution centred.
"""
from __future__ import annotations

import math

import numpy as np


def holdout_bands(cfg):
    """``spawn.holdout_theta_deg`` as a list of (lo, hi), in degrees.

    Accepts a single ``[lo, hi]`` as well as ``[[lo, hi], ...]`` so an older
    config still loads.
    """
    raw = cfg["spawn"]["holdout_theta_deg"]
    bands = [raw] if raw and not isinstance(raw[0], (list, tuple)) else list(raw)
    return [(float(lo), float(hi)) for lo, hi in bands]


def is_holdout(theta_deg, bands):
    return any(lo <= theta_deg <= hi for lo, hi in bands)


def sample_theta(cfg, rng, holdout=False):
    """One theta in degrees, from the held-out bands or from everything else."""
    bands = holdout_bands(cfg)
    if holdout:
        # Weight by width so a wider band is not under-sampled.
        widths = np.array([hi - lo for lo, hi in bands], dtype=float)
        lo, hi = bands[int(rng.choice(len(bands), p=widths / widths.sum()))]
        return rng.uniform(lo, hi)
    sweep = cfg["spawn"]["theta_deg"]
    while True:
        th = rng.uniform(*sweep)
        if not is_holdout(th, bands):
            return th


def sample_cube_pose(cfg, rng, holdout=False):
    """``(position, yaw_rad, r, theta_deg)`` for one episode.

    ``rng`` is passed in rather than seeded here: a single stream across an
    episode batch is what makes the batch varied, and a caller that wants one
    reproducible pose can hand in ``default_rng(seed)``.
    """
    s = cfg["spawn"]
    th = sample_theta(cfg, rng, holdout)
    r = rng.uniform(*s["radius"])
    yaw = math.radians(rng.uniform(*s["yaw_deg"]))
    a = math.radians(th)
    pos = np.array([r * math.cos(a), r * math.sin(a), cfg["cube"]["size"] / 2])
    return pos, yaw, r, th


def yaw_quat(yaw):
    """w,x,y,z for a rotation about +Z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def describe(cfg):
    bands = holdout_bands(cfg)
    lo, hi = cfg["spawn"]["theta_deg"]
    kept = (hi - lo) - sum(b - a for a, b in bands)
    return (f"theta {lo:+.0f}..{hi:+.0f}°,保留 "
            + " 與 ".join(f"[{a:+.0f},{b:+.0f}]" for a, b in bands)
            + f" → 訓練用 {kept:.0f}°")
