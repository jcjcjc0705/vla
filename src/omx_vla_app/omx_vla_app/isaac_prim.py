"""Write USD prim attributes over Isaac Sim's ROS 2 services.

Reading the cube goes over TF -- a stream, wanted every tick. This is the other
direction: putting the cube somewhere for the next episode, a few requests per
episode.

``ROS2SubscribeTransformTree`` looks like it should do this and does not: it
modifies articulation roots, and a loose rigid body is not one.

**The services only answer while the timeline is playing**, which is what a
timeout almost always means.
"""
from __future__ import annotations

import json
import time

from isaac_ros2_messages.srv import GetPrimAttribute, SetPrimAttribute


class IsaacPrimError(RuntimeError):
    """A prim service call failed. The message is meant to be shown as-is."""


class IsaacPrim:
    """Client for Isaac's prim attribute services, bound to an existing node.

    Calls block the caller. That is safe only because the node is spun by an
    executor on **another** thread; called from inside the executor's own thread
    this would deadlock.
    """

    def __init__(self, node, cfg, call_timeout: float = 5.0):
        self._node = node
        self._timeout = call_timeout
        ros = cfg["ros"]
        self._get = node.create_client(GetPrimAttribute, ros["get_attribute_service"])
        self._set = node.create_client(SetPrimAttribute, ros["set_attribute_service"])
        self._names = (ros["get_attribute_service"], ros["set_attribute_service"])

    def wait_for_isaac(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        for client, name in zip((self._get, self._set), self._names):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not client.wait_for_service(timeout_sec=remaining):
                print(f"[expert] 等不到服務 {name} ——"
                      "\n  1. 場景要是 vla/assets/pick_cube.usd(只有它帶 ROS2ServicePrim)"
                      "\n  2. Isaac 的 Play 要按下去")
                return False
        return True

    def _call(self, client, request, what):
        future = client.call_async(request)
        deadline = time.monotonic() + self._timeout
        while not future.done():
            if time.monotonic() > deadline:
                raise IsaacPrimError(f"{what} 逾時 —— Isaac 的 Play 還在跑嗎?")
            time.sleep(0.005)
        result = future.result()
        if result is None:
            raise IsaacPrimError(f"{what} 沒有結果")
        if not result.success:
            raise IsaacPrimError(f"{what} 失敗: {result.message or '(沒有訊息)'}")
        return result

    def get(self, path: str, attribute: str):
        request = GetPrimAttribute.Request()
        request.path, request.attribute = path, attribute
        result = self._call(self._get, request, f"讀 {path}.{attribute}")
        try:
            return json.loads(result.value)
        except (json.JSONDecodeError, TypeError):
            return result.value

    def set(self, path: str, attribute: str, value) -> None:
        request = SetPrimAttribute.Request()
        request.path, request.attribute = path, attribute
        request.value = value if isinstance(value, str) else json.dumps(value)
        self._call(self._set, request, f"寫 {path}.{attribute}")
