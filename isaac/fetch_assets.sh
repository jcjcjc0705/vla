#!/bin/bash
# Pull the robot's data out of omx_bridge_image so the **host-side** Isaac can
# read it.
#
#     bash isaac/fetch_assets.sh
#
# Why this exists: Isaac runs natively on the host (it needs the GPU, RTX and
# WebRTC), but everything that describes this arm -- the USD, the meshes, the
# joint-order profile -- is baked into the image. Rather than asking every
# machine to clone omx_bridge_image and hope the checkout matches the image
# tag, this copies the three things the host needs straight out of the image
# that the containers are already running.
#
# That is the whole point: **one source of truth, not one per machine.** The
# previous arrangement listed a laptop path, a server path and a home-directory
# path in task/pick_cube.task.yaml, any of which could silently be a different
# version from the image.
#
# ⚠️ Re-run this after pulling a new omx_bridge_image. The cache is not
# versioned; it is a copy, and a stale copy of omx_f.usd against a newer image
# is exactly the kind of mismatch that shows up as "the scene looks right but
# the numbers are off".
set -euo pipefail

IMAGE="${OMX_BRIDGE_IMAGE:-registry.screamtrumpet.csie.ncku.edu.tw/pochun/omx_bridge_image:latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="$HERE/../.image_cache"

echo "來源 image : $IMAGE"
echo "匯出到     : $CACHE"

rm -rf "$CACHE"
mkdir -p "$CACHE"

# One container, three copies. `docker cp` needs a container rather than an
# image, so create one without running it.
CID=$(docker create "$IMAGE")
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

docker cp "$CID:/assets"  "$CACHE/assets"
docker cp "$CID:/profile" "$CACHE/profile"
# The profile *reader*, not just the yaml -- joint order has exactly one
# implementation and the host must use that one too (invariant 3).
#
# ⚠️ Copied from **src/**, not build/. The image was built with
# `colcon build --symlink-install`, so /workspaces/build/... is a symlink into
# /workspaces/src/... and `docker cp` faithfully copies the *link*, leaving a
# dangling pointer to a path that only exists inside the container. The failure
# is quiet: the directory appears to exist and contains nothing.
docker cp "$CID:/workspaces/src/sim_real_bridge/sim_real_bridge" "$CACHE/sim_real_bridge"

# Belt and braces -- if the layout changes again, fail here rather than three
# steps later with "no module named sim_real_bridge".
if [ ! -f "$CACHE/sim_real_bridge/profile.py" ]; then
    echo "匯出失敗:$CACHE/sim_real_bridge/profile.py 不存在" >&2
    echo "image 的目錄結構可能變了,檢查 /workspaces/src/sim_real_bridge/" >&2
    exit 1
fi

echo
echo "  assets/omx_f.usd  $(du -h "$CACHE/assets/omx_f.usd" | cut -f1)"
echo "  assets/omx_f/     $(find "$CACHE/assets/omx_f" -name '*.stl' | wc -l) 個 STL"
echo "  profile/          $(ls "$CACHE/profile")"
echo "  sim_real_bridge/  $(ls "$CACHE/sim_real_bridge"/*.py | wc -l) 個模組"
echo
echo "✓ 完成。host 上的 isaac/build_scene.py 現在讀得到這隻手臂的資料。"
