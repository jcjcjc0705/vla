#!/bin/bash
# Run a script under Isaac's Python (CPython 3.11) with the USD libraries wired up.
#
# Isaac only puts `pxr` on sys.path once its extensions load, i.e. after a
# SimulationApp exists. The scene tooling deliberately avoids SimulationApp -- it
# only needs to author USD, so it should not take a GPU, take seconds not minutes,
# and never contend with someone else running Isaac on the same machine. So point
# at the USD libraries directly.
#
# LD_LIBRARY_PATH has to be set before the process starts (the dynamic linker
# reads it at exec), which is why this is a wrapper rather than a few lines
# inside the Python file.
#
#   bash sim/isaac_python.sh sim/build_scene.py --force
set -euo pipefail

ISAAC="${ISAAC_SIM_PATH:-}"
if [ -z "$ISAAC" ]; then
  for c in "$HOME/isaac_sim_5.1" "$HOME/Desktop/isaac_sim_5.1"; do
    [ -x "$c/python.sh" ] && ISAAC="$c" && break
  done
fi
if [ -z "$ISAAC" ] || [ ! -x "$ISAAC/python.sh" ]; then
  echo "找不到 Isaac Sim。設 ISAAC_SIM_PATH 指向安裝目錄。" >&2
  exit 1
fi

USDLIBS=$(ls -d "$ISAAC"/extscache/omni.usd.libs-* 2>/dev/null | head -1)
if [ -z "$USDLIBS" ]; then
  echo "在 $ISAAC/extscache 找不到 omni.usd.libs-* —— Isaac 安裝不完整?" >&2
  exit 1
fi

export PYTHONPATH="$USDLIBS:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$USDLIBS/bin:$ISAAC/kit:${LD_LIBRARY_PATH:-}"

# Unbuffered, or nothing you print survives. Redirecting stdout to a file makes
# print() block-buffered, and SimulationApp.close() ends the process hard enough
# that the buffer is never flushed -- the run looks silent even when it worked.
export PYTHONUNBUFFERED=1

exec "$ISAAC/python.sh" "$@"
