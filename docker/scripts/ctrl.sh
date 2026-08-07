#!/bin/bash
# 控制層容器(omx_bridge_image:numpy 1.x + MoveIt)。需要 ik:=moveit、jog、
# ik_target 或 monitor 時用這一支;轉檔 / 訓練 / 推論 / ik:=analytic 用 vla.sh。
# 為什麼要分兩個,見 docker/compose/docker-compose-ctrl.yml 的開頭。
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"
main "${1:-up}" "$COMPOSE_DIR/docker-compose-ctrl.yml" "${@:2}"
