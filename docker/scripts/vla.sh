#!/bin/bash
# Idle container for the VLA tools. Nothing starts on its own -- open Isaac
# yourself, press Play, then run `ros2 run data_collection expert` in here.
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"
main "${1:-up}" "$COMPOSE_DIR/docker-compose-vla.yml" "${@:2}"
