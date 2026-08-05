#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_DIR="$REPO_ROOT/docker/compose"
ENV_FILE="$COMPOSE_DIR/.env"

export COMPOSE_IGNORE_ORPHANS=True

if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_COMMAND="docker compose --env-file $ENV_FILE"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_COMMAND="docker-compose --env-file $ENV_FILE"
else
    echo "Neither 'docker compose' nor 'docker-compose' is installed."
    exit 1
fi

cleanup() {
    local compose_file="$1"
    echo
    echo "Shutting down docker compose services..."
    echo "Stopping services for $compose_file..."
    $DOCKER_COMPOSE_COMMAND -f "$compose_file" down --timeout 0
    exit 0
}

main() {
    local mode="$1"
    local compose_file="$2"
    shift 2 2>/dev/null

    if [[ ! -f "$compose_file" ]]; then
        echo "Error: Compose file not found: $compose_file" >&2
        exit 1
    fi

    local service
    service=$($DOCKER_COMPOSE_COMMAND -f "$compose_file" config --services | head -1)
    local container_name
    container_name=$(grep -oP 'container_name:\s*\K\S+' "$compose_file" | head -1)
    local debug_name="${container_name:-$service}_debug"

    case "$mode" in
        down)
            docker rm -f "$debug_name" &> /dev/null || true
            cleanup "$compose_file"
            ;;

        exec)
            $DOCKER_COMPOSE_COMMAND -f "$compose_file" exec -it "$service" bash
            ;;

        debug)
            if ! docker ps --format '{{.Names}}' | grep -qx "$debug_name"; then
                $DOCKER_COMPOSE_COMMAND -f "$compose_file" stop "$service"
                $DOCKER_COMPOSE_COMMAND -f "$compose_file" run --rm -d --name "$debug_name" \
                    --entrypoint bash "$service" -c "tail -f /dev/null"
            fi
            docker exec -it "$debug_name" bash
            ;;

        up | "")
            trap 'cleanup "$compose_file"' SIGINT

            echo "Starting services for $compose_file..."
            $DOCKER_COMPOSE_COMMAND -f "$compose_file" up -d

            $DOCKER_COMPOSE_COMMAND -f "$compose_file" exec -it "$service" bash
            cleanup "$compose_file"
            ;;

        *)
            $DOCKER_COMPOSE_COMMAND -f "$compose_file" "$mode" "$@"
            ;;
    esac
}

# to_sim / to_real run their node from the compose `command`, so their default is
# a FOREGROUND `up` -- the node's log is the point of watching them. Every other
# mode falls through to main() above.
main_foreground() {
    local mode="$1"
    local compose_file="$2"
    shift 2 2>/dev/null

    case "$mode" in
        up | "")
            trap 'cleanup "$compose_file"' SIGINT
            $DOCKER_COMPOSE_COMMAND -f "$compose_file" up
            ;;
        *)
            main "$mode" "$compose_file" "$@"
            ;;
    esac
}
