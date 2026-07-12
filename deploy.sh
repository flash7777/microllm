#!/bin/bash
#
# microllm - Deploy Script
#
# Standalone Podman-Container (nicht Teil des Compose-Stacks).
# Läuft im Host-Netzwerk auf Port 8012.
#
# Usage:
#   ./deploy.sh              # Full deploy: push + pull + start
#   ./deploy.sh setup        # Initial setup (push image, copy config, start)
#   ./deploy.sh pull         # Pull image on remote
#   ./deploy.sh start        # Start/replace container on remote
#   ./deploy.sh stop         # Stop container
#   ./deploy.sh restart      # Restart container
#   ./deploy.sh logs         # Show container logs
#   ./deploy.sh status       # Show container status
#   ./deploy.sh config       # Sync config.yaml to remote

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_FILE="${SCRIPT_DIR}/DIST"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "\n${BLUE}=== $1 ===${NC}"; }

#=============================================================================
# Configuration
#=============================================================================

load_config() {
    if [[ -f "${DIST_FILE}" ]]; then
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^#.*$ ]] && continue
            [[ -z "$key" ]] && continue
            value=$(echo "$value" | sed 's/^["'\'']//' | sed 's/["'\'']$//')
            case "$key" in
                HOST) HOST="${value}" ;;
                DOCKER_REGISTRY) DOCKER_REGISTRY="${value}" ;;
                DOCKER_NS) DOCKER_NAMESPACE="${value}" ;;
                IMAGE) IMAGE_NAME="${value}" ;;
                CONTAINER_NAME) CONTAINER_NAME="${value}" ;;
                PORT) PORT="${value}" ;;
                REPO) REPO="${value}" ;;
                GIT_BASE) GIT_BASE="${value}" ;;
            esac
        done < "${DIST_FILE}"
    else
        log_error "DIST file not found"
        exit 1
    fi

    REMOTE_HOST="root@${HOST}"
    REMOTE_CONFIG_DIR="/data/microllm"
    # Support both old (Docker Hub) and new (Codeberg via job.py) image paths
    if [[ -n "${GIT_BASE:-}" && -n "${REPO:-}" ]]; then
        # Codeberg: https://codeberg.org/Gemini-Foundation -> codeberg.org/Gemini-Foundation/microllm
        IMAGE="$(echo "${GIT_BASE}" | sed 's|https://||' | tr '[:upper:]' '[:lower:]')/${REPO}"
    else
        IMAGE="${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${IMAGE_NAME}"
    fi
}

#=============================================================================
# Commands
#=============================================================================

cmd_push() {
    log_step "Building and Pushing Image"
    "${SCRIPT_DIR}/push.sh"
}

cmd_pull() {
    log_step "Pulling Image on Remote"
    ssh "${REMOTE_HOST}" "podman pull ${IMAGE}:latest"
    log_info "Image pulled"
}

cmd_config() {
    log_step "Syncing Config"
    ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_CONFIG_DIR}"
    scp "${SCRIPT_DIR}/config.yaml" "${REMOTE_HOST}:${REMOTE_CONFIG_DIR}/config.yaml"
    log_info "config.yaml synced to ${REMOTE_CONFIG_DIR}/"
}

cmd_start() {
    log_step "Starting microllm"
    ssh "${REMOTE_HOST}" "podman run -d \
        --replace \
        --name ${CONTAINER_NAME} \
        --network opencloud_full_opencloud-net \
        -p ${PORT}:${PORT} \
        --restart always \
        -v ${REMOTE_CONFIG_DIR}/config.yaml:/config/config.yaml:ro \
        ${IMAGE}:latest \
        /config/config.yaml ${PORT}"
    sleep 1
    log_info "Container started on port ${PORT}"
    ssh "${REMOTE_HOST}" "podman logs ${CONTAINER_NAME} 2>&1 | tail -5"
}

cmd_stop() {
    log_step "Stopping microllm"
    ssh "${REMOTE_HOST}" "podman stop ${CONTAINER_NAME} 2>/dev/null; podman rm ${CONTAINER_NAME} 2>/dev/null" || true
    log_info "Container stopped"
}

cmd_restart() {
    log_step "Restarting microllm"
    ssh "${REMOTE_HOST}" "podman restart ${CONTAINER_NAME}"
    sleep 1
    log_info "Container restarted"
    ssh "${REMOTE_HOST}" "podman logs ${CONTAINER_NAME} 2>&1 | tail -5"
}

cmd_logs() {
    ssh "${REMOTE_HOST}" "podman logs -f ${CONTAINER_NAME}"
}

cmd_status() {
    log_step "microllm Status"
    ssh "${REMOTE_HOST}" "podman ps -a --filter name=${CONTAINER_NAME} --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"
    echo ""
    ssh "${REMOTE_HOST}" "curl -s http://localhost:${PORT}/health 2>/dev/null || echo '(nicht erreichbar)'"
}

cmd_setup() {
    log_step "Initial Setup"
    log_info "Host: ${HOST}"
    log_info "Image: ${IMAGE}"
    log_info "Port: ${PORT}"
    echo ""

    cmd_push
    cmd_pull
    cmd_config
    cmd_start

    log_step "Setup Complete!"
    echo ""
    echo "Health:  http://${HOST}:${PORT}/health"
    echo "Stats:   http://${HOST}:${PORT}/stats"
    echo "Models:  http://${HOST}:${PORT}/v1/models"
}

cmd_deploy() {
    log_step "Full Deployment"
    log_info "Host: ${HOST}"
    log_info "Image: ${IMAGE}"
    echo ""

    cmd_push
    cmd_pull
    cmd_config
    cmd_start

    log_step "Deployment Complete!"
}

show_help() {
    cat << EOF
Usage: $(basename "$0") [command]

Commands:
  (default)   Full deploy: push + pull + config + start
  setup       Initial setup (same as deploy)
  push        Build and push image to registry
  pull        Pull image on remote server
  config      Sync config.yaml to remote
  start       Start/replace container on remote
  stop        Stop container
  restart     Restart container
  logs        Show container logs
  status      Show container status
  help        Show this help

Configuration (from DIST):
  HOST:  ${HOST:-not set}
  PORT:  ${PORT:-8012}
  IMAGE: ${IMAGE:-not set}

EOF
}

#=============================================================================
# Main
#=============================================================================

cd "${SCRIPT_DIR}"
load_config

case "${1:-}" in
    setup)    cmd_setup ;;
    push)     cmd_push ;;
    pull)     cmd_pull ;;
    config)   cmd_config ;;
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    restart)  cmd_restart ;;
    logs)     cmd_logs ;;
    status)   cmd_status ;;
    -h|--help|help) show_help ;;
    "")       cmd_deploy ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
