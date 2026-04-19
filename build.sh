#!/bin/bash
#
# microllm - Build Script
#
# Usage:
#   ./build.sh              # Build container image
#   ./build.sh clean        # Remove image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_FILE="${SCRIPT_DIR}/DIST"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "\n${BLUE}=== $1 ===${NC}"; }

load_config() {
    if [[ -f "${DIST_FILE}" ]]; then
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^#.*$ ]] && continue
            [[ -z "$key" ]] && continue
            value=$(echo "$value" | sed 's/^["'\'']//' | sed 's/["'\'']$//')
            case "$key" in
                DOCKER_REGISTRY) DOCKER_REGISTRY="${value}" ;;
                DOCKER_NS) DOCKER_NAMESPACE="${value}" ;;
                IMAGE) IMAGE_NAME="${value}" ;;
            esac
        done < "${DIST_FILE}"
    fi

    GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "")
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

    if [[ -n "${GIT_TAG}" ]]; then
        VERSION="${GIT_TAG}"
    else
        VERSION="${GIT_BRANCH}-${GIT_HASH}"
    fi

    IMAGE="${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${IMAGE_NAME}"
}

build_docker() {
    log_step "Building microllm Image"

    podman build \
        -t "${IMAGE}:${VERSION}" \
        -t "${IMAGE}:${GIT_HASH}" \
        -t "${IMAGE}:latest" \
        "${SCRIPT_DIR}"

    log_info "Built: ${IMAGE}:${VERSION}"
}

clean() {
    log_step "Cleaning"
    podman rmi "${IMAGE}:latest" 2>/dev/null || true
    log_info "Clean complete"
}

cd "${SCRIPT_DIR}"
load_config

case "${1:-}" in
    clean)    clean ;;
    -h|--help|help) echo "Usage: $0 [clean]" ;;
    "")       build_docker ;;
    *)        log_error "Unknown: $1"; exit 1 ;;
esac
