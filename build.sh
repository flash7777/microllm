#!/bin/bash
set -euo pipefail
IMAGE="${IMAGE:-codeberg.org/Gemini-Foundation/microllm}"
TAG="${TAG:-$(date +%Y%m%d-%H%M)}"

if command -v buildah &>/dev/null; then
    buildah bud --no-cache --network=host --security-opt label=disable -t "${IMAGE}:${TAG}" .
else
    podman build --no-cache --network=host --security-opt label=disable -t "${IMAGE}:${TAG}" .
fi

if [ -n "${PUSH_TOKEN:-}" ]; then
    buildah push --creds="token:${PUSH_TOKEN}" "${IMAGE}:${TAG}"
    buildah tag "${IMAGE}:${TAG}" "${IMAGE}:latest"
    buildah push --creds="token:${PUSH_TOKEN}" "${IMAGE}:latest"
fi
echo "=== Built: ${IMAGE}:${TAG} ==="
