#!/usr/bin/env bash
# 一键构建镜像。用法:  bash docker/build.sh   (可选 IMAGE=xxx 覆盖镜像名)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lov2-stream:latest}"

cd "$REPO_ROOT"
echo "[build] context=$REPO_ROOT  image=$IMAGE"
DOCKER_BUILDKIT=1 docker build -t "$IMAGE" -f docker/Dockerfile .
echo "[build] done -> $IMAGE"
echo "[build] 验证 GPU:  docker run --rm --gpus all $IMAGE nvidia-smi"
