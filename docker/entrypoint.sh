#!/usr/bin/env bash
# 容器入口。AIAK 路径 / PYTHONPATH / PYTORCH_CUDA_ALLOC_CONF 已在镜像 ENV 里设好
# （见 Dockerfile），docker exec 进来的 shell 也会继承，这里保持最小化。
set -e
exec "$@"
