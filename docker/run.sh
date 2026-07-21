#!/usr/bin/env bash
# 起一个交互容器（单节点快速验证用），挂载仓库 + 数据 + 权重目录。
# 用法:  DATA_ROOT=/data CKPT_ROOT=/ckpt bash docker/run.sh
#
# 注意：WebDataset 里 video_path 是绝对路径，所以数据/视频目录必须以“同样的路径”挂进容器。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lov2-stream:latest}"
NAME="${NAME:-lov2-stream}"

DATA_ROOT="${DATA_ROOT:-/data}"   # 标注 jsonl / 原始视频 / streaming WebDataset 根目录
CKPT_ROOT="${CKPT_ROOT:-/ckpt}"   # HF 权重 / mcore 权重 / preprocessor / 输出

MOUNTS=(-v "$REPO_ROOT":/workspace/LLaVA-OneVision-2)
[ -d "$DATA_ROOT" ] && MOUNTS+=(-v "$DATA_ROOT":"$DATA_ROOT")
[ -d "$CKPT_ROOT" ] && MOUNTS+=(-v "$CKPT_ROOT":"$CKPT_ROOT")

exec docker run --gpus all --rm -it --name "$NAME" \
  --network host --ipc host --shm-size 128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  "${MOUNTS[@]}" \
  -w /workspace/LLaVA-OneVision-2 \
  "$IMAGE" bash
