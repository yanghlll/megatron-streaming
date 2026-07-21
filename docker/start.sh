#!/usr/bin/env bash
# 一键启动：build 镜像 → 后台起容器 → 进入 shell。
# 自动检测 docker compose v2；没有则回退原生 docker build/run，不依赖 compose。
# （范式与 ms-swift 那套 start.sh 一致）
set -euo pipefail
cd "$(dirname "$0")"                       # docker/
REPO_ROOT="$(cd .. && pwd)"               # 仓库根 = 训练代码

# 启用 BuildKit：Dockerfile 的 --mount=type=cache 缓存语法需要它
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

IMAGE=lov2-stream:ngc25.04                 # 与 docker-compose.yml 保持一致
NAME=lov2-stream

# 额外挂载的宿主机目录（逗号分隔），以相同路径出现在容器内（数据/权重必须同路径，
# 因为 WebDataset 里 video_path 是绝对路径）：
#   EXTRA_MOUNT=/data,/ckpt bash docker/start.sh
EXTRA_ARGS=()
if [ -n "${EXTRA_MOUNT:-}" ]; then
    IFS=',' read -ra _dirs <<< "${EXTRA_MOUNT}"
    for d in "${_dirs[@]}"; do
        EXTRA_ARGS+=(-v "${d}:${d}")
        echo ">>> Extra mount: ${d} -> ${d}"
    done
fi

mkdir -p "${REPO_ROOT}/workdir" "${HOME}/.cache/huggingface"

if docker compose version >/dev/null 2>&1 && [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    echo ">>> Building image & starting container (docker compose) ..."
    echo ">>> （需要额外挂载数据/权重目录时，编辑 docker-compose.yml 的 volumes，或用 EXTRA_MOUNT=... 走 docker run 路径）"
    docker compose up -d --build
else
    echo ">>> docker compose v2 不可用或用了 EXTRA_MOUNT，走原生 docker build/run ..."
    docker build -t "${IMAGE}" -f Dockerfile "${REPO_ROOT}"
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
    docker run -d --name "${NAME}" \
        --gpus all \
        --ipc host \
        --network host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --restart unless-stopped \
        -e HF_HOME=/root/.cache/huggingface \
        -v "${REPO_ROOT}:/workspace/LLaVA-OneVision-2" \
        -v "${REPO_ROOT}/workdir:/workspace/workdir" \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        "${IMAGE}"
fi

echo ">>> Entering container (再次进入可执行: docker exec -it ${NAME} bash)"
docker exec -it "${NAME}" bash
