#!/usr/bin/env bash
# 容器入口：若仓库被挂到 /workspace/LLaVA-OneVision-2，自动导出 AIAK 路径 + PYTHONPATH，
# 这样进容器就能直接跑 examples 下的训练脚本，无需再手动 export。
set -e

REPO_DEFAULT=/workspace/LLaVA-OneVision-2
if [ -d "${AIAK_TRAINING_PATH:-$REPO_DEFAULT}/aiak_training_llm" ]; then
    export AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-$REPO_DEFAULT}"
    export AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-$AIAK_TRAINING_PATH/aiak_megatron}"
    export PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:${PYTHONPATH:-}"
fi

# 显存碎片整理（长序列训练建议）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$@"
