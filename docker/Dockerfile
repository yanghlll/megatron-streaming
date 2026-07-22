# syntax=docker/dockerfile:1.4
# =============================================================================
# LLaVA-OneVision-2 Megatron + streaming video understanding 训练镜像
#
# 一键构建（在仓库根目录）:
#     DOCKER_BUILDKIT=1 docker build -t lov2-stream -f docker/Dockerfile .
#   或  bash docker/build.sh
#
# 底座用 NVIDIA NGC PyTorch —— 自带 TransformerEngine / Apex / flash-attn / cuDNN，
# Megatron-Core 训练必需这些（layer spec 走 te_spec、--attention-backend flash）。
# 换普通 pytorch 底座要自己编 TE+Apex，很折腾，所以沿用仓库官方底座。
# H20 驱动 575 / CUDA 12.9 与本镜像(CUDA 12.8 工具链)向前兼容。
# =============================================================================

FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_RETRIES=10 \
    PIP_TIMEOUT=120 \
    PIP_RESUME_RETRIES=5 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_ENABLE_HF_TRANSFER=0
# PIP_RESUME_RETRIES 需要 pip>=25.1（下面会升级）；大包下一半断了从断点续。

# NGC 镜像自带 /etc/pip/constraint.txt 会钉死某些包版本，装 requirements 时冲突 -> 清空
RUN if [ -f /etc/pip/constraint.txt ]; then cp /etc/pip/constraint.txt /etc/pip/constraint.txt.bak; : > /etc/pip/constraint.txt; fi

# 系统工具 + ffmpeg（streaming 数据 preprocessor 用 ffprobe 拿视频时长；
# 否则每行时长=0 -> 数据集全空）
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget vim tmux htop rsync openssh-client ca-certificates jq \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 先升级 pip（拿到断点续传能力）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel packaging ninja

# ── 缓存策略 ─────────────────────────────────────────────────────────
# 用 BuildKit cache mount 替代 --no-cache-dir：下载缓存留在宿主机构建缓存里，
# 不进镜像层（镜像不变大），但任何包成功下过一次，重 build 就本地秒装。
# 构建必须开 BuildKit：DOCKER_BUILDKIT=1（build.sh 已设）。
# ────────────────────────────────────────────────────────────────────

# 仓库训练依赖（megatron-energon==5.0.0 / transformers==5.7.0 / qwen_vl_utils / timm ...）
COPY requirements.txt /tmp/requirements.txt
# NGC 25.04 = Python 3.12：sentencepiece==0.1.98 没有 cp312 预编译 wheel，会回退源码编译并
# 因构建环境缺 cmake/C++ 依赖而失败。放宽到有 cp312 wheel 的 0.2.0（对 transformers/tokenizer
# 兼容，纯装 wheel 不编译）。只改临时文件，不动仓库 requirements.txt。
RUN --mount=type=cache,target=/root/.cache/pip \
    sed -i 's/^sentencepiece==0\.1\.98/sentencepiece>=0.2.0/' /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# streaming 在线解码依赖：
#   opencv-python-headless —— cv2 兜底解码路径，服务器无显示环境用 headless 版免 libGL（必装）
#   decord                —— 首选解码器，编解码覆盖更广；无 wheel 的环境装不上也没关系，
#                            OV2 video_processing 是惰性 import、会自动退回 opencv。
#
# 关键：NGC 底座预装的 opencv 是按 NumPy 1.x 编译的，在 NumPy 2.x 下 `import cv2` 会崩
# （numpy.core.multiarray failed to import）。且本栈里 scipy/pandas 等是按 NumPy 2.0 编的，
# 不能降 numpy。所以先卸掉所有 opencv 变体，再装 >=4.10.0.84（提供 NumPy 2 的 wheel）。
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y opencv-python opencv-python-headless \
                     opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true; \
    pip install "opencv-python-headless>=4.10.0.84" && \
    ( pip install decord || pip install eva-decord || \
      echo "[warn] decord 装不上，将用 opencv 兜底解码（功能正常，仅编解码覆盖略窄）" )

# NGC 自带 nvidia-modelopt 会干扰 from_pretrained，卸掉（仓库官方 dockerfile 同样处理）
RUN pip uninstall -y nvidia-modelopt || true

# 运行期环境：设进镜像 ENV，这样 `docker exec -it lov2-stream bash` 进来的 shell 也直接
# 带上 AIAK 路径 / PYTHONPATH（exec 不走 entrypoint，只有镜像 ENV 会被继承）。
# 仓库固定挂在 /workspace/LLaVA-OneVision-2（见 start.sh / docker-compose.yml）。
ENV AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2 \
    AIAK_MAGATRON_PATH=/workspace/LLaVA-OneVision-2/aiak_megatron \
    PYTHONPATH=/workspace/LLaVA-OneVision-2/aiak_megatron:/workspace/LLaVA-OneVision-2 \
    HF_HOME=/root/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
