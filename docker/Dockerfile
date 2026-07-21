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
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# streaming 在线解码依赖：
#   opencv-python-headless —— cv2 兜底解码路径，服务器无显示环境用 headless 版免 libGL（必装）
#   decord                —— 首选解码器，编解码覆盖更广；无 wheel 的环境装不上也没关系，
#                            OV2 video_processing 是惰性 import、会自动退回 opencv。
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install opencv-python-headless && \
    ( pip install decord || pip install eva-decord || \
      echo "[warn] decord 装不上，将用 opencv 兜底解码（功能正常，仅编解码覆盖略窄）" )

# NGC 自带 nvidia-modelopt 会干扰 from_pretrained，卸掉（仓库官方 dockerfile 同样处理）
RUN pip uninstall -y nvidia-modelopt || true

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
