# 一键环境搭建（Docker）— LLaVA-OneVision-2 Megatron + streaming

目标：在你的训练机（16×H20 = 2 机×8，驱动 575 / CUDA 12.9）上，用一条命令 build 好镜像、
起容器，然后照 [`../examples/llava_onevision2/quick_start_video_2b/STREAMING_RUNBOOK.md`](../examples/llava_onevision2/quick_start_video_2b/STREAMING_RUNBOOK.md)
跑训练。

> 底座是 NVIDIA NGC PyTorch（自带 TransformerEngine / Apex / flash-attn，Megatron-Core 必需）。
> 你的驱动 575 / CUDA 12.9 与镜像的 CUDA 12.8 工具链**向前兼容**，能跑。

---

## 0. 前提检查（先跑这条）

```bash
docker --version && \
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.04-py3 nvidia-smi | tail -3
```
- 能打印 16 张 H20 → docker + NVIDIA Container Toolkit 齐全，**跳到第 1 步**。
- `docker: command not found` 或 `--gpus` 报错 → 先看文末「附录：装 docker」。

---

## 1. 构建镜像（一键）

```bash
git clone git@github.com:yanghlll/megatron-streaming.git
cd megatron-streaming
bash docker/build.sh                 # = DOCKER_BUILDKIT=1 docker build -t lov2-stream -f docker/Dockerfile .
```
首次拉 NGC 底座 + 装依赖需要一会儿（底座 ~20GB）。build 完验证：
```bash
docker run --rm --gpus all lov2-stream nvidia-smi
```

镜像里已装好：仓库 `requirements.txt`（megatron-energon / transformers==5.7 / qwen_vl_utils …）
+ **ffmpeg**（转换脚本读时长）+ **opencv-headless / decord**（在线解码）。

---

## 2. 起容器

### 单节点（先在 1 台上验证）
```bash
DATA_ROOT=/你的数据根  CKPT_ROOT=/你的权重根  bash docker/run.sh
# 进容器后，AIAK_TRAINING_PATH / AIAK_MAGATRON_PATH / PYTHONPATH 已由 entrypoint 自动设好
```
> `video_path` 是绝对路径 → 数据/视频目录必须以**同样路径**挂进容器（run.sh 已按 `DATA_ROOT`/`CKPT_ROOT` 挂）。

### 两节点（你的 16×H20 = 2 机×8，正式训练）
1）编辑 `examples/llava_onevision2/quick_start_video_2b/instruct_video_streaming.sh` 顶部 `list_ip`，
填两台机 IP（顺序即 node rank，第 0 台是 master）：
```bash
declare -a list_ip=( "10.0.0.1"  "10.0.0.2" )
```
2）在**每台机**各起一个容器（`--network host` 让机间 NCCL 走宿主网络）：
```bash
docker run --gpus all --rm -it --network host --ipc host --shm-size 128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NCCL_SOCKET_IFNAME=<机间网卡,如 bond0/eth0>  -e MASTER_PORT=26000 \
  -v $(pwd):/workspace/LLaVA-OneVision-2 \
  -v /你的数据根:/你的数据根  -v /你的权重根:/你的权重根 \
  -w /workspace/LLaVA-OneVision-2  lov2-stream  bash
```

---

## 3. 容器内：转换权重 + 数据 + 训练

详细步骤见 [`STREAMING_RUNBOOK.md`](../examples/llava_onevision2/quick_start_video_2b/STREAMING_RUNBOOK.md)。速览：

```bash
# ① HF -> mcore（一次性，单卡）
bash examples/llava_onevision2/convert/convert_2b_hf_to_mcore.sh  <HF权重>  <mcore输出>  1 1

# ② 生成 streaming 数据（CPU，快，不抽帧）
python tools/data_preprocess/convert_streaming_to_webdataset.py \
  --jsonl <标注.jsonl> --output_dir <WebDataset输出> --video_root <视频目录> \
  --max_duration 230 --tail_margin 10 --num_workers 32

# ③ 训练（两台机各跑同一条；先冒烟 20 步）
#    位置参数: TP PP SEQ_LEN MBS GBS NSTEP
DATA_PATH=<WebDataset> TOKENIZER_PATH=<preprocessor> CHECKPOINT_PATH=<mcore> OUTPUT_DIR=<输出> \
STREAM_FPS=0 bash examples/llava_onevision2/quick_start_video_2b/instruct_video_streaming.sh \
    1 1 40000 1 16 20
# 冒烟通过后正式训练把末两参数换成:  224 3500
```

---

## 附录：装 docker（没装 / 无 GPU 支持时，需 sudo）

```bash
# Docker Engine
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER      # 重新登录一次生效

# NVIDIA Container Toolkit（--gpus all 必需）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.04-py3 nvidia-smi
```

> 共享 HPC 无 sudo / 不给 docker 的：在有 docker 的机器 `docker save lov2-stream -o lov2.tar`，
> 再用 `enroot import dockerd://lov2-stream` 或 `apptainer build lov2.sif docker-archive://lov2.tar` 免 root 跑。
