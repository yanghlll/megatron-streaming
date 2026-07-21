# 一键环境搭建（Docker）— LLaVA-OneVision-2 Megatron + streaming

范式与你之前那套 `start.sh` 一致：**一条命令 build 镜像 → 后台起容器 → 进 shell**，
compose 优先、无 compose 或用 `EXTRA_MOUNT` 时自动回退原生 docker。

> 底座 NVIDIA NGC PyTorch（自带 TransformerEngine / Apex / flash-attn，Megatron-Core 必需）。
> H20 驱动 575 / CUDA 12.9 与镜像的 CUDA 12.8 工具链**向前兼容**，能跑。

命名统一：镜像 `lov2-stream:ngc25.04`，容器 `lov2-stream`，仓库固定挂 `/workspace/LLaVA-OneVision-2`。

---

## 0. 前提检查（先跑这条）

```bash
docker --version && \
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.04-py3 nvidia-smi | tail -3
```
- 能打印 16 张 H20 → 环境就绪，跳到第 1 步。
- `docker: command not found` 或 `--gpus` 报错 → 看文末「附录：装 docker」。

---

## 1. 一键启动

```bash
git clone git@github.com:yanghlll/megatron-streaming.git
cd megatron-streaming

# 只验证环境（不挂数据）：
bash docker/start.sh

# 训练要挂数据/权重（EXTRA_MOUNT 里的目录以“相同路径”进容器；走原生 docker run 路径）：
EXTRA_MOUNT=/你的数据根,/你的权重根 bash docker/start.sh
```
`start.sh` 会：build 镜像 → 后台起容器 `lov2-stream`（`--gpus all --ipc host --network host
--restart unless-stopped`，挂仓库 + workdir + HF cache）→ `docker exec -it lov2-stream bash` 进入。

- **再次进入**：`docker exec -it lov2-stream bash`
- **停止/删除**：`docker rm -f lov2-stream`
- 进容器后 `AIAK_TRAINING_PATH` / `PYTHONPATH` 已由镜像 ENV 设好，直接跑脚本即可。

> compose 路径见 `docker/docker-compose.yml`；要在 compose 下挂数据就在它的 `volumes` 加一行。

---

## 2. 容器内：转换权重 + 数据 + 训练

详见 [`../examples/llava_onevision2/quick_start_video_2b/STREAMING_RUNBOOK.md`](../examples/llava_onevision2/quick_start_video_2b/STREAMING_RUNBOOK.md)。速览：

```bash
# ① HF -> mcore（一次性，单卡）
bash examples/llava_onevision2/convert/convert_2b_hf_to_mcore.sh  <HF权重>  <mcore输出>  1 1

# ② 生成 streaming 数据（CPU，快，不抽帧）
python tools/data_preprocess/convert_streaming_to_webdataset.py \
  --jsonl <标注.jsonl> --output_dir <WebDataset输出> --video_root <视频目录> \
  --max_duration 230 --tail_margin 10 --num_workers 32

# ③ 训练（先冒烟 20 步；位置参数: TP PP SEQ_LEN MBS GBS NSTEP）
DATA_PATH=<WebDataset> TOKENIZER_PATH=<preprocessor> CHECKPOINT_PATH=<mcore> OUTPUT_DIR=<输出> \
STREAM_FPS=0 bash examples/llava_onevision2/quick_start_video_2b/instruct_video_streaming.sh \
    1 1 40000 1 16 20
# 冒烟通过后正式训练把末两参数换成:  224 3500
```

### 两节点（16×H20 = 2 机×8）
1）编辑 `examples/llava_onevision2/quick_start_video_2b/instruct_video_streaming.sh` 顶部 `list_ip`，
填两台机 IP（顺序即 node rank，第 0 台是 master）。
2）在**每台机**上各跑一次 `EXTRA_MOUNT=... bash docker/start.sh` 起容器进 shell（容器已 `--network host`），
然后各自在容器内跑上面 ③ 的同一条训练命令。设机间网卡：容器里 `export NCCL_SOCKET_IFNAME=<bond0/eth0>`。

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

> 共享 HPC 无 sudo / 不给 docker：在有 docker 的机器 `docker save lov2-stream:ngc25.04 -o lov2.tar`，
> 再 `enroot import dockerd://lov2-stream:ngc25.04` 或 `apptainer build lov2.sif docker-archive://lov2.tar` 免 root 跑。
