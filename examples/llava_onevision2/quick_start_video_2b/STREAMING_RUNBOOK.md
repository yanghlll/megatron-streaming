# Streaming Video SFT — 目标机运行 runbook (16×H20 = 2 机×8, docker+torchrun)

改动代码见 `instruct_video_streaming.sh` + `tools/data_preprocess/convert_streaming_to_webdataset.py`
+ task encoder / loss 管线。本文件是**在目标机上跑起来**的步骤。

环境：Docker `nvcr.io/nvidia/pytorch:25.04-py3`（CUDA 12.8 工具链，H20 驱动 12.9 向前兼容）。
并行：2B 模型很小 → **TP=1 PP=1 DP=16**。

---

## 0. 前提：代码 + 数据 + 权重在两台机的**共享存储**上（NFS），路径一致

需要就位：
- 本仓库 `LLaVA-OneVision-2/`（含改动）
- JoyAI 标注 jsonl + 原始视频目录
- OV2 preprocessor 目录（tokenizer + processor，即 `TOKENIZER_PATH`）
- LOV2-2B 的 **HuggingFace** 权重（`HF_CKPT`）

> `video_path` 在数据里是绝对路径 → 两台机 + 容器内都要能按同一路径读到视频（用 `-v` 挂进容器）。

---

## 1. 容器：补 decord + ffmpeg（在线解码 + 转换脚本必需）

`requirements.txt` 没有 decord/opencv/ffmpeg，而 streaming 在线解码走 OV2 的
`video_processing_llava_onevision2.py`（decord 首选、cv2 兜底），转换脚本读时长用 ffprobe。
建议在镜像里 bake（追加到 `dockerfile` 末尾再 build）：

```dockerfile
RUN pip install --no-cache-dir decord && \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*
```

```bash
docker build -t lov2-stream -f dockerfile .
```

---

## 2. HF → mcore 转换（一次性，TP=1 PP=1，与训练一致）

在**任一台**机的容器里（单卡即可）：

```bash
docker run --gpus '"device=0"' --rm -it --ipc host --shm-size 64g \
  -v /shared:/shared lov2-stream bash

# 容器内：
export AIAK_TRAINING_PATH=/shared/LLaVA-OneVision-2
export AIAK_MAGATRON_PATH=$AIAK_TRAINING_PATH/aiak_megatron
cd $AIAK_TRAINING_PATH
bash examples/llava_onevision2/convert/convert_2b_hf_to_mcore.sh \
    /shared/ckpt/LLaVA-OneVision-2-2B-hf \
    /shared/ckpt/lov2_2b_mcore_tp1pp1 \
    1 1
# 产出: /shared/ckpt/lov2_2b_mcore_tp1pp1/release/... + latest_checkpointed_iteration.txt
```

> `CHECKPOINT_PATH` = `/shared/ckpt/lov2_2b_mcore_tp1pp1`（训练 `--load` 用）。

---

## 3. 生成 streaming 数据（一次性，CPU，容器内）

```bash
python tools/data_preprocess/convert_streaming_to_webdataset.py \
  --jsonl      /shared/data/joyai_annotations.jsonl \
  --output_dir /shared/data/joy_streaming_webdataset \
  --video_root /shared/data/videos \
  --max_duration 230 --tail_margin 10 --num_workers 32
```

只读时长、不抽帧，快。产出 energon WebDataset（视频按路径引用），即 `DATA_PATH`。
改 fps 无需重转数据（fps 是训练时参数）。

---

## 4. 预检（容器内，2 条，都应通过）

```bash
export TOKENIZER_PATH=/shared/ckpt/ov2_preprocessor
# ① 词表余量：加 2 个 token 后 len 应 < 151936 (2B 的 padded vocab) → 无需 resize
python -c "from transformers import AutoTokenizer as A; t=A.from_pretrained('$TOKENIZER_PATH'); print('added',t.add_special_tokens({'additional_special_tokens':['</silence>','</response>']}),'len',len(t))"
# ② 视频处理器可用（在线解码依赖 .video_processor）
python -c "from transformers import AutoProcessor as P; p=P.from_pretrained('$TOKENIZER_PATH',trust_remote_code=True); print('video_processor:',type(getattr(p,'video_processor',None)).__name__)"
```
预期：① `len` ~151671 (<151936)；② 打印出 `LlavaOnevision2VideoProcessor`（若为 `NoneType` 说明 preprocessor 目录缺 video processor 的 remote code，需补上再训）。

---

## 5. 两机启动（docker run + torchrun）

**编辑 `instruct_video_streaming.sh` 顶部的 `list_ip`**，填两台机的 IP（顺序即 node rank，node0 是 master）：
```bash
declare -a list_ip=(
    "10.0.0.1"   # node0 (master)
    "10.0.0.2"   # node1
)
```
`GPUS_PER_NODE=8` 已是默认，无需改。

在**每台机**上各起一个容器并跑**同一条命令**（脚本按本机 IP 自动判定 node rank）：
```bash
docker run --gpus all --rm -it --network host --ipc host --shm-size 128g \
  -e NCCL_SOCKET_IFNAME=<机间网卡名,如 bond0/eth0>  \
  -e MASTER_PORT=26000 \
  -v /shared:/shared \
  -v /shared/data/videos:/shared/data/videos \
  lov2-stream bash

# 容器内（两台都执行）：
export AIAK_TRAINING_PATH=/shared/LLaVA-OneVision-2
export AIAK_MAGATRON_PATH=$AIAK_TRAINING_PATH/aiak_megatron
export DATA_PATH=/shared/data/joy_streaming_webdataset
export TOKENIZER_PATH=/shared/ckpt/ov2_preprocessor
export CHECKPOINT_PATH=/shared/ckpt/lov2_2b_mcore_tp1pp1
export OUTPUT_DIR=/shared/output
cd $AIAK_TRAINING_PATH

# 位置参数: TP PP SEQ_LEN MBS GBS NSTEP
# ---- 先 20 步冒烟(小 GBS 让 20 步快) ----
STREAM_FPS=0 bash examples/llava_onevision2/quick_start_video_2b/instruct_video_streaming.sh \
    1 1 40000 1 16 20
```

DP=16、GBS=16 → grad_accum=1（冒烟快）。冒烟看日志：loss 有限且下降、无 NCCL 卡死、无 assert。

**跑通后正式训练**（GBS=224 → grad_accum=14，`--train-iters` 设你的总步数）：
```bash
STREAM_FPS=0 bash examples/llava_onevision2/quick_start_video_2b/instruct_video_streaming.sh \
    1 1 40000 1 224 3500
```

---

## 6. 调参 / 排障

- **显存**：TP=1 下 2B + 40k 序列 + `--recompute-granularity full` 在 96G 够用。若长视频 OOM：
  降 `--max-pixels`（每帧 token）→ 降 `--seq-length` → 降 `STREAM_FPS` / `--max_duration`（转换时）。
- **每帧 token 预算**：在 `instruct_video_streaming.sh` 的 `DATA_ARGS` 里加 `--max-pixels <N>`
  （类比 ms-swift 的 64~128 tok/帧；OV2 sms=3，每帧 token = H*W/9）。
- **解码吞吐**（在线解码是 CPU 瓶颈）：`--num-workers` 已 16；CPU 富余可加大；decord 已装。
- **序列超长被丢**：`--enable-discard-sample` 时超 `--seq-length` 的样本会被跳过（日志有计数）；
  想留就调大 seq-length 或降 fps/max_pixels。
- **多机 NCCL 卡住**：确认 `--network host` + 正确的 `NCCL_SOCKET_IFNAME`（机间网卡）；
  IB 环境可加 `-e NCCL_IB_HCA=...`。两机 `MASTER_PORT` 一致且 node0 端口可达。
- **断点续训**：`--load $CHECKPOINT_PATH` 会自动接 `$OUTPUT_DIR` 下最新 checkpoint；
  dataloader 状态也会存/恢复（`--dataloader-save`）。

## 7. 验证权重在学（可选）
训练几步后，`</silence>`/`</response>` 的 embedding 行应有非零梯度（三模块全训）。
loss 曲线里控制 token 的加权（0.4/1.5）会体现为对静默/响应时刻的不同惩罚。
