#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert JoyAI streaming-video annotations to a lightweight WebDataset.

Each input JSONL row (JoyAI-VL-Interaction style)::

    {"video_name", "video_path", "task_type", "source",
     "question": [{"content", "time"}], "response": [{"content", "time"}]}

is turned into per-second interleaved turns **without extracting frames**. The
video stays referenced by absolute path (``video_path``) and is decoded ONLINE at
train time by ``Qwen2VLTaskEncoder.encode_streaming_video`` (fps chosen via
``--stream-fps``). Storage is ~annotation-only; the original videos are reused.

Per second ``sec`` in ``[0, n_seconds)``:
  - user turn:      ``"[<question>\\n]<sec.0 seconds>\\n<|video_pad|>"`` — exactly one
                    ``<|video_pad|>`` sentinel; the encoder splices that second's
                    decoded frame tokens in its place.
  - assistant turn: ``"</response> <answer>"`` if a response fires at ``sec``, else
                    ``"</silence>"``.

Ported from ms-swift ``swift/dataset/preprocessor/streaming_video.py``
(``JoyStreamingVideoPreprocessor``), minus the frame extraction.

Output is a Megatron-Energon WebDataset with ``sample_type = MultiMixQASample`` and
a streaming ``sample_loader`` that fills ``messages`` + ``video_path``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import webdataset as wds
import yaml
from tqdm import tqdm

try:
    from megatron.energon.epathlib import EPath
    from megatron.energon.flavors import BaseWebdatasetFactory
    from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME

    ENERGON_AVAILABLE = True
except ImportError:
    ENERGON_AVAILABLE = False
    MAIN_FOLDER_NAME = ".nv-meta"

logger = logging.getLogger(__name__)

# Reuse the model's own <|video_pad|> token as the per-second sentinel: it is a
# single vocab token and the encoder replaces it with that second's frame blocks.
STREAM_FRAME_TAG = "<|video_pad|>"


# --------------------------------------------------------------------------- loader
def sample_loader_template() -> str:
    """Return sample_loader.py content for streaming MultiMixQASample (video by path)."""
    return """# Auto-generated sample loader for streaming video (online decode).


def sample_loader(sample: dict) -> dict:
	data = sample['json']

	system = None
	messages = []
	for msg in data.get('messages', []):
		if msg.get('role') == 'system':
			system = msg.get('content')
			continue
		messages.append({'role': msg.get('role'), 'content': msg.get('content')})

	return dict(
		__key__=sample['__key__'],
		__restore_key__=sample['__restore_key__'],
		messages=messages,
		system=system,
		video_path=data.get('video_path'),
	)


def part_filter(part: str) -> bool:
	return True
"""


def sample_loader_template_offline() -> str:
    """sample_loader.py for offline-frame streaming (frames stored in the shard)."""
    return """# Auto-generated sample loader for streaming video (offline pre-extracted frames).


def sample_loader(sample: dict) -> dict:
	data = sample['json']

	# energon decodes img_*.jpg entries to PIL (image_decode="pil")
	images = [sample.get(name) for name in data.get('image_keys', [])]

	system = None
	messages = []
	for msg in data.get('messages', []):
		if msg.get('role') == 'system':
			system = msg.get('content')
			continue
		messages.append({'role': msg.get('role'), 'content': msg.get('content')})

	return dict(
		__key__=sample['__key__'],
		__restore_key__=sample['__restore_key__'],
		messages=messages,
		system=system,
		image=images if len(images) > 0 else None,
		bucket_counts=data.get('bucket_counts'),
		fps=data.get('fps'),
	)


def part_filter(part: str) -> bool:
	return True
"""


def write_config(output_dir: str, offline: bool = False) -> None:
    """Write minimal Energon config (online video-path loader or offline-frame loader)."""
    meta_dir = Path(output_dir) / MAIN_FOLDER_NAME
    meta_dir.mkdir(parents=True, exist_ok=True)

    dataset_definition = {
        "sample_type": {
            "__module__": "aiak_training_llm.data.multimodal",
            "__class__": "MultiMixQASample",
        },
        "part_filter": "sample_loader.py:part_filter",
        "sample_loader": "sample_loader.py:sample_loader",
    }
    with (meta_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(dataset_definition, f, sort_keys=False)
    with (meta_dir / "sample_loader.py").open("w", encoding="utf-8") as f:
        f.write(sample_loader_template_offline() if offline else sample_loader_template())

    if ENERGON_AVAILABLE:
        path = EPath(output_dir).absolute()
        all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
        all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]
        BaseWebdatasetFactory.prepare_dataset(
            path,
            all_tars,
            split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
            tar_index_only=False,
            workers=min(96, os.cpu_count() or 1),
        )


# ---------------------------------------------------------------- streaming preprocess
def _ffprobe_duration(video_path: str) -> float:
    """Video duration in seconds: ffprobe first, opencv fallback, 0.0 on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True,
        )
        d = float(result.stdout.strip())
        if d > 0:
            return d
    except (ValueError, AttributeError, OSError):
        pass
    # opencv fallback — lazily/​once imported and guarded: a NumPy-1.x-built cv2 raises
    # ImportError under NumPy 2, which we swallow so it never crashes or spams per row.
    cv2 = _get_cv2()
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(video_path)
            try:
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
            finally:
                cap.release()
            if frames > 0 and fps > 0:
                return float(frames) / float(fps)
        except Exception:
            pass
    return 0.0


def _parse_times(time_str) -> list:
    """'8' or '5,6,7' -> [8] / [5,6,7]."""
    if not time_str and time_str != 0:
        return []
    return [int(float(t.strip())) for t in str(time_str).split(",") if str(t).strip()]


def _adaptive_fps(duration: float) -> float:
    if duration >= 160:
        return 1.0
    if duration >= 64:
        return 2.0
    return 4.0


# opencv is imported at most once per process, guarded: if the installed cv2 was built
# against NumPy 1.x it can't import under NumPy 2 -> we set _CV2=None and skip the fallback
# instead of crashing / printing a NumPy warning+stack on every single row.
_CV2 = "unset"


def _get_cv2():
    global _CV2
    if _CV2 == "unset":
        try:
            import cv2 as _c
            _CV2 = _c
        except Exception:
            _CV2 = None
            print("[info] opencv 不可用（可能是 NumPy 1.x/2.x ABI 问题）；只用 ffprobe 读时长。",
                  file=sys.stderr)
    return _CV2


_WARN_COUNTS: dict = {}


def _warn_once(key: str, msg: str, limit: int = 5) -> None:
    n = _WARN_COUNTS.get(key, 0) + 1
    _WARN_COUNTS[key] = n
    if n <= limit:
        print(f"[skip] {msg}", file=sys.stderr)
        if n == limit:
            print(f"[skip] (further '{key}' messages suppressed)", file=sys.stderr)


def _print_skip_summary(warn_agg: dict) -> None:
    """Print a per-reason breakdown of skipped rows so it's clear WHY they were dropped."""
    if not warn_agg:
        return
    total = sum(warn_agg.values())
    reasons = {
        "no_video_path": "无 video_path",
        "video_not_found": "视频文件找不到（路径/挂载问题）",
        "duration_zero": "读不到时长（ffprobe/opencv 失败、文件损坏）",
        "event_beyond_video": "Q/A 时间戳超过视频时长（标注与视频不匹配）",
        "all_responses_outside_window": f"所有 response 在 max_duration 之外（按设计丢弃，避免全 </silence>）",
        "extract_no_frames": "抽帧得到 0 帧",
    }
    print(f"\n[skip breakdown] 共跳过 {total} 条，按原因：", file=sys.stderr)
    for key, cnt in sorted(warn_agg.items(), key=lambda kv: -kv[1]):
        print(f"    {cnt:>8}  {key}  —— {reasons.get(key, '')}", file=sys.stderr)
    print("    说明：'all_responses_outside_window' 属正常按设计丢弃；max_duration 越小丢得越多。",
          file=sys.stderr)


def _report_zero(kept: int) -> None:
    if kept == 0:
        print(
            "\n[WARN] kept=0 —— 所有样本都被跳过！最可能的原因:\n"
            "   (1) video 路径不对: --video_root + jsonl 里的 video_path 拼不出真实文件;\n"
            "   (2) 容器内没有 ffprobe(ffmpeg 没装)。\n"
            "   手动验证一条(把 <video> 换成一个真实路径):\n"
            "       which ffprobe && ffprobe -v quiet -show_entries format=duration "
            "-of default=nk=1:nw=1 <video>",
            file=sys.stderr,
        )


def preprocess_row(row: dict, video_root, max_duration, tail_margin) -> dict | None:
    """JoyAI row -> {messages, video_path}. None => skip the row."""
    video_path = row.get("video_path") or (row.get("videos") or [None])[0]
    if not video_path:
        _warn_once("no_video_path", f"row has no video_path: id={row.get('id') or row.get('video_name')}")
        return None
    if video_root and not os.path.isabs(video_path) and not str(video_path).startswith("http"):
        video_path = os.path.join(video_root, video_path)

    is_remote = str(video_path).startswith("http")
    if not is_remote and not os.path.exists(video_path):
        _warn_once("video_not_found",
                   f"video not found: {video_path}  (检查 --video_root 和 jsonl 里的 video_path)")
        return None

    duration = _ffprobe_duration(video_path)
    if duration <= 0:
        _warn_once("duration_zero",
                   f"读不到时长(ffprobe+opencv 都失败): {video_path}  "
                   f"(确认容器内有 ffprobe: `which ffprobe`，或视频文件可读)")
        return None

    if max_duration and max_duration > 0:
        truncated = duration > max_duration
        effective_duration = min(duration, max_duration)
    else:
        truncated = False
        effective_duration = duration
    n_seconds = max(int(effective_duration), 1)

    question_map: dict = {}
    for q in row.get("question") or []:
        for t in _parse_times(q.get("time")):
            if t > effective_duration:
                if truncated:
                    continue
                _warn_once("event_beyond_video",
                           f"question time {t}s > 视频时长 {effective_duration:.0f}s（未截断，疑似标注/视频不匹配）: {video_path}")
                return None
            question_map[min(t, n_seconds - 1)] = q["content"]

    response_map: dict = {}
    raw_responses = row.get("response") or []
    flat: list = []
    for item in raw_responses:
        flat.extend(item) if isinstance(item, list) else flat.append(item)
    for r in flat:
        for t in _parse_times(r.get("time")):
            if t > effective_duration:
                if truncated:
                    continue
                _warn_once("event_beyond_video",
                           f"response time {t}s > 视频时长 {effective_duration:.0f}s（未截断，疑似标注/视频不匹配）: {video_path}")
                return None
            response_map[min(t, n_seconds - 1)] = r["content"]

    # All responses fell outside the window -> would become all-</silence> -> drop.
    if raw_responses and not response_map:
        _warn_once("all_responses_outside_window",
                   f"所有 response 都超出 max_duration={max_duration}s（会退化成全 </silence>，故丢弃）: {video_path}")
        return None

    # Trim trailing pure-silence tail beyond the last event + margin.
    if tail_margin is not None:
        events = list(question_map) + list(response_map)
        if events:
            n_seconds = min(n_seconds, max(events) + tail_margin + 1)

    messages: list = []
    for sec in range(n_seconds):
        parts = []
        if sec in question_map:
            parts.append(question_map[sec])
        parts.append(f"<{sec:.1f} seconds>")
        parts.append(STREAM_FRAME_TAG)  # exactly one sentinel per second
        messages.append({"role": "user", "content": "\n".join(parts)})
        if sec in response_map:
            messages.append({"role": "assistant", "content": f"</response> {response_map[sec]}"})
        else:
            messages.append({"role": "assistant", "content": "</silence>"})

    return {"messages": messages, "video_path": video_path}


# ------------------------------------------------------------------------- io helpers
def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_frames(video_path, n_seconds, fps, frame_max_side=0):
    """Decode `video_path` at `fps`, bucket frames by integer second, keep [0, n_seconds).

    Returns (jpg_bytes_list, bucket_counts) where frames are in temporal order and
    bucket_counts[sec] = #frames stored for that second (sum == len(jpg_bytes_list)).
    decord preferred, OpenCV fallback. jpg_bytes are what gets stored in the shard.
    """
    import io

    import numpy as np
    from PIL import Image

    sampled = []  # list of (sec:int, rgb ndarray)
    try:
        import decord

        vr = decord.VideoReader(video_path)
        src_fps = float(vr.get_avg_fps() or 30.0)
        total = len(vr)
        duration = total / src_fps if src_fps > 0 else float(n_seconds)
        horizon = min(duration, float(n_seconds)) if duration > 0 else float(n_seconds)
        ts = np.arange(0.0, horizon, 1.0 / fps)
        idxs = np.clip(np.floor(ts * src_fps).astype(np.int64), 0, max(total - 1, 0))
        batch = vr.get_batch(idxs.tolist()).asnumpy()  # [K,H,W,3] RGB
        sampled = [(int(ts[k]), batch[k]) for k in range(len(idxs))]
    except Exception:
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = total / src_fps if src_fps > 0 else float(n_seconds)
            horizon = min(duration, float(n_seconds)) if duration > 0 else float(n_seconds)
            for t in np.arange(0.0, horizon, 1.0 / fps):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * src_fps))
                ok, fr = cap.read()
                if ok:
                    sampled.append((int(t), cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            cap.release()
        except Exception:
            return [], [0] * n_seconds

    bucket_counts = [0] * n_seconds
    jpgs = []
    for sec, arr in sampled:
        if sec < 0 or sec >= n_seconds:
            continue
        img = Image.fromarray(arr)
        if frame_max_side and max(img.size) > frame_max_side:
            img.thumbnail((frame_max_side, frame_max_side))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        jpgs.append(buf.getvalue())
        bucket_counts[sec] += 1
    return jpgs, bucket_counts


def _build_sample(entry, idx, sample_prefix, video_root, max_duration, tail_margin, xopts):
    """Build one WebDataset sample dict, or None to skip.

    xopts = (extract_frames: bool, stream_fps: float, frame_max_side: int).
    """
    processed = preprocess_row(entry, video_root, max_duration, tail_margin)
    if processed is None:
        return None
    sample_id = entry.get("id") or entry.get("video_name") or f"{sample_prefix}{idx}"
    sample_id = str(sample_id).replace(".", "_").replace("/", "_")
    messages = processed["messages"]
    video_path = processed["video_path"]

    extract_frames, stream_fps, frame_max_side = xopts
    if not extract_frames:
        payload = {"messages": messages, "video_path": video_path}
        return {"__key__": str(sample_id), "json": json.dumps(payload, ensure_ascii=False).encode("utf-8")}

    # offline mode: extract frames now, store them in the shard + bucket_counts.
    n_seconds = sum(m.get("content", "").count(STREAM_FRAME_TAG) for m in messages)
    fps = stream_fps if stream_fps and stream_fps > 0 else _adaptive_fps(float(n_seconds))
    jpgs, bucket_counts = _extract_frames(video_path, n_seconds, fps, frame_max_side)
    if sum(bucket_counts) == 0:
        _warn_once("extract_no_frames", f"extracted 0 frames in [0,{n_seconds}): {video_path}")
        return None
    sample = {"__key__": str(sample_id)}
    image_keys = []
    for i, b in enumerate(jpgs):
        key = f"img_{i:06d}.jpg"
        sample[key] = b
        image_keys.append(key)
    payload = {
        "messages": messages,
        "image_keys": image_keys,
        "bucket_counts": bucket_counts,
        "fps": fps,
    }
    sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return sample


_COUNTER = None
_KEPT = None


def _init_worker(counter, kept) -> None:
    global _COUNTER, _KEPT
    _COUNTER = counter
    _KEPT = kept


def _process_chunk(src_files, tar_pattern, maxcount, maxsize, sample_prefix, worker_id,
                   video_root, max_duration, tail_margin, xopts):
    """Process one or more source jsonl files into a shard set (tar_pattern)."""
    if isinstance(src_files, str):
        src_files = [src_files]
    count = 0
    kept = 0
    with wds.ShardWriter(tar_pattern, maxcount=maxcount, maxsize=maxsize, verbose=0) as shard_writer:
        idx = 0
        for src in src_files:
            for entry in iter_jsonl(src):
                sample = _build_sample(entry, idx, sample_prefix, video_root, max_duration, tail_margin, xopts)
                idx += 1
                count += 1
                if sample is not None:
                    shard_writer.write(sample)
                    kept += 1
                if _COUNTER is not None and count % 50 == 0:
                    with _COUNTER.get_lock():
                        _COUNTER.value += 50
        if _COUNTER is not None:
            remainder = count % 50
            if remainder:
                with _COUNTER.get_lock():
                    _COUNTER.value += remainder
    if _KEPT is not None:
        with _KEPT.get_lock():
            _KEPT.value += kept
    return worker_id, count, kept, dict(_WARN_COUNTS)


def collect_jsonl_paths(input_path: str) -> list:
    """Resolve input to a list of .jsonl files.

    Accepts:
      - a single file:  /data/a.jsonl
      - a directory:    /data/anno         (recursively globs **/*.jsonl, incl. subdirs)
      - a glob pattern: '/data/**/*.jsonl' or '/data/*.jsonl'
    """
    import glob as _glob

    if os.path.isdir(input_path):
        files = _glob.glob(os.path.join(input_path, "**", "*.jsonl"), recursive=True)
    elif any(c in input_path for c in "*?["):
        files = _glob.glob(input_path, recursive=True)
    else:
        files = [input_path]
    files = sorted(f for f in files if os.path.isfile(f) and f.endswith(".jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl found under: {input_path}")
    return files


def _count_lines(paths) -> int:
    if isinstance(paths, str):
        paths = [paths]
    total = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


def _split_jsonl(files: list, num_workers: int, tmp_dir: str):
    """Merge all input files' lines and split evenly into <= num_workers chunk files."""
    total_lines = _count_lines(files)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    if total_lines == 0:
        empty_path = os.path.join(tmp_dir, "chunk-00.jsonl")
        Path(empty_path).write_text("", encoding="utf-8")
        return [empty_path], 0

    per_chunk = max(1, math.ceil(total_lines / num_workers))
    chunk_paths = [os.path.join(tmp_dir, f"chunk-{i:02d}.jsonl") for i in range(num_workers)]
    writers = [open(p, "w", encoding="utf-8") for p in chunk_paths]
    counts = [0] * num_workers
    current = 0
    written = 0
    for src in files:
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if written >= per_chunk and current < num_workers - 1:
                    current += 1
                    written = 0
                writers[current].write(line if line.endswith("\n") else line + "\n")
                written += 1
                counts[current] += 1
    for w in writers:
        w.close()
    return [p for p, c in zip(chunk_paths, counts) if c > 0], total_lines


def convert(input_path, output_dir, maxcount, maxsize, num_workers, keep_chunks,
            video_root, max_duration, tail_margin, extract_frames=False,
            stream_fps=0.0, frame_max_side=0):
    files = collect_jsonl_paths(input_path)
    mode = "OFFLINE frames (decoded now, stored in shard)" if extract_frames else "ONLINE (video path)"
    print(f"[collect] {len(files)} jsonl file(s) under {input_path}  |  mode: {mode}")
    os.makedirs(output_dir, exist_ok=True)
    xopts = (extract_frames, stream_fps, frame_max_side)

    if num_workers <= 1:
        tar_pattern = os.path.join(output_dir, "instruct_%06d.tar")
        counter = mp.Value("i", 0)
        kept = mp.Value("i", 0)
        _init_worker(counter, kept)
        with tqdm(total=_count_lines(files)) as pbar:
            _, count, n_kept, warn = _process_chunk(
                files, tar_pattern, maxcount, maxsize, "sample_", 0,
                video_root, max_duration, tail_margin, xopts)
            pbar.n = count
            pbar.refresh()
        print(f"total={count} kept={n_kept} skipped={count - n_kept}")
        _print_skip_summary(warn)
        _report_zero(n_kept)
    else:
        tmp_dir = os.path.join(output_dir, ".tmp_jsonl_chunks")
        chunk_paths, total_lines = _split_jsonl(files, num_workers, tmp_dir)
        worker_count = len(chunk_paths)
        ctx = mp.get_context("fork")
        counter = ctx.Value("i", 0)
        kept = ctx.Value("i", 0)
        with ctx.Pool(processes=worker_count, initializer=_init_worker, initargs=(counter, kept)) as pool:
            args_list = []
            for worker_id, chunk_path in enumerate(chunk_paths):
                tar_pattern = os.path.join(output_dir, f"instruct_{worker_id:02d}_%06d.tar")
                args_list.append(([chunk_path], tar_pattern, maxcount, maxsize,
                                  f"sample_{worker_id:02d}_", worker_id,
                                  video_root, max_duration, tail_margin, xopts))
            results_async = pool.starmap_async(_process_chunk, args_list)
            with tqdm(total=total_lines) as pbar:
                while not results_async.ready():
                    with counter.get_lock():
                        pbar.n = counter.value
                    pbar.refresh()
                    time.sleep(0.5)
                results = results_async.get()
                with counter.get_lock():
                    pbar.n = counter.value
                pbar.refresh()
        total = sum(c for _, c, _, _ in results)
        total_kept = sum(k for _, _, k, _ in results)
        warn_agg: dict = {}
        for _, _, _, w in results:
            for key, cnt in (w or {}).items():
                warn_agg[key] = warn_agg.get(key, 0) + cnt
        print(f"total={total} kept={total_kept} skipped={total - total_kept}")
        _print_skip_summary(warn_agg)
        _report_zero(total_kept)
        if not keep_chunks:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    write_config(output_dir, offline=extract_frames)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True,
                   help="Input JoyAI streaming annotations: a single .jsonl file, a DIRECTORY "
                        "(recursively globs **/*.jsonl, including subdirs), or a glob pattern "
                        "(e.g. '/data/**/*.jsonl').")
    p.add_argument("--output_dir", required=True, help="Output dir for WDS shards")
    p.add_argument("--video_root", default=None, help="Root dir for relative video_path")
    p.add_argument("--max_duration", type=int, default=320, help="Time-axis cap in seconds (0=off)")
    p.add_argument("--tail_margin", type=int, default=None,
                   help="Keep this many silent seconds after the last event (None=keep all)")
    p.add_argument("--maxcount", type=int, default=10000, help="Max samples per shard")
    p.add_argument("--maxsize", type=int, default=3_000_000_000, help="Max shard size in bytes")
    p.add_argument("--num_workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--keep_chunks", action="store_true")
    # offline-frame mode: decode + store frames now; training reads frames, no online decode.
    p.add_argument("--extract_frames", action="store_true",
                   help="Offline mode: decode videos now and store frames in the shard "
                        "(+ per-second bucket_counts). Training reads frames, no online decode.")
    p.add_argument("--stream_fps", type=float, default=0.0,
                   help="Extraction fps for --extract_frames. 0 = adaptive by duration "
                        "(>=160s->1, >=64s->2, else 4).")
    p.add_argument("--frame_max_side", type=int, default=0,
                   help="Optional: downscale extracted frames so max(H,W)<=this (0=off) to save disk.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        input_path=args.jsonl,
        output_dir=args.output_dir,
        maxcount=args.maxcount,
        maxsize=args.maxsize,
        num_workers=args.num_workers,
        keep_chunks=args.keep_chunks,
        video_root=args.video_root,
        max_duration=args.max_duration,
        tail_margin=args.tail_margin,
        extract_frames=args.extract_frames,
        stream_fps=args.stream_fps,
        frame_max_side=args.frame_max_side,
    )


if __name__ == "__main__":
    main()
