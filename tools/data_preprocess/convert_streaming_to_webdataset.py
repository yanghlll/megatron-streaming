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


def write_config(output_dir: str) -> None:
    """Write minimal Energon config."""
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
        f.write(sample_loader_template())

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
    try:
        import cv2

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


def preprocess_row(row: dict, video_root, max_duration, tail_margin) -> dict | None:
    """JoyAI row -> {messages, video_path}. None => skip the row."""
    video_path = row.get("video_path") or (row.get("videos") or [None])[0]
    if not video_path:
        return None
    if video_root and not os.path.isabs(video_path) and not str(video_path).startswith("http"):
        video_path = os.path.join(video_root, video_path)

    duration = _ffprobe_duration(video_path)
    if duration <= 0:
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
                return None
            response_map[min(t, n_seconds - 1)] = r["content"]

    # All responses fell outside the window -> would become all-</silence> -> drop.
    if raw_responses and not response_map:
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


def _build_sample(entry: dict, idx: int, sample_prefix: str, video_root, max_duration, tail_margin):
    """Build one WebDataset sample dict, or None to skip."""
    processed = preprocess_row(entry, video_root, max_duration, tail_margin)
    if processed is None:
        return None
    sample_id = entry.get("id") or entry.get("video_name") or f"{sample_prefix}{idx}"
    sample_id = str(sample_id).replace(".", "_").replace("/", "_")
    payload = {"messages": processed["messages"], "video_path": processed["video_path"]}
    return {
        "__key__": str(sample_id),
        "json": json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    }


_COUNTER = None
_KEPT = None


def _init_worker(counter, kept) -> None:
    global _COUNTER, _KEPT
    _COUNTER = counter
    _KEPT = kept


def _process_chunk(chunk_path, tar_pattern, maxcount, maxsize, sample_prefix, worker_id,
                   video_root, max_duration, tail_margin):
    count = 0
    kept = 0
    with wds.ShardWriter(tar_pattern, maxcount=maxcount, maxsize=maxsize, verbose=0) as shard_writer:
        for idx, entry in enumerate(iter_jsonl(chunk_path)):
            sample = _build_sample(entry, idx, sample_prefix, video_root, max_duration, tail_margin)
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
    return worker_id, count, kept


def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _split_jsonl(jsonl_path: str, num_workers: int, tmp_dir: str):
    total_lines = _count_lines(jsonl_path)
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
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if written >= per_chunk and current < num_workers - 1:
                current += 1
                written = 0
            writers[current].write(line)
            written += 1
            counts[current] += 1
    for w in writers:
        w.close()
    return [p for p, c in zip(chunk_paths, counts) if c > 0], total_lines


def convert(jsonl_path, output_dir, maxcount, maxsize, num_workers, keep_chunks,
            video_root, max_duration, tail_margin):
    os.makedirs(output_dir, exist_ok=True)

    if num_workers <= 1:
        tar_pattern = os.path.join(output_dir, "instruct_%06d.tar")
        counter = mp.Value("i", 0)
        kept = mp.Value("i", 0)
        _init_worker(counter, kept)
        with tqdm(total=_count_lines(jsonl_path)) as pbar:
            _, count, n_kept = _process_chunk(
                jsonl_path, tar_pattern, maxcount, maxsize, "sample_", 0,
                video_root, max_duration, tail_margin)
            pbar.n = count
            pbar.refresh()
        print(f"total={count} kept={n_kept} skipped={count - n_kept}")
    else:
        tmp_dir = os.path.join(output_dir, ".tmp_jsonl_chunks")
        chunk_paths, total_lines = _split_jsonl(jsonl_path, num_workers, tmp_dir)
        worker_count = len(chunk_paths)
        ctx = mp.get_context("fork")
        counter = ctx.Value("i", 0)
        kept = ctx.Value("i", 0)
        with ctx.Pool(processes=worker_count, initializer=_init_worker, initargs=(counter, kept)) as pool:
            args_list = []
            for worker_id, chunk_path in enumerate(chunk_paths):
                tar_pattern = os.path.join(output_dir, f"instruct_{worker_id:02d}_%06d.tar")
                args_list.append((chunk_path, tar_pattern, maxcount, maxsize,
                                  f"sample_{worker_id:02d}_", worker_id,
                                  video_root, max_duration, tail_margin))
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
        total = sum(c for _, c, _ in results)
        total_kept = sum(k for _, _, k in results)
        print(f"total={total} kept={total_kept} skipped={total - total_kept}")
        if not keep_chunks:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    write_config(output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="Input JoyAI streaming annotation jsonl")
    p.add_argument("--output_dir", required=True, help="Output dir for WDS shards")
    p.add_argument("--video_root", default=None, help="Root dir for relative video_path")
    p.add_argument("--max_duration", type=int, default=320, help="Time-axis cap in seconds (0=off)")
    p.add_argument("--tail_margin", type=int, default=None,
                   help="Keep this many silent seconds after the last event (None=keep all)")
    p.add_argument("--maxcount", type=int, default=10000, help="Max samples per shard")
    p.add_argument("--maxsize", type=int, default=3_000_000_000, help="Max shard size in bytes")
    p.add_argument("--num_workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--keep_chunks", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        jsonl_path=args.jsonl,
        output_dir=args.output_dir,
        maxcount=args.maxcount,
        maxsize=args.maxsize,
        num_workers=args.num_workers,
        keep_chunks=args.keep_chunks,
        video_root=args.video_root,
        max_duration=args.max_duration,
        tail_margin=args.tail_margin,
    )


if __name__ == "__main__":
    main()
