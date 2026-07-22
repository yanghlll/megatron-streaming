"""MultiMixQASample"""

from dataclasses import dataclass
from typing import List, Optional, Union
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import VideoData
import torch
import numpy as np


@dataclass
class MultiMixQASample(Sample):
    """Sample type for mix question answering."""

    #: The context/question for the video, image or pure text QA.
    messages: List[dict]

    #: The video data containing the image and audio info.
    video: List[VideoData] = None

    #: Streaming: absolute path(s) to the source video, decoded online at train
    #: time (no frames stored in the shard). Set by the streaming sample_loader.
    video_path: Optional[str] = None

    #: Streaming (offline-frame mode): number of pre-extracted frames per second,
    #: length == number of <|video_pad|> sentinels; sum == len(image). When set,
    #: the streaming encoder reads frames from `image` instead of decoding a video.
    bucket_counts: Optional[List[int]] = None

    #: The input image tensor in the shape (C, H, W)
    image: List[torch.Tensor] = None

    # system
    system: Optional[str] = None

    # patch positions for each image: List of np.ndarray with shape (num_patches, 3) containing [T, H, W]
    patch_positions: Optional[List[np.ndarray]] = None

    #: The frames per second of the video
    fps: Optional[Union[float, int]] = None

    #: Number of decimal places for frame timestamps (1 or 2, default 1)
    timestamp_decimal: Optional[int] = None
