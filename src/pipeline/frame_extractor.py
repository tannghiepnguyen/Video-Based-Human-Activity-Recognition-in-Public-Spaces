from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameConfig:
    frame_size: tuple[int, int] = (160, 160)
    sequence_length: int = 16
    sample_stride: int = 2


def resize_frame(frame: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Resize a BGR frame to model input dimensions."""
    width, height = frame_size
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Convert uint8 image data into float32 RGB values in [0, 1]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def sample_frames_from_capture(
    capture: cv2.VideoCapture,
    config: FrameConfig,
    normalize: bool = False,
) -> list[np.ndarray]:
    """Read a fixed-size temporal sample from an opened video capture."""
    frames: list[np.ndarray] = []
    frame_index = 0

    while len(frames) < config.sequence_length:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % config.sample_stride == 0:
            resized = resize_frame(frame, config.frame_size)
            frames.append(normalize_frame(resized) if normalize else resized)

        frame_index += 1

    return frames


def extract_video_frames(video_path: str | Path, config: FrameConfig) -> list[np.ndarray]:
    """Extract a temporally sampled frame sequence from a video file."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    try:
        return sample_frames_from_capture(capture, config)
    finally:
        capture.release()


def sliding_windows(
    frames: Iterable[np.ndarray],
    sequence_length: int,
    step: int = 1,
) -> Iterable[list[np.ndarray]]:
    """Yield fixed-length frame windows from an iterable of frames."""
    buffer: list[np.ndarray] = []
    for frame in frames:
        buffer.append(frame)
        if len(buffer) == sequence_length:
            yield list(buffer)
            del buffer[:step]
