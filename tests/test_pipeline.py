from __future__ import annotations

import numpy as np
import torch

from src.inference.stream_processor import StreamProcessor
from src.models.har_net import HARNet
from src.pipeline.frame_extractor import FrameConfig, resize_frame, sliding_windows
from src.pipeline.optical_flow import compute_flow_sequence
from src.pipeline.pose_estimator import PoseEstimator


def test_resize_frame_dimensions() -> None:
    frame = np.zeros((120, 80, 3), dtype=np.uint8)
    resized = resize_frame(frame, (160, 160))
    assert resized.shape == (160, 160, 3)


def test_sliding_windows_count_and_shape() -> None:
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]
    windows = list(sliding_windows(frames, sequence_length=3, step=1))
    assert len(windows) == 3
    assert all(len(window) == 3 for window in windows)


def test_optical_flow_sequence_shape() -> None:
    first = np.zeros((32, 32, 3), dtype=np.uint8)
    second = first.copy()
    second[:, 4:] = first[:, :-4]
    flows = compute_flow_sequence([first, second])
    assert flows.shape == (2, 32, 32, 2)
    assert flows.dtype == np.float32


def test_pose_fallback_dimension() -> None:
    estimator = PoseEstimator()
    vector, _ = estimator.extract(np.zeros((64, 64, 3), dtype=np.uint8))
    assert vector.shape == (33 * 4,)


def test_harnet_forward_shape() -> None:
    model = HARNet(num_classes=4, pose_dim=132)
    flows = torch.zeros(2, 16, 2, 64, 64)
    poses = torch.zeros(2, 16, 132)
    logits = model(flows, poses)
    assert logits.shape == (2, 4)


def test_stream_processor_returns_prediction() -> None:
    processor = StreamProcessor()
    try:
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        output_frame, _, prediction = processor.process_frame(frame)
        assert output_frame.shape == (160, 160, 3)
        assert prediction.label in {"Walking", "Running", "Standing", "Falling"}
        assert set(prediction.probabilities) == {"Walking", "Running", "Standing", "Falling"}
    finally:
        processor.close()
