from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
import torch
import yaml

from src.models.har_net import HARNet
from src.pipeline.frame_extractor import resize_frame
from src.pipeline.optical_flow import compute_farneback_flow, flow_magnitude
from src.pipeline.pose_estimator import PoseConfig, PoseEstimator


DEFAULT_CLASSES = ["Walking", "Running", "Standing", "Falling"]


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    fps: float
    latency_ms: float
    alert: bool


class StreamProcessor:
    """Maintains a temporal frame buffer and returns activity predictions."""

    def __init__(
        self,
        config_path: str | Path = "config/hyperparams.yaml",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.config = self._load_config(config_path)
        self.classes = self.config["project"].get("classes", DEFAULT_CLASSES)
        self.frame_size = tuple(self.config["video"].get("frame_size", [160, 160]))
        self.sequence_length = int(self.config["video"].get("sequence_length", 16))
        self.alert_threshold = float(self.config["inference"].get("falling_alert_threshold", 0.8))
        self.device = self._resolve_device(self.config["inference"].get("device", "auto"))
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.flow_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.pose_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.confidence_history: Deque[np.ndarray] = deque(
            maxlen=int(self.config["inference"].get("smoothing_window", 5))
        )
        self.previous_frame: np.ndarray | None = None
        self.pose_estimator = PoseEstimator(PoseConfig(**self.config.get("pose", {})))
        self.model = self._build_model()
        self.has_checkpoint = False

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def _load_config(self, config_path: str | Path) -> dict:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _build_model(self) -> HARNet:
        model_cfg = self.config["model"]
        pose_cfg = self.config["pose"]
        model = HARNet(
            num_classes=len(self.classes),
            pose_dim=int(pose_cfg["num_landmarks"]) * int(pose_cfg["values_per_landmark"]),
            flow_channels=int(model_cfg["flow_channels"]),
            cnn_embedding_dim=int(model_cfg["cnn_embedding_dim"]),
            rnn_hidden_dim=int(model_cfg["rnn_hidden_dim"]),
            rnn_layers=int(model_cfg["rnn_layers"]),
            dropout=float(model_cfg["dropout"]),
            rnn_type=str(model_cfg["rnn_type"]),
        )
        model.to(self.device)
        model.eval()
        return model

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.has_checkpoint = True

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, object | None, Prediction]:
        start = time.perf_counter()
        resized = resize_frame(frame, self.frame_size)
        self.frame_buffer.append(resized)

        pose_vector, pose_result = self.pose_estimator.extract(resized)
        self.pose_buffer.append(pose_vector)
        if self.previous_frame is None:
            height, width = resized.shape[:2]
            flow = np.zeros((height, width, 2), dtype=np.float32)
        else:
            flow = compute_farneback_flow(self.previous_frame, resized)
        self.flow_buffer.append(flow)
        self.previous_frame = resized

        if len(self.frame_buffer) < self.sequence_length:
            probabilities = self._cold_start_probabilities(pose_vector)
        elif self.has_checkpoint:
            probabilities = self._model_probabilities()
        else:
            probabilities = self._heuristic_probabilities()

        self.confidence_history.append(probabilities)
        smoothed = np.mean(np.stack(self.confidence_history, axis=0), axis=0)
        class_index = int(np.argmax(smoothed))
        latency_ms = (time.perf_counter() - start) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-6)
        label = self.classes[class_index]
        confidence = float(smoothed[class_index])
        prediction = Prediction(
            label=label,
            confidence=confidence,
            probabilities={name: float(value) for name, value in zip(self.classes, smoothed)},
            fps=fps,
            latency_ms=latency_ms,
            alert=label == "Falling" and confidence >= self.alert_threshold,
        )
        return resized, pose_result, prediction

    def _model_probabilities(self) -> np.ndarray:
        flows = np.stack(list(self.flow_buffer), axis=0)
        poses = np.stack(list(self.pose_buffer), axis=0)
        flow_tensor = torch.from_numpy(flows).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)
        pose_tensor = torch.from_numpy(poses).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            probabilities = self.model.predict_proba(flow_tensor, pose_tensor)[0]
        return probabilities.cpu().numpy()

    def _heuristic_probabilities(self) -> np.ndarray:
        flows = np.stack(list(self.flow_buffer), axis=0)
        motion = flow_magnitude(flows)
        fall_score = self._fall_score(self.pose_buffer[-1])

        scores = np.array(
            [
                0.45 + min(motion / 2.5, 0.35),
                max((motion - 1.8) / 5.0, 0.02),
                0.55 - min(motion / 4.0, 0.35),
                fall_score,
            ],
            dtype=np.float32,
        )
        scores = np.clip(scores, 0.01, None)
        return scores / scores.sum()

    def _cold_start_probabilities(self, pose_vector: np.ndarray) -> np.ndarray:
        scores = np.ones(len(self.classes), dtype=np.float32) * 0.1
        scores[self.classes.index("Standing")] = 0.7
        if pose_vector.any():
            scores[self.classes.index("Standing")] = 0.55
            scores[self.classes.index("Walking")] = 0.25
        return scores / scores.sum()

    def _fall_score(self, pose_vector: np.ndarray) -> float:
        if not pose_vector.any():
            return 0.05

        landmarks = pose_vector.reshape(-1, 4)
        visible = landmarks[:, 3] > 0.35
        if visible.sum() < 8:
            return 0.05

        xs = landmarks[visible, 0]
        ys = landmarks[visible, 1]
        width = max(float(xs.max() - xs.min()), 1e-6)
        height = max(float(ys.max() - ys.min()), 1e-6)
        horizontal_ratio = width / height
        return float(np.clip((horizontal_ratio - 1.0) / 1.5, 0.0, 0.9))

    def reset(self) -> None:
        self.frame_buffer.clear()
        self.flow_buffer.clear()
        self.pose_buffer.clear()
        self.confidence_history.clear()
        self.previous_frame = None

    def close(self) -> None:
        self.pose_estimator.close()


def open_video_source(source: int | str | Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video source: {source}")
    return capture
