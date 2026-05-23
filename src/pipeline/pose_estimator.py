from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PoseConfig:
    num_landmarks: int = 33
    values_per_landmark: int = 4
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PoseEstimator:
    """MediaPipe Pose wrapper with a zero-landmark fallback when unavailable."""

    def __init__(self, config: PoseConfig | None = None) -> None:
        self.config = config or PoseConfig()
        self._mp_pose = None
        self._pose = None

        try:
            import mediapipe as mp  # type: ignore

            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
        except (ImportError, AttributeError):
            self._mp_pose = None
            self._pose = None

    @property
    def is_available(self) -> bool:
        return self._pose is not None

    @property
    def landmark_dim(self) -> int:
        return self.config.num_landmarks * self.config.values_per_landmark

    def extract(self, frame: np.ndarray) -> tuple[np.ndarray, object | None]:
        """Return flattened landmarks and the raw MediaPipe result."""
        if self._pose is None:
            return np.zeros(self.landmark_dim, dtype=np.float32), None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return np.zeros(self.landmark_dim, dtype=np.float32), result

        values: list[float] = []
        for landmark in result.pose_landmarks.landmark:
            values.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

        return np.asarray(values, dtype=np.float32), result

    def extract_sequence(self, frames: list[np.ndarray]) -> tuple[np.ndarray, list[object | None]]:
        landmarks = []
        results = []
        for frame in frames:
            vector, result = self.extract(frame)
            landmarks.append(vector)
            results.append(result)
        return np.stack(landmarks, axis=0), results

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
