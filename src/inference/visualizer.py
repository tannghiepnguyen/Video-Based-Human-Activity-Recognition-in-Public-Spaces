from __future__ import annotations

import cv2
import numpy as np


CLASS_COLORS = {
    "Walking": (0, 180, 255),
    "Running": (0, 120, 255),
    "Standing": (80, 220, 80),
    "Falling": (40, 40, 255),
}


def draw_prediction_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    fps: float,
    alert: bool = False,
) -> np.ndarray:
    canvas = frame.copy()
    color = CLASS_COLORS.get(label, (255, 255, 255))
    height, width = canvas.shape[:2]

    cv2.rectangle(canvas, (0, 0), (width, 72), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        f"{label}  {confidence * 100:.1f}%",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    if alert:
        cv2.rectangle(canvas, (0, height - 58), (width, height), (0, 0, 180), thickness=-1)
        cv2.putText(
            canvas,
            "FALLING ALERT",
            (16, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def draw_pose_landmarks(frame: np.ndarray, pose_result: object | None) -> np.ndarray:
    """Draw MediaPipe landmarks if MediaPipe is installed and produced a result."""
    if pose_result is None or not getattr(pose_result, "pose_landmarks", None):
        return frame

    try:
        import mediapipe as mp  # type: ignore

        canvas = frame.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            canvas,
            pose_result.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
        )
        return canvas
    except (ImportError, AttributeError):
        return frame
