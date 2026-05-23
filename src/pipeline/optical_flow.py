from __future__ import annotations

import cv2
import numpy as np


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def compute_farneback_flow(previous_frame: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
    """Compute dense Farneback optical flow as an HxWx2 float32 array."""
    previous_gray = to_gray(previous_frame)
    current_gray = to_gray(current_frame)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


def compute_flow_sequence(frames: list[np.ndarray]) -> np.ndarray:
    """Create one flow map per frame, padding the first map with zeros."""
    if not frames:
        raise ValueError("At least one frame is required to compute optical flow.")

    height, width = frames[0].shape[:2]
    flows = [np.zeros((height, width, 2), dtype=np.float32)]
    for previous, current in zip(frames[:-1], frames[1:]):
        flows.append(compute_farneback_flow(previous, current))

    return np.stack(flows, axis=0)


def flow_magnitude(flow_sequence: np.ndarray) -> float:
    """Return the average motion magnitude for a flow sequence."""
    magnitude = np.linalg.norm(flow_sequence, axis=-1)
    return float(magnitude.mean())
