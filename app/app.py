from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "har_net.pt"

from components.metrics_panel import render_metrics
from src.inference.stream_processor import StreamProcessor, open_video_source
from src.inference.visualizer import draw_pose_landmarks, draw_prediction_overlay


st.set_page_config(page_title="HAR Public Spaces", layout="wide")


@st.cache_resource
def get_processor(checkpoint_path: str | None) -> StreamProcessor:
    return StreamProcessor(checkpoint_path=checkpoint_path or None)


def process_capture(capture: cv2.VideoCapture, processor: StreamProcessor, max_frames: int) -> None:
    left_margin, video_column, right_margin = st.columns([1, 5, 1])
    with video_column:
        frame_slot = st.empty()
        alert_slot = st.empty()
    metrics_slot = st.sidebar.empty()
    processed = 0

    while capture.isOpened() and processed < max_frames:
        ok, frame = capture.read()
        if not ok:
            break

        output_frame, pose_result, prediction = processor.process_frame(frame)
        output_frame = draw_pose_landmarks(output_frame, pose_result)
        output_frame = draw_prediction_overlay(
            output_frame,
            prediction.label,
            prediction.confidence,
            prediction.fps,
            prediction.alert,
        )
        frame_slot.image(
            cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )

        if prediction.alert:
            alert_slot.error("Falling detected above the configured confidence threshold.")
        else:
            alert_slot.info(f"Current action: {prediction.label}")

        with metrics_slot.container():
            render_metrics(prediction.probabilities, prediction.fps, prediction.latency_ms)

        processed += 1


def main() -> None:
    st.title("Video-Based Human Activity Recognition")
    st.caption("Walking, running, standing, and falling recognition from video streams.")

    source_type = st.sidebar.radio("Input source", ["Video file", "Webcam/IP camera"])
    checkpoint = st.sidebar.text_input(
        "Checkpoint path",
        value=str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.exists() else "",
    )
    max_frames = st.sidebar.slider("Frames to process", min_value=30, max_value=1200, value=300, step=30)
    processor = get_processor(checkpoint.strip() or None)

    if source_type == "Video file":
        uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is None:
            st.info("Upload a video clip to start inference.")
            return

        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        processor.reset()
        capture = open_video_source(tmp_path)
        try:
            process_capture(capture, processor, max_frames)
        finally:
            capture.release()
    else:
        camera_source = st.sidebar.text_input("Camera source", value="0")
        start = st.button("Start stream")
        if not start:
            st.info("Select a webcam or IP camera source, then start the stream.")
            return

        processor.reset()
        capture = open_video_source(camera_source)
        try:
            process_capture(capture, processor, max_frames)
        finally:
            capture.release()


if __name__ == "__main__":
    main()
