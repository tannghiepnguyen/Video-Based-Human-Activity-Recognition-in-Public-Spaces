from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.inference.stream_processor import Prediction, StreamProcessor, open_video_source
from src.inference.visualizer import draw_pose_landmarks, draw_prediction_overlay


def summarize_predictions(predictions: list[Prediction]) -> tuple[str, dict[str, float]]:
    if not predictions:
        raise ValueError("No predictions were produced. Check that the video contains readable frames.")

    class_names = list(predictions[-1].probabilities.keys())
    averages = {
        class_name: float(np.mean([prediction.probabilities[class_name] for prediction in predictions]))
        for class_name in class_names
    }
    label = max(averages, key=averages.get)
    return label, averages


def write_csv(path: str | Path, predictions: list[Prediction]) -> None:
    if not predictions:
        return

    class_names = list(predictions[-1].probabilities.keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "label", "confidence", "fps", "latency_ms", *class_names])
        for frame_index, prediction in enumerate(predictions):
            writer.writerow(
                [
                    frame_index,
                    prediction.label,
                    f"{prediction.confidence:.6f}",
                    f"{prediction.fps:.3f}",
                    f"{prediction.latency_ms:.3f}",
                    *[f"{prediction.probabilities[class_name]:.6f}" for class_name in class_names],
                ]
            )


def validate_video(
    video_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_video: str | Path | None,
    csv_path: str | Path | None,
    max_frames: int | None,
) -> tuple[str, dict[str, float]]:
    processor = StreamProcessor(config_path=config_path, checkpoint_path=checkpoint_path)
    capture = open_video_source(video_path)
    predictions: list[Prediction] = []
    writer = None

    try:
        if output_video:
            width = int(processor.frame_size[0])
            height = int(processor.frame_size[1])
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        frame_count = 0
        while capture.isOpened():
            if max_frames is not None and frame_count >= max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            output_frame, pose_result, prediction = processor.process_frame(frame)
            predictions.append(prediction)

            if writer is not None:
                output_frame = draw_pose_landmarks(output_frame, pose_result)
                output_frame = draw_prediction_overlay(
                    output_frame,
                    prediction.label,
                    prediction.confidence,
                    prediction.fps,
                    prediction.alert,
                )
                writer.write(output_frame)

            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        processor.close()

    if csv_path:
        write_csv(csv_path, predictions)

    return summarize_predictions(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one video with a trained HAR checkpoint.")
    parser.add_argument("video", help="Path to the video file to classify.")
    parser.add_argument("--checkpoint", default="checkpoints/har_net.pt", help="Path to trained checkpoint.")
    parser.add_argument("--config", default="config/hyperparams.yaml", help="Path to model/pipeline config.")
    parser.add_argument("--output-video", default=None, help="Optional annotated MP4 output path.")
    parser.add_argument("--csv", default=None, help="Optional per-frame prediction CSV output path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick testing.")
    args = parser.parse_args()

    label, probabilities = validate_video(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_video=args.output_video,
        csv_path=args.csv,
        max_frames=args.max_frames,
    )

    print(f"Predicted action: {label}")
    for class_name, probability in probabilities.items():
        print(f"{class_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
