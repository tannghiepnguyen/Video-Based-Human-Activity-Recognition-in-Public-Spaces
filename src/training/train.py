from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.models.har_net import HARNet
from src.pipeline.frame_extractor import FrameConfig, extract_video_frames
from src.pipeline.optical_flow import compute_flow_sequence
from src.pipeline.pose_estimator import PoseConfig, PoseEstimator


class VideoActivityDataset(Dataset):
    """Loads videos from data/raw/<class_name>/*.mp4 for supervised training."""

    def __init__(self, root_dir: str | Path, config: dict) -> None:
        self.root_dir = Path(root_dir)
        self.classes = config["project"]["classes"]
        self.frame_config = FrameConfig(
            frame_size=tuple(config["video"]["frame_size"]),
            sequence_length=int(config["video"]["sequence_length"]),
            sample_stride=int(config["video"]["sample_stride"]),
        )
        self.pose_estimator = PoseEstimator(PoseConfig(**config.get("pose", {})))
        self.samples: list[tuple[Path, int]] = []
        extensions = {".mp4", ".avi", ".mov", ".mkv"}

        for class_index, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.rglob("*"):
                if path.suffix.lower() in extensions:
                    self.samples.append((path, class_index))

        if not self.samples:
            raise ValueError(
                f"No videos found in {self.root_dir}. Expected folders: {', '.join(self.classes)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[index]
        frames = extract_video_frames(video_path, self.frame_config)
        if len(frames) < self.frame_config.sequence_length:
            frames = self._pad_frames(frames)

        flows = compute_flow_sequence(frames)
        poses, _ = self.pose_estimator.extract_sequence(frames)
        flow_tensor = torch.from_numpy(flows).permute(0, 3, 1, 2).float()
        pose_tensor = torch.from_numpy(poses).float()
        return flow_tensor, pose_tensor, torch.tensor(label, dtype=torch.long)

    def _pad_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if not frames:
            width, height = self.frame_config.frame_size
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            frames = [blank]
        while len(frames) < self.frame_config.sequence_length:
            frames.append(frames[-1].copy())
        return frames


def build_model(config: dict) -> HARNet:
    model_cfg = config["model"]
    pose_cfg = config["pose"]
    return HARNet(
        num_classes=len(config["project"]["classes"]),
        pose_dim=int(pose_cfg["num_landmarks"]) * int(pose_cfg["values_per_landmark"]),
        flow_channels=int(model_cfg["flow_channels"]),
        cnn_embedding_dim=int(model_cfg["cnn_embedding_dim"]),
        rnn_hidden_dim=int(model_cfg["rnn_hidden_dim"]),
        rnn_layers=int(model_cfg["rnn_layers"]),
        dropout=float(model_cfg["dropout"]),
        rnn_type=str(model_cfg["rnn_type"]),
    )


def run_epoch(
    model: HARNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for flows, poses, labels in loader:
        flows = flows.to(device)
        poses = poses.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(flows, poses)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HARNet on data/raw class folders.")
    parser.add_argument("--config", default="config/hyperparams.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="checkpoints/har_net.pt")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoActivityDataset(args.data_dir, config)
    validation_size = max(1, int(len(dataset) * 0.2)) if len(dataset) > 1 else 0
    train_size = len(dataset) - validation_size
    train_dataset, validation_dataset = (
        random_split(dataset, [train_size, validation_size])
        if validation_size
        else (dataset, None)
    )

    training_cfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
    )
    validation_loader = (
        DataLoader(validation_dataset, batch_size=int(training_cfg["batch_size"]))
        if validation_dataset is not None
        else None
    )

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        if validation_loader:
            val_loss, val_acc = run_epoch(model, validation_loader, criterion, None, device)
        else:
            val_loss, val_acc = train_loss, train_acc

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({"model_state_dict": model.state_dict(), "config": config}, output_path)


if __name__ == "__main__":
    main()
