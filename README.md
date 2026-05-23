# Video-Based Human Activity Recognition (HAR) in Public Spaces

An end-to-end, lightweight deep learning system optimized for real-time human activity recognition (HAR) in public surveillance zones. This project utilizes a hybrid pipeline combining spatial features (skeletal pose landmarks) and temporal motion dynamics (dense optical flow) to classify four baseline human activities: **Walking, Running, Standing, and Falling**.

---

## 🚀 Key Features
*   **Multi-Stream Fusion:** Combines frame-by-frame structural joint configuration via MediaPipe with macro-motion vectors via OpenCV Dense Optical Flow.
*   **Lightweight Deep Learning:** Built on a custom feature-fused CNN-LSTM sequence architecture optimized for real-time execution ($\geq 30$ FPS).
*   **Interactive Evaluation UI:** Built-in Streamlit dashboard enabling real-time video upload testing, webcam processing, metrics reporting, and an immediate visual alarm for critical events (e.g., falling detection).

---

## Core Tech Stack & Operational Constraints
* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch / PyTorch Lightning
* **Computer Vision Frameworks:** OpenCV (`cv2`), MediaPipe Pose (for ultra-lightweight skeletal feature extraction)
* **Configuration Management:** YAML (`PyYAML`)
* **Frontend Testing Dashboard:** Streamlit
* **Hardware Architecture:** Optimized to achieve real-time throughput ($\geq 30$ FPS) on consumer-grade CPUs and lightweight Edge GPUs (e.g., NVIDIA Jetson or standard laptops).

---

## Algorithmic Pipeline Architecture
The system processes video via a multi-stream hybrid approach that splits spatial, structural, and temporal vectors before combining them into a sequential classifier:

1. **Frame Extraction & Preprocessing:** Stream decompression using OpenCV, frame uniform resizing, normalization, and constant-rate temporal sampling (e.g., slicing a sliding window of $N$ sequential frames per inference step).
2. **Spatial / Structural Stream (Pose Estimation):** Compute frame-by-frame 2D or 3D coordinate landmarks ($x, y, z, \text{visibility}$) of key human skeletal joints using MediaPipe Pose. This handles structural configurations independent of background noise, making it highly effective for identifying structural transitions like *standing* vs. *falling*.
3. **Temporal Motion Stream (Optical Flow):** Calculate Dense Optical Flow (Farneback or TV-L1) across successive frame pairs. This isolates explicit velocity vectors, handling macro-movements such as distinguishing between *walking* and *running*.
4. **Sequential Deep Learning Classification (Hybrid CNN + LSTM / GRU):**
    * **Spatial Feature Extraction:** Optical flow maps are routed through a lightweight 2D CNN backbone (such as a customized MobileNetV3 or an efficient block-shrunk ResNet) to generate compact motion embeddings.
    * **Feature Fusion:** The flattened CNN motion embeddings are concatenated along the feature dimension with the normalized MediaPipe skeletal coordinate vectors[cite: 1].
    * **Temporal Modeling:** The fused vector sequences representing a fixed time-horizon window are ingested by a Recurrent Neural Network (LSTM or GRU) to model temporal dependencies[cite: 1].
    * **Output Layer:** A final linear projection layer with a Softmax activation produces class probabilities over the 4 targets: `[Walking, Running, Standing, Falling]`[cite: 1].

---

## 📁 Repository Structure
The project follows the exact architecture managed by Codex and the Antigravity framework:

```text
har-public-spaces/
├── README.md                       # This file (Setup & Execution)
├── requirements.txt                # Python package dependencies
├── config/                         # Configuration management
│   └── hyperparams.yaml            # Model & video pipeline hyper-parameters
├── data/                           # Data storage (Git ignored except .gitkeep)
│   ├── raw/                        # Original video clips grouped by class labels
│   └── processed/                  # Extracted frames, optical flows, and landmark tensors
├── src/                            # Implementation source code
│   ├── pipeline/                   # Preprocessing (Frame extraction, Optical Flow, MediaPipe)
│   ├── models/                     # Deep learning networks (CNN, RNN, Fused HarNet)
│   ├── training/                   # PyTorch Training and validation routines
│   └── inference/                  # Sequence streaming buffer and visualizer utilities
├── app/                            # Presentation Layer
│   └── app.py                      # Interactive Streamlit application interface
├── tests/                          # Automated unit test suite
└── notebooks/                      # Research sandboxes & EDA
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For CPU-only machines, install the PyTorch build that matches your environment from the official PyTorch selector if the default package resolver chooses a CUDA wheel you do not need.

---

## Run the Streamlit Demo

```bash
streamlit run app/app.py
```

The app supports uploaded video files and webcam/IP camera sources. If no trained checkpoint is supplied, the stream processor uses the implemented pose and optical-flow pipeline with a lightweight heuristic fallback so the interface can be demonstrated before training. After training, pass the checkpoint path in the sidebar for neural CNN+GRU predictions.

---

## Train

Place videos under class-named folders:

```text
data/raw/
├── Walking/
├── Running/
├── Standing/
└── Falling/
```

Then run:

```bash
python -m src.training.train --data-dir data/raw --output checkpoints/har_net.pt
```

---

## Test

```bash
pytest
```
