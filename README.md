# Video-Based Human Activity Recognition (HAR) in Public Spaces

An end-to-end, lightweight deep learning system optimized for real-time human activity recognition (HAR) in public surveillance zones. This project utilizes a hybrid pipeline combining spatial features (skeletal pose landmarks) and temporal motion dynamics (dense optical flow) to classify four baseline human activities: **Walking, Running, Standing, and Falling**.

---

## 🚀 Key Features
*   **Multi-Stream Fusion:** Combines frame-by-frame structural joint configuration via MediaPipe with macro-motion vectors via OpenCV Dense Optical Flow.
*   **Lightweight Deep Learning:** Built on a custom feature-fused CNN-LSTM sequence architecture optimized for real-time execution ($\geq 30$ FPS).
*   **Interactive Evaluation UI:** Built-in Streamlit dashboard enabling real-time video upload testing, webcam processing, metrics reporting, and an immediate visual alarm for critical events (e.g., falling detection).

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