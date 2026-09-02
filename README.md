# 🚦 Traffic Sign AI Recognition Studio (v2.0)

A high-performance, modern AI desktop application for German Traffic Sign Recognition Benchmark (GTSRB) classification, real-time webcam highway detection, and batch multi-image reporting.

---

## ✨ Features & Upgrades

- **Modern CustomTkinter GUI**: Sleek responsive cards, high-DPI scaling, and Light / Dark / System theme switching.
- **Top-K Probability Breakdown**: Displays full confidence percentages with visual progress meters for the top 5 most likely sign classes.
- **Rich Traffic Intelligence & Rules**: Complete mapping of all 43 classes including legal categories (Speed Limit, Priority, Prohibitory, Mandatory, Warning, Derestriction) and actionable safety instructions.
- **Live Webcam Recognition HUD**: Real-time camera targeting box (Center ROI), live prediction overlay, category badge, safety advice, and FPS monitor.
- **Batch Processing & CSV Export**: Select multiple images or entire directories to generate a structured analysis table and export reports to CSV.
- **Sample Test Signs Included**: Bundled test signs in `samples/` for instant one-click testing without external datasets.
- **Clean Modular Architecture**: Separated into core engines (`model_engine`, `sign_metadata`, `webcam_engine`, `batch_processor`).

---

## 🛠️ Project Structure

```
traffic-sign-detection-with-GUI-/
├── app.py                      # Main Modern Desktop Application (CustomTkinter)
├── run.py                      # Python entrypoint launcher
├── run.bat                     # Double-click Windows launcher
├── requirements.txt            # Dependency manifest
├── README.md                   # Documentation
├── core/
│   ├── __init__.py             # Core package initialization
│   ├── model_engine.py         # Thread-safe model inference, timing, & Top-K metrics
│   ├── sign_metadata.py        # 43 classes with categories, descriptions, & rules
│   ├── webcam_engine.py        # OpenCV real-time video stream & ROI targeting
│   └── batch_processor.py      # Multi-image processing & CSV/JSON export
├── samples/                    # Pre-generated sample sign images for testing
├── traffic_classifier.keras    # Trained CNN weights (Keras 3 format)
├── traffic_classifier.h5       # Trained CNN weights (H5 format)
├── Meta.csv                    # GTSRB sign class metadata
├── Train.csv                   # Training dataset metadata
└── Test.csv                    # Test dataset metadata
```

---

## 🚀 Getting Started

### 1. Requirements & Installation
Install the necessary Python packages:
```bash
pip install -r requirements.txt
```

### 2. Launching the Application
Run via python:
```bash
python run.py
```
Or simply double-click `run.bat` on Windows.

---

## 🧭 Using the Application

### 1. Single Image Analysis
1. Switch to the **Single Image Analysis** tab.
2. Click **Upload Image** to browse for any image, or pick from the **Load Sample Sign...** dropdown.
3. View the recognized sign, confidence score, traffic regulations, and top-5 probability distribution.

### 2. Live Webcam Recognition
1. Switch to the **Live Webcam Recognition** tab.
2. Choose your camera index (default: `0`).
3. Click **Start Webcam**.
4. Hold a traffic sign in front of the camera inside the green targeting box to receive real-time classification and safety advice.

### 3. Batch Analysis & Export
1. Switch to the **Batch Analysis & Export** tab.
2. Click **Select Image Files** or **Select Folder** (e.g. select the `samples/` folder).
3. Review the summary metrics (total signs, average confidence) and the result table.
4. Click **Export CSV Report** to save your report.

---

## 🧠 Model Architecture & Dataset
- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Input Dimensions**: 30 × 30 × 3 (RGB)
- **Architecture**: Deep Convolutional Neural Network (Conv2D -> MaxPool -> Dropout -> Dense -> Softmax)
- **Classes**: 43 Traffic Sign Categories
